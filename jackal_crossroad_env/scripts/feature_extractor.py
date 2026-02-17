#!/usr/bin/env python3
"""
Custom Feature Extractor for Multi-Modal Observations
Processes: image + lidar + robot_coords + goal_coords for SAC

Supports pretrained backbones (MobileNetV3, EfficientNet) with optional freezing.
"""

from pyexpat import features
import torch
import torch.nn as nn
import numpy as np
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Optional
import torchvision.models as models
from torchvision import transforms


class PretrainedImageEncoder(nn.Module):
    """
    MobileNetV3-Small pretrained image encoder with optional freezing.
    
    - 2.5M parameters, very fast
    - Pretrained on ImageNet
    - Recommended to keep frozen for robotics tasks
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        freeze_backbone: bool = True,
        freeze_bn: bool = True,
    ):
        """
        Initialize MobileNetV3-Small pretrained encoder.
        
        Args:
            output_dim: Output feature dimension
            freeze_backbone: If True, freeze all backbone weights (recommended)
            freeze_bn: If True, freeze BatchNorm layers (recommended when frozen)
        """
        super().__init__()
        
        self.backbone_name = 'mobilenet_v3_small'
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = freeze_bn
        
        # Load MobileNetV3-Small with ImageNet pretrained weights
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.backbone = models.mobilenet_v3_small(weights=weights)
        backbone_out_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        
        # Projection head (always trainable)
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[PretrainedImageEncoder] Frozen {self.backbone_name} backbone")
    
    def unfreeze_backbone(self, unfreeze_last_n_layers: int = -1):
        """
        Unfreeze backbone for fine-tuning.
        
        Args:
            unfreeze_last_n_layers: Number of layers to unfreeze from the end.
                                    -1 = unfreeze all layers
        """
        if unfreeze_last_n_layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"[PretrainedImageEncoder] Unfrozen all {self.backbone_name} layers")
        else:
            # Get all named parameters
            params = list(self.backbone.named_parameters())
            # Unfreeze last N
            for name, param in params[-unfreeze_last_n_layers:]:
                param.requires_grad = True
            print(f"[PretrainedImageEncoder] Unfrozen last {unfreeze_last_n_layers} layers")
    
    def train(self, mode: bool = True):
        """Override train to handle frozen BatchNorm."""
        super().train(mode)
        if self.freeze_backbone and self.freeze_bn:
            # Keep BN in eval mode when backbone is frozen
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eval()
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Image tensor (B, C, H, W) in [0, 1] range
            
        Returns:
            Features tensor (B, output_dim)
        """
        # Normalize with ImageNet stats
        x = (x - self.mean) / self.std
        
        # Backbone features
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
            
        # Project to output dimension
        return self.projection(features)


class MultiModalFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observation space containing:
    - raw_image: (120, 160, 3) RGB camera image
    - laser_scan: (20,) LIDAR ranges
    - robot_coords: (3,) [x, y, yaw]
    - goal_coords: (2,) [x, y]
    
    Architecture:
    - CNN pretrained for image processing
    - MLP for lidar processing
    - MLP for coordinate processing
    - Fusion layer to combine all features
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        image_features_dim: int = 256,
        lidar_features_dim: int = 64,
        coord_features_dim: int = 32,
        combined_features_dim: int = 256,
    ):
        """
        Initialize the multi-modal feature extractor.
        
        Args:
            observation_space: Dict observation space
            image_features_dim: Output dimension of image encoder
            lidar_features_dim: Output dimension of lidar MLP
            coord_features_dim: Output dimension of coordinate MLP
            combined_features_dim: Final combined feature dimension
        """
        # Calculate total features dimension
        super().__init__(observation_space, features_dim=combined_features_dim)
        
        self.image_features_dim = image_features_dim
        self.lidar_features_dim = lidar_features_dim
        self.coord_features_dim = coord_features_dim
        
        # Get observation shapes
        image_shape = observation_space['raw_image'].shape  # (120, 160, 3)
        lidar_shape = observation_space['laser_scan'].shape  # (20,)
        robot_coord_shape = observation_space['robot_coords'].shape  # (3,)
        goal_coord_shape = observation_space['goal_coords'].shape  # (2,)
        
        # ==================== Image Encoder ====================
        #  Pretrained frozen model
        self.image_encoder = PretrainedImageEncoder(
                output_dim=image_features_dim,
        )

        # ==================== Lidar MLP ====================
        # Input: (batch, 20)
        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, lidar_features_dim),
            nn.ReLU(),
        )
        
        # ==================== Coordinate MLP ====================
        # Input: robot_coords (3) + goal_coords (2) = 5
        coord_input_dim = robot_coord_shape[0] + goal_coord_shape[0]
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, coord_features_dim),
            nn.ReLU(),
        )
        
        # ==================== Fusion Layer ====================
        # Combine all features
        fusion_input_dim = image_features_dim + lidar_features_dim + coord_features_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, combined_features_dim),
            nn.ReLU(),
            nn.Linear(combined_features_dim, combined_features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            observations: Dict containing all observation tensors
            
        Returns:
            Combined feature tensor of shape (batch, combined_features_dim)
        """
        # Process image robustly: accept NHWC or NCHW.
        image = observations['raw_image']       # to prevent shape-mismatch crash
        if image.dim() == 3:
            image = image.unsqueeze(0)

        if image.dim() != 4:    # crash handling
            raise ValueError(f"Expected 4D image tensor, got shape {tuple(image.shape)}")

        # if channel-last (B, H, W, C), move to channel-first (B, C, H, W).
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        elif image.shape[1] != 3:
            raise ValueError(f"Unsupported image layout for raw_image: {tuple(image.shape)}")

        image = image.float()
        # normalize only when needed.
        if image.max() > 1.0:
            image = image / 255.0
        
        image = nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_features = self.image_encoder(image)
        
        # Process lidar
        lidar = observations['laser_scan'].float()
        if lidar.dim() == 1:
            lidar = lidar.unsqueeze(0)
        lidar_features = self.lidar_mlp(lidar)
        
        # Process coordinates
        robot_coords = observations['robot_coords'].float()
        goal_coords = observations['goal_coords'].float()
        if robot_coords.dim() == 1:
            robot_coords = robot_coords.unsqueeze(0)
        if goal_coords.dim() == 1:
            goal_coords = goal_coords.unsqueeze(0)
        coords = torch.cat([robot_coords, goal_coords], dim=-1)
        coord_features = self.coord_mlp(coords)
        
        # Fuse all features
        combined = torch.cat([image_features, lidar_features, coord_features], dim=-1)
        return self.fusion(combined)
    
    def get_trainable_params(self):
        """Get count of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        return {'trainable': trainable, 'frozen': frozen, 'total': total}
