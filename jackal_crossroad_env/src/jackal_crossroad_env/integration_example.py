#!/usr/bin/env python3
"""
Example: How to integrate preprocessing modules into the CrossroadEnv

The preprocessing modules would be applied at two levels:
1. Within the environment (for observation preprocessing)
2. Within the RL agent/policy network (for feature extraction)
"""

import torch
import numpy as np
from preprocessing import create_preprocessing_pipeline


# ============================================================================
# APPROACH 1: Integration within Environment (Gym wrapper)
# ============================================================================

class PreprocessedObservationWrapper:
    """
    Wrapper that applies neural network preprocessing to raw observations.
    This would wrap around CrossroadEnv to provide preprocessed observations.
    """
    
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        
        # Create preprocessing pipeline
        self.pipeline = create_preprocessing_pipeline(use_camera=True, use_lidar=True)
        
        # Move models to device
        for module in self.pipeline.values():
            module.to(device)
            module.eval()  # Set to evaluation mode
        
        # Update observation space to be the fused features
        # Original space was (19,), now it's (64,) from fusion module
        from gym import spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(64,),  # fusion module output_dim
            dtype=np.float32
        )
        
        self.action_space = env.action_space
    
    def _preprocess_observation(self, raw_obs):
        """
        Convert raw observations to preprocessed features.
        
        Raw obs structure (from CrossroadEnv._get_obs):
        - [0:10]: laser_scan (10 rays)
        - [10:12]: position (x, y)
        - [12:14]: velocity (x, y)
        - [14]: yaw
        - [15]: angular_velocity
        - [16]: distance_to_goal
        - [17]: angle_to_goal
        - [18]: traffic_light
        """
        with torch.no_grad():
            # Extract components
            lidar = raw_obs[0:10]
            robot_location = raw_obs[10:12]  # Current position
            velocity = raw_obs[12:14]
            yaw = raw_obs[14]
            angular_velocity = raw_obs[15]
            traffic_light = raw_obs[18]
            
            # Get goal location from environment
            goal_location = self.env.goal_position  # [x, y]
            
            # Additional state: velocity, yaw, angular_velocity, traffic_light
            additional_state = np.array([
                velocity[0], velocity[1], yaw, angular_velocity, traffic_light
            ])
            
            # Get camera image from environment's raw observation
            # Note: Need to modify CrossroadEnv to expose raw camera image
            camera_image = self._get_camera_image()  # Shape: (H, W, 3) BGR
            
            # Convert to tensors and add batch dimension
            lidar_tensor = torch.FloatTensor(lidar).unsqueeze(0).to(self.device)
            robot_loc_tensor = torch.FloatTensor(robot_location).unsqueeze(0).to(self.device)
            goal_loc_tensor = torch.FloatTensor(goal_location).unsqueeze(0).to(self.device)
            state_tensor = torch.FloatTensor(additional_state).unsqueeze(0).to(self.device)
            
            # Preprocess camera image
            # Convert BGR to RGB, resize, normalize, convert to tensor
            camera_tensor = self._preprocess_camera(camera_image)
            camera_tensor = camera_tensor.unsqueeze(0).to(self.device)  # Add batch dim
            
            # Apply preprocessing networks
            image_features = self.pipeline['image_preprocessor'](camera_tensor)
            lidar_features = self.pipeline['lidar_preprocessor'](lidar_tensor)
            
            # Fuse all features
            fused_features = self.pipeline['fusion_module'](
                image_features, 
                lidar_features,
                robot_loc_tensor,
                goal_loc_tensor,
                state_tensor
            )
            
            # Convert back to numpy
            return fused_features.cpu().numpy().squeeze(0)
    
    def _preprocess_camera(self, image):
        """Convert BGR numpy image to preprocessed RGB tensor"""
        import cv2
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to EfficientNet input size (224x224)
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_resized / 255.0 - mean) / std
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.FloatTensor(image_normalized).permute(2, 0, 1)
        
        return image_tensor
    
    def _get_camera_image(self):
        """Get raw camera image from environment"""
        # Access the stored raw camera image from environment
        if hasattr(self.env, '_raw_camera_image'):
            return self.env.bridge.imgmsg_to_cv2(
                self.env._raw_camera_image, 
                desired_encoding='bgr8'
            )
        else:
            # Return dummy image if not available
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def reset(self):
        raw_obs = self.env.reset()
        return self._preprocess_observation(raw_obs)
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        preprocessed_obs = self._preprocess_observation(raw_obs)
        return preprocessed_obs, reward, done, info
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def close(self):
        return self.env.close()


# ============================================================================
# APPROACH 2: Integration within RL Agent/Policy
# ============================================================================

class PolicyNetworkWithPreprocessing(torch.nn.Module):
    """
    RL policy network that includes preprocessing modules.
    The environment provides raw observations, and the policy does preprocessing.
    """
    
    def __init__(self, action_dim=2, device='cpu'):
        super().__init__()
        
        # Create preprocessing modules
        self.pipeline = create_preprocessing_pipeline(use_camera=True, use_lidar=True)
        
        # Policy head (takes fused features and outputs action)
        fusion_output_dim = 64
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(fusion_output_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
            torch.nn.Tanh()  # Scale actions to [-1, 1]
        )
        
        # Value head (for actor-critic algorithms)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(fusion_output_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
        self.device = device
        self.to(device)
    
    def forward(self, raw_obs_dict):
        """
        Forward pass through preprocessing and policy networks.
        
        Args:
            raw_obs_dict: Dictionary containing:
                - 'camera': (batch, 3, 224, 224) tensor
                - 'lidar': (batch, 10) tensor
                - 'robot_location': (batch, 2) tensor
                - 'goal_location': (batch, 2) tensor
                - 'state': (batch, 5) tensor (velocity, yaw, angular_vel, traffic_light)
        
        Returns:
            action: (batch, action_dim) tensor
            value: (batch, 1) tensor
        """
        # Preprocess sensor data
        image_features = self.pipeline['image_preprocessor'](raw_obs_dict['camera'])
        lidar_features = self.pipeline['lidar_preprocessor'](raw_obs_dict['lidar'])
        
        # Fuse features
        fused = self.pipeline['fusion_module'](
            image_features,
            lidar_features,
            raw_obs_dict['robot_location'],
            raw_obs_dict['goal_location'],
            raw_obs_dict['state']
        )
        
        # Compute action and value
        action = self.policy_head(fused)
        value = self.value_head(fused)
        
        return action, value


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_approach_1():
    """
    Example: Using the wrapper approach
    This is simpler and keeps preprocessing separate from the agent
    """
    from crossroad_task_env import CrossroadEnv
    
    # Create base environment
    env = CrossroadEnv()
    
    # Wrap with preprocessing
    env = PreprocessedObservationWrapper(env, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Now use with any RL algorithm (PPO, SAC, etc.)
    obs = env.reset()
    print(f"Preprocessed observation shape: {obs.shape}")  # Should be (64,)
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if done:
            obs = env.reset()


def example_approach_2():
    """
    Example: Using policy with integrated preprocessing
    This is more flexible for end-to-end training
    """
    from crossroad_task_env import CrossroadEnv
    
    # Create environment (provides raw observations)
    env = CrossroadEnv()
    
    # Create policy with preprocessing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PolicyNetworkWithPreprocessing(action_dim=2, device=device)
    
    # Training loop
    raw_obs = env.reset()
    
    for _ in range(100):
        # Convert raw observation to dict format for policy
        obs_dict = {
            'camera': torch.FloatTensor(dummy_camera_image).unsqueeze(0).to(device),
            'lidar': torch.FloatTensor(raw_obs[0:10]).unsqueeze(0).to(device),
            'robot_location': torch.FloatTensor(raw_obs[10:12]).unsqueeze(0).to(device),
            'goal_location': torch.FloatTensor(env.goal_position).unsqueeze(0).to(device),
            'state': torch.FloatTensor([
                raw_obs[12], raw_obs[13], raw_obs[14], raw_obs[15], raw_obs[18]
            ]).unsqueeze(0).to(device)
        }
        
        # Get action from policy
        with torch.no_grad():
            action, value = policy(obs_dict)
        
        # Execute in environment
        action_np = action.cpu().numpy().squeeze(0)
        raw_obs, reward, done, info = env.step(action_np)
        
        if done:
            raw_obs = env.reset()


# ============================================================================
# WHICH APPROACH TO USE?
# ============================================================================

"""
APPROACH 1 (Wrapper): 
- Pros: Clean separation, easier to debug, works with any RL library
- Cons: Preprocessing runs during environment interaction (can't parallelize)
- Best for: Simple setups, when using existing RL libraries

APPROACH 2 (Integrated Policy):
- Pros: End-to-end training, preprocessing gradients can flow, more flexible
- Cons: Need to modify training loop, tighter coupling
- Best for: Custom training loops, when you want gradients through preprocessing

RECOMMENDATION: Start with Approach 1, move to Approach 2 if you need 
end-to-end gradient flow or have specific architectural needs.
"""


if __name__ == "__main__":
    print("See examples above for how to integrate preprocessing modules")
    print("\nKey integration points:")
    print("1. Lidar preprocessing: Takes raw 10 lidar rays -> 16D features")
    print("2. Image preprocessing: Takes camera image -> 128D features") 
    print("3. Fusion module: Combines everything -> 64D final features")
    print("4. Robot location (x,y) and goal location (x,y) are explicit inputs")
