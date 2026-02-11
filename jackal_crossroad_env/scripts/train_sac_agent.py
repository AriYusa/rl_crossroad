#!/usr/bin/env python3
"""
SAC Training Script for Jackal Crossroad Environment
Uses Stable Baselines3 SAC with custom multi-modal feature extractor
Includes wandb logging, checkpointing, and hyperparameter configuration
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import rospy
import gym

# Stable Baselines 3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Weights & Biases
import wandb
from wandb.integration.sb3 import WandbCallback

# Local imports
import jackal_crossroad_env  # Registers the environment
from feature_extractor import (
    MultiModalFeatureExtractor,
    LightweightMultiModalExtractor,
)


# ==================== Default Configuration ====================
DEFAULT_CONFIG = {
    # Environment
    "env_id": "JackalCrossroad-v0",
    
    # SAC Hyperparameters
    "learning_rate": 3e-4,
    "buffer_size": 10_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,  # Soft update coefficient
    "gamma": 0.99,  # Discount factor
    "train_freq": 1,  # Update policy every n steps
    "gradient_steps": 1,  # Gradient steps per update
    "ent_coef": "auto",  # Entropy coefficient (auto-tuned)
    "target_entropy": "auto",  # Target entropy (auto-calculated)
    "target_update_interval": 1,
    "use_sde": False,  # State Dependent Exploration
    "sde_sample_freq": -1,
    
    # Network Architecture
    "use_lightweight_extractor": False,  # Use lightweight (no image) extractor
    "image_features_dim": 256,
    "lidar_features_dim": 64,
    "coord_features_dim": 32,
    "combined_features_dim": 256,
    "net_arch": [256, 256],  # Policy and value network hidden layers
    
    # Training
    "total_timesteps": 500_000,
    "eval_freq": 10_000,
    "n_eval_episodes": 5,
    "checkpoint_freq": 25_000,
    
    # Optimization
    "gradient_clip": 0.5,  # Max gradient norm
    "optimizer_eps": 1e-5,  # Adam optimizer epsilon
    
    # Logging
    "use_wandb": True,
    "wandb_project": "jackal-crossroad-sac",
    "wandb_entity": None,  # Your wandb username/team
    "log_interval": 10,
    
    # Paths
    "save_dir": "./sac_checkpoints",
    "log_dir": "./sac_logs",
    
    # Reproducibility
    "seed": 42,
    
    # Device
    "device": "auto",  # "auto", "cuda", "cpu"
}


# ==================== Custom Callbacks ====================
class WandbMetricsCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to wandb.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Log SAC-specific metrics
        if hasattr(self.model, 'ent_coef') and self.model.ent_coef is not None:
            if isinstance(self.model.ent_coef, torch.Tensor):
                ent_coef = self.model.ent_coef.item()
            else:
                ent_coef = self.model.ent_coef
            wandb.log({
                "train/entropy_coefficient": ent_coef,
                "train/timestep": self.num_timesteps,
            })
        
        # Check for episode completion
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # Check success (goal reached)
                    success = ep_reward > 50  # Assuming goal reward > 50
                    self.episode_successes.append(success)
                    
                    wandb.log({
                        "episode/reward": ep_reward,
                        "episode/length": ep_length,
                        "episode/success": int(success),
                        "episode/success_rate_100": np.mean(self.episode_successes[-100:]) if len(self.episode_successes) > 0 else 0,
                        "episode/avg_reward_100": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0,
                        "train/timestep": self.num_timesteps,
                    })
        
        return True


class MemoryProfileCallback(BaseCallback):
    """
    Callback to profile memory usage during training.
    """
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            memory_stats = {}
            
            # GPU memory
            if torch.cuda.is_available():
                memory_stats["memory/gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                memory_stats["memory/gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
                memory_stats["memory/gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            
            # Replay buffer size
            if hasattr(self.model, 'replay_buffer'):
                buffer = self.model.replay_buffer
                memory_stats["memory/replay_buffer_size"] = buffer.size()
                memory_stats["memory/replay_buffer_pos"] = buffer.pos
            
            if memory_stats:
                wandb.log({**memory_stats, "train/timestep": self.num_timesteps})
                
        return True


class GradientMonitorCallback(BaseCallback):
    """
    Callback to monitor gradient norms and detect training issues.
    """
    
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            grad_stats = {}
            
            # Actor gradients
            if hasattr(self.model, 'actor') and self.model.actor is not None:
                actor_grad_norm = self._get_grad_norm(self.model.actor)
                if actor_grad_norm is not None:
                    grad_stats["gradients/actor_grad_norm"] = actor_grad_norm
            
            # Critic gradients
            if hasattr(self.model, 'critic') and self.model.critic is not None:
                critic_grad_norm = self._get_grad_norm(self.model.critic)
                if critic_grad_norm is not None:
                    grad_stats["gradients/critic_grad_norm"] = critic_grad_norm
            
            if grad_stats:
                wandb.log({**grad_stats, "train/timestep": self.num_timesteps})
                
        return True
    
    def _get_grad_norm(self, model):
        """Calculate total gradient norm for a model."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm if total_norm > 0 else None


# ==================== SAC Training Functions ====================
def create_env(config):
    """Create and wrap the environment."""
    env = gym.make(config["env_id"])
    env = Monitor(env)
    return env


def create_policy_kwargs(config, observation_space):
    """Create policy keyword arguments with custom feature extractor."""
    
    if config["use_lightweight_extractor"]:
        features_extractor_class = LightweightMultiModalExtractor
        features_extractor_kwargs = {
            "features_dim": config["combined_features_dim"],
        }
    else:
        features_extractor_class = MultiModalFeatureExtractor
        features_extractor_kwargs = {
            "image_features_dim": config["image_features_dim"],
            "lidar_features_dim": config["lidar_features_dim"],
            "coord_features_dim": config["coord_features_dim"],
            "combined_features_dim": config["combined_features_dim"],
        }
    
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": features_extractor_kwargs,
        "net_arch": config["net_arch"],
        "optimizer_kwargs": {
            "eps": config["optimizer_eps"],
        },
    }
    
    return policy_kwargs


def create_sac_model(env, config):
    """Create SAC model with custom configuration."""
    
    policy_kwargs = create_policy_kwargs(config, env.observation_space)
    
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        gamma=config["gamma"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        ent_coef=config["ent_coef"],
        target_update_interval=config["target_update_interval"],
        target_entropy=config["target_entropy"],
        use_sde=config["use_sde"],
        sde_sample_freq=config["sde_sample_freq"],
        policy_kwargs=policy_kwargs,
        replay_buffer_kwargs=dict(optimize_memory_usage=True),
        verbose=1,
        seed=config["seed"],
        device=config["device"],
        tensorboard_log=config["log_dir"],
    )
    
    return model


def setup_callbacks(config, env):
    """Setup training callbacks."""
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=config["save_dir"],
        name_prefix="sac_jackal",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Wandb callbacks
    if config["use_wandb"]:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"{config['save_dir']}/wandb_models",
            verbose=2,
        )
        callbacks.append(wandb_callback)
        callbacks.append(WandbMetricsCallback())
        callbacks.append(MemoryProfileCallback(log_freq=1000))
        callbacks.append(GradientMonitorCallback(log_freq=100))
    
    return CallbackList(callbacks)


def train_sac(config):
    """Main SAC training function."""
    
    # Create directories
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if config["use_wandb"]:
        run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            name=f"sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    
    try:
        # Create environment
        rospy.loginfo("Creating environment...")
        env = create_env(config)
        rospy.loginfo(f"Observation space: {env.observation_space}")
        rospy.loginfo(f"Action space: {env.action_space}")
        
        # Create model
        rospy.loginfo("Creating SAC model...")
        model = create_sac_model(env, config)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.policy.parameters())
        rospy.loginfo(f"Total model parameters: {total_params:,}")
        
        # Setup callbacks
        callbacks = setup_callbacks(config, env)
        
        # Train
        rospy.loginfo(f"Starting training for {config['total_timesteps']} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=config["log_interval"],
            progress_bar=True,
        )
        
        training_time = time.time() - start_time
        rospy.loginfo(f"Training completed in {training_time/3600:.2f} hours")
        
        # Save final model
        final_model_path = f"{config['save_dir']}/sac_jackal_final"
        model.save(final_model_path)
        rospy.loginfo(f"Final model saved to {final_model_path}")
        
        # Save replay buffer
        buffer_path = f"{config['save_dir']}/sac_replay_buffer_final"
        model.save_replay_buffer(buffer_path)
        rospy.loginfo(f"Replay buffer saved to {buffer_path}")
        
    finally:
        if config["use_wandb"]:
            wandb.finish()
        env.close()
    
    return model


def load_config(config_path=None):
    """Load configuration from YAML file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)
        rospy.loginfo(f"Loaded config from {config_path}")
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train SAC agent for Jackal Crossroad")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--lightweight", action="store_true", help="Use lightweight extractor (no image)")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto/cuda/cpu)")
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('sac_training', anonymous=True, log_level=rospy.INFO)
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.no_wandb:
        config["use_wandb"] = False
    if args.lightweight:
        config["use_lightweight_extractor"] = True
    if args.timesteps:
        config["total_timesteps"] = args.timesteps
    if args.seed:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    
    # Set seeds for reproducibility
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("SAC Training Configuration")
    rospy.loginfo("=" * 60)
    for key, value in config.items():
        rospy.loginfo(f"  {key}: {value}")
    rospy.loginfo("=" * 60)
    
    # Train
    model = train_sac(config)
    
    rospy.loginfo("Training complete!")


if __name__ == "__main__":
    main()
