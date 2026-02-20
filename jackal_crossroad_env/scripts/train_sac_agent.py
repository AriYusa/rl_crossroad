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
import json
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import torch
import rospy
import gym
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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

# Weights & Biases
import wandb
from wandb.integration.sb3 import WandbCallback

# Local imports
import jackal_crossroad_env  # Registers the environment
from feature_extractor import (
    MultiModalFeatureExtractor,
)
from video_recorder import VideoRecorderCallback


# ==================== Default Configuration ====================
DEFAULT_CONFIG = {
    # Environment
    "env_id": "JackalCrossroad-v0",
    
    # SAC Hyperparameters
    "learning_rate": 3e-4,
    "buffer_size": 50_000,
    "learning_starts": 2_000,
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
    "image_features_dim": 256,
    "lidar_features_dim": 64,
    "coord_features_dim": 32,
    "combined_features_dim": 256,
    "net_arch": [256, 256],  # Policy and value network hidden layers
    
    # Training
    "total_timesteps": 500_000,
    "manual_training_loop": False,
    "eval_freq": 1_000,
    "n_eval_episodes": 5,
    "eval_deterministic": True,
    "checkpoint_freq": 1_000,
    "checkpoint_name_prefix": "sac_jackal_newGen",
    "replay_buffer_name_prefix": "sac_replay_buffer_newGen",
    "checkpoint_save_replay_buffer": True,
    
    # Optimization
    "gradient_clip": 0.5,  # Max gradient norm
    "optimizer_eps": 1e-5,  # Adam optimizer epsilon
    
    # Logging
    "use_wandb": True,
    "wandb_project": "jackal-crossroad-sac",
    "wandb_entity": "aau-uni-rl",  # Your wandb username/team
    "wandb_run_id": None,  # Resume run ID (optional)
    "log_interval": 10,
    "metrics_log_freq": 500,
    "enable_metric_visualization": True,
    "visualization_dir": "./sac_logs/visualizations",
    
    # Video Recording
    "record_video": False,
    "video_episode_freq": 100,  # Record video every N episodes
    
    # Paths
    "save_dir": "./sac_checkpoints",
    "log_dir": "./sac_logs",
    "video_dir": "./sac_videos",
    "resume_path": None,  # Path to checkpoint to resume from
    
    # Reproducibility
    "seed": 42,
    
    # Device
    "device": "auto",  # "auto", "cuda", "cpu"
}


# ==================== Custom Callbacks ====================
class WandbMetricsCallback(BaseCallback):
    """
    Callback for logging rewards, losses and training metrics.

    Logs to ROS console always and to wandb optionally.
    """
    
    def __init__(
        self,
        use_wandb=True,
        log_freq=500,
        visualization_enabled=True,
        visualization_dir="./sac_logs/visualizations",
        verbose=0,
        initial_episode_count=0,
    ):
        super().__init__(verbose)
        self.use_wandb = use_wandb
        self.log_freq = log_freq
        self.visualization_enabled = visualization_enabled
        self.visualization_dir = Path(visualization_dir)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_illegal_crossings = []
        self.training_metric_history = []
        self.episode_metric_history = []
        self.episode_count = initial_episode_count
        self.episode_terminations = {
            'collision': 0,
            'goal_reached': 0,
            'red_light_violation': 0,
            'off_crosswalk': 0,
            'goal_timeout': 0,
            'max_steps': 0,
        }
    
    def _safe_float(self, value):
        """Convert numeric-like values to float when possible."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return value
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            metrics = {"train/timestep": self.num_timesteps}
            
            # Entropy coefficient
            if hasattr(self.model, 'ent_coef') and self.model.ent_coef is not None:
                ent_coef = self._safe_float(self.model.ent_coef)
                metrics["train/entropy_coefficient"] = ent_coef

            # SAC losses and other values captured by SB3 logger
            logger_values = getattr(getattr(self, "logger", None), "name_to_value", {})
            tracked_keys = [
                "train/actor_loss",
                "train/critic_loss",
                "train/ent_coef_loss",
                "train/ent_coef",
                "train/learning_rate",
                "train/n_updates",
                "rollout/ep_len_mean",
                "rollout/ep_rew_mean",
                "eval/mean_reward",
                "eval/mean_ep_length",
                "eval/success_rate",
                "time/fps",
            ]
            for key in tracked_keys:
                if key in logger_values:
                    metrics[key] = self._safe_float(logger_values[key])

            self.training_metric_history.append(dict(metrics))

            if self.use_wandb and metrics:
                wandb.log(metrics)

            summary_parts = [f"t={self.num_timesteps}"]
            if "rollout/ep_rew_mean" in metrics:
                summary_parts.append(f"ep_rew_mean={metrics['rollout/ep_rew_mean']:.2f}")
            if "train/actor_loss" in metrics:
                summary_parts.append(f"actor_loss={metrics['train/actor_loss']:.4f}")
            if "train/critic_loss" in metrics:
                summary_parts.append(f"critic_loss={metrics['train/critic_loss']:.4f}")
            if "train/entropy_coefficient" in metrics:
                summary_parts.append(f"ent_coef={metrics['train/entropy_coefficient']:.4f}")
            if "eval/mean_reward" in metrics:
                summary_parts.append(f"eval_rew={metrics['eval/mean_reward']:.2f}")
            rospy.loginfo("[train_metrics] " + " | ".join(summary_parts))
        
        # Check for episode completion
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_count += 1
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    # Track termination reason
                    done_reason = info.get('done_reason', 'unknown')
                    if done_reason in self.episode_terminations:
                        self.episode_terminations[done_reason] += 1

                    # Goal reached (explicit, not reward heuristic)
                    goal_reached = done_reason == 'goal_reached'
                    self.episode_successes.append(goal_reached)

                    goal_reach_rate_100 = (
                        np.mean(self.episode_successes[-100:])
                        if len(self.episode_successes) > 0 else 0
                    )
                    goal_reach_rate_all = (
                        self.episode_terminations['goal_reached']
                        / max(1, sum(self.episode_terminations.values()))
                    )
                    illegal_crossing = done_reason in ('red_light_violation', 'off_crosswalk')
                    self.episode_illegal_crossings.append(illegal_crossing)
                    illegal_crossing_rate_100 = (
                        np.mean(self.episode_illegal_crossings[-100:])
                        if len(self.episode_illegal_crossings) > 0 else 0
                    )
                    illegal_crossing_rate_all = (
                        (self.episode_terminations['red_light_violation'] + self.episode_terminations['off_crosswalk'])
                        / max(1, sum(self.episode_terminations.values()))
                    )

                    episode_metrics = {
                        "episode/number": self.episode_count,
                        "episode/reward": ep_reward,
                        "episode/length": ep_length,
                        "episode/success": int(goal_reached),
                        "episode/success_rate_100": goal_reach_rate_100,
                        "episode/goal_reached": int(goal_reached),
                        "episode/goal_reach_rate_100": goal_reach_rate_100,
                        "episode/goal_reach_rate_all": goal_reach_rate_all,
                        "episode/illegal_crossing": int(illegal_crossing),
                        "episode/illegal_crossing_rate_100": illegal_crossing_rate_100,
                        "episode/illegal_crossing_rate_all": illegal_crossing_rate_all,
                        "episode/avg_reward_100": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0,
                        f"episode/termination/{done_reason}": 1,
                        "episode/termination/collision_rate": self.episode_terminations['collision'] / max(1, sum(self.episode_terminations.values())),
                        "episode/termination/goal_reached_rate": goal_reach_rate_all,
                        "episode/termination/violation_rate": illegal_crossing_rate_all,
                        "train/timestep": self.num_timesteps,
                    }

                    if "episode_path_length" in info:
                        episode_metrics["episode/path_length"] = self._safe_float(info.get("episode_path_length"))
                    if "optimal_path_length" in info:
                        episode_metrics["episode/optimal_path_length"] = self._safe_float(info.get("optimal_path_length"))
                    if "path_efficiency" in info:
                        episode_metrics["episode/path_efficiency"] = self._safe_float(info.get("path_efficiency"))

                    self.episode_metric_history.append(dict(episode_metrics))

                    if self.use_wandb:
                        wandb.log(episode_metrics)

                    rospy.loginfo(
                        "[episode] #{} | reward={:.2f} | length={} | goal_reached={} | goal_reach_rate_100={:.2f} | illegal_crossing_rate_100={:.2f} | path_eff={:.3f} | done_reason={}".format(
                            self.episode_count,
                            ep_reward,
                            ep_length,
                            int(goal_reached),
                            goal_reach_rate_100,
                            illegal_crossing_rate_100,
                            float(episode_metrics.get("episode/path_efficiency", 0.0)),
                            done_reason,
                        )
                    )
        
        return True

    def _on_training_end(self) -> None:
        if not self.visualization_enabled:
            return
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self._save_csv(self.visualization_dir / "training_metrics.csv", self.training_metric_history)
        self._save_csv(self.visualization_dir / "episode_metrics.csv", self.episode_metric_history)
        self._save_summary_table()
        self._save_plots()

    def _save_csv(self, file_path, rows):
        if not rows:
            return
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)
        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _save_summary_table(self):
        total_episodes = len(self.episode_metric_history)
        final_goal_reach_100 = float(np.mean(self.episode_successes[-100:])) if self.episode_successes else 0.0
        final_illegal_100 = float(np.mean(self.episode_illegal_crossings[-100:])) if self.episode_illegal_crossings else 0.0
        avg_reward_100 = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0

        actor_loss = None
        critic_loss = None
        for row in reversed(self.training_metric_history):
            if actor_loss is None and "train/actor_loss" in row:
                actor_loss = row["train/actor_loss"]
            if critic_loss is None and "train/critic_loss" in row:
                critic_loss = row["train/critic_loss"]
            if actor_loss is not None and critic_loss is not None:
                break

        summary_rows = [
            ("total_episodes", total_episodes),
            ("avg_reward_100", round(avg_reward_100, 4)),
            ("goal_reach_rate_100", round(final_goal_reach_100, 4)),
            ("illegal_crossing_rate_100", round(final_illegal_100, 4)),
            ("latest_actor_loss", "n/a" if actor_loss is None else round(float(actor_loss), 6)),
            ("latest_critic_loss", "n/a" if critic_loss is None else round(float(critic_loss), 6)),
        ]

        csv_path = self.visualization_dir / "metrics_summary_table.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["metric", "value"])
            for key, value in summary_rows:
                writer.writerow([key, value])

        md_path = self.visualization_dir / "metrics_summary_table.md"
        with open(md_path, "w", encoding="utf-8") as mdfile:
            mdfile.write("| metric | value |\n")
            mdfile.write("|---|---|\n")
            for key, value in summary_rows:
                mdfile.write(f"| {key} | {value} |\n")

    def _save_plots(self):
        if plt is None:
            rospy.logwarn("Metric visualization skipped: matplotlib not available")
            return

        if self.episode_metric_history:
            timesteps = [row.get("train/timestep", idx) for idx, row in enumerate(self.episode_metric_history, start=1)]
            rewards = [row.get("episode/reward", 0.0) for row in self.episode_metric_history]
            goal_rate_100 = [row.get("episode/goal_reach_rate_100", 0.0) for row in self.episode_metric_history]
            illegal_rate_100 = [row.get("episode/illegal_crossing_rate_100", 0.0) for row in self.episode_metric_history]

            plt.figure(figsize=(10, 5))
            plt.plot(timesteps, rewards, label="episode_reward", alpha=0.7)
            plt.xlabel("timestep")
            plt.ylabel("reward")
            plt.title("Episode Reward over Time")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.visualization_dir / "episode_reward_curve.png", dpi=150)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(timesteps, goal_rate_100, label="goal_reach_rate_100")
            plt.plot(timesteps, illegal_rate_100, label="illegal_crossing_rate_100")
            plt.xlabel("timestep")
            plt.ylabel("rate")
            plt.title("Safety and Success Rates (100-episode window)")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.visualization_dir / "episode_rate_curves.png", dpi=150)
            plt.close()

        if self.training_metric_history:
            train_steps = [row.get("train/timestep", idx) for idx, row in enumerate(self.training_metric_history, start=1)]
            actor_loss = [row.get("train/actor_loss", np.nan) for row in self.training_metric_history]
            critic_loss = [row.get("train/critic_loss", np.nan) for row in self.training_metric_history]

            plt.figure(figsize=(10, 5))
            plt.plot(train_steps, actor_loss, label="actor_loss")
            plt.plot(train_steps, critic_loss, label="critic_loss")
            plt.xlabel("timestep")
            plt.ylabel("loss")
            plt.title("SAC Loss Curves")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.visualization_dir / "loss_curves.png", dpi=150)
            plt.close()


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


class WandbCheckpointCallback(CheckpointCallback):
    """
    Extended checkpoint callback that also uploads checkpoints to wandb.
    """
    
    def __init__(self, save_freq, save_path, name_prefix="rl_model",
                 save_replay_buffer=True, save_vecnormalize=True,
                 use_wandb=True, verbose=0):
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose
        )
        self.use_wandb = use_wandb
    
    def _on_step(self) -> bool:
        # Call parent to save checkpoint locally
        result = super()._on_step()
        
        # Upload to wandb if checkpoint was just saved
        if self.use_wandb and self.n_calls % self.save_freq == 0:
            # Upload model checkpoint
            checkpoint_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            if os.path.exists(checkpoint_path):
                rospy.loginfo(f"Uploading checkpoint to wandb: {checkpoint_path}")
                wandb.save(checkpoint_path, base_path=self.save_path, policy="now")
            
            # Upload replay buffer if it exists
            if self.save_replay_buffer:
                replay_buffer_prefix = self.name_prefix.replace("sac_jackal", "sac_replay_buffer")
                buffer_path = os.path.join(
                    self.save_path,
                    f"{replay_buffer_prefix}_{self.num_timesteps}_steps.pkl"
                )
                if os.path.exists(buffer_path):
                    rospy.loginfo(f"Uploading replay buffer to wandb: {buffer_path}")
                    wandb.save(buffer_path, base_path=self.save_path, policy="now")
        
        return result


# ==================== SAC Training Functions ====================
def create_env(config):
    """Create and wrap the environment."""
    env = gym.make(config["env_id"])
    env = Monitor(env)
    return env


def create_policy_kwargs(config, observation_space):
    """Create policy keyword arguments with custom feature extractor."""
    
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
        verbose=1,
        seed=config["seed"],
        device=config["device"],
        tensorboard_log=config["log_dir"],
    )
    
    return model


def setup_callbacks(config, env, initial_episode_count=0):
    """Setup training callbacks."""
    callbacks = []

    # Periodic evaluation callback
    if config.get("eval_freq", 0) and config["eval_freq"] > 0:
        eval_save_dir = os.path.join(config["save_dir"], "best_model")
        eval_log_dir = os.path.join(config["log_dir"], "eval")
        Path(eval_save_dir).mkdir(parents=True, exist_ok=True)
        Path(eval_log_dir).mkdir(parents=True, exist_ok=True)

        eval_env = create_env(config)
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=eval_save_dir,
            log_path=eval_log_dir,
            eval_freq=config["eval_freq"],
            n_eval_episodes=config["n_eval_episodes"],
            deterministic=config.get("eval_deterministic", True),
            render=False,
            verbose=1,
        )
        callbacks.append(eval_callback)
    
    # Checkpoint callback (with wandb upload if enabled)
    checkpoint_callback = WandbCheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=config["save_dir"],
        name_prefix=config.get("checkpoint_name_prefix", "sac_jackal_newGen"),
        save_replay_buffer=config.get("checkpoint_save_replay_buffer", True),
        save_vecnormalize=True,
        use_wandb=config["use_wandb"],
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Video recording callback
    if config.get("record_video", False):
        video_callback = VideoRecorderCallback(
            video_dir=config["video_dir"],
            record_freq=config["video_episode_freq"],
            fps=30,
            use_wandb=config["use_wandb"],
            robot_topic="/front/image_raw",
            overhead_topic="/overhead_camera/image_raw",
            verbose=1
        )
        callbacks.append(video_callback)
    
    # Reward/Loss/Metrics callback (console always, wandb optional)
    callbacks.append(
        WandbMetricsCallback(
            use_wandb=config["use_wandb"],
            log_freq=config.get("metrics_log_freq", 500),
            visualization_enabled=config.get("enable_metric_visualization", True),
            visualization_dir=config.get("visualization_dir", "./sac_logs/visualizations"),
            initial_episode_count=initial_episode_count,
        )
    )

    # Wandb-specific callbacks
    if config["use_wandb"]:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"{config['save_dir']}/wandb_models",
            verbose=2,
        )
        callbacks.append(wandb_callback)
        callbacks.append(MemoryProfileCallback(log_freq=1000))
        callbacks.append(GradientMonitorCallback(log_freq=100))
    
    return CallbackList(callbacks)


def save_training_metadata(save_dir, episode_count):
    """Save training metadata to JSON file."""
    metadata_path = os.path.join(save_dir, "training_metadata.json")
    metadata = {"episode_count": episode_count}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def load_training_metadata(save_dir):
    """Load training metadata from JSON file."""
    metadata_path = os.path.join(save_dir, "training_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get("episode_count", 0)
    return 0


def train_sac(config):
    """Main SAC training function."""
    
    # Create directories
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)
    if config.get("record_video", False):
        Path(config["video_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = create_env(config)
    
    # Initialize or resume wandb
    if config["use_wandb"]:
        wandb_kwargs = {
            "project": config["wandb_project"],
            "entity": config["wandb_entity"],
            "config": config,
            "sync_tensorboard": True,
            "monitor_gym": True,
            "save_code": True,
        }
        
        # Resume or create new run
        if config.get("wandb_run_id"):
            wandb_kwargs["id"] = config["wandb_run_id"]
            wandb_kwargs["resume"] = "must"
        
        wandb.init(**wandb_kwargs)
    
    try:
        # Load initial episode count for resuming training
        initial_episode_count = 0
        is_resuming = False
        resume_path = config.get("resume_path")
        
        # Create or load model
        if resume_path:
            checkpoint_zip = resume_path + ".zip"
            if not os.path.exists(checkpoint_zip):
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {checkpoint_zip}. "
                    "Use --resume with path without .zip suffix."
                )

            rospy.loginfo(f"Loading model from {resume_path}...")
            model = SAC.load(
                resume_path,
                env=env,
                device=config["device"],
            )
            is_resuming = True
            # Load replay buffer if it exists
            buffer_path = resume_path.replace("sac_jackal", "sac_replay_buffer")
            if os.path.exists(buffer_path + ".pkl"):
                rospy.loginfo(f"Loading replay buffer from {buffer_path}...")
                model.load_replay_buffer(buffer_path)
            else:
                rospy.logwarn(f"Replay buffer not found at {buffer_path}.pkl (continuing without it)")
            
            # Load episode count from metadata
            initial_episode_count = load_training_metadata(config["save_dir"])
            rospy.loginfo(
                f"Model loaded successfully (model timesteps: {model.num_timesteps}, "
                f"episode count: {initial_episode_count})"
            )
        else:
            rospy.loginfo("Creating new SAC model...")
            model = create_sac_model(env, config)
        
        # Log actual device being used by the model
        rospy.loginfo(f"Model device: {model.device}")
        rospy.loginfo(f"Policy network device: {model.policy.device if hasattr(model.policy, 'device') else 'N/A'}")
        
        # Print model summary
        total_params = sum(p.numel() for p in model.policy.parameters())
        rospy.loginfo(f"Total model parameters: {total_params:,}")
        
        # Setup callbacks
        callbacks = setup_callbacks(config, env, initial_episode_count=initial_episode_count)

        rospy.loginfo(
            "Checkpointing enabled: every {} steps -> {} (prefix: {})".format(
                config["checkpoint_freq"],
                config["save_dir"],
                config.get("checkpoint_name_prefix", "sac_jackal"),
            )
        )
        
        # Train
        rospy.loginfo(f"Starting training for {config['total_timesteps']} timesteps...")
        start_time = time.time()

        if config.get("manual_training_loop", False):
            rospy.loginfo("Using manual training loop (episode collection + batch updates).")
            train_sac_manual_loop(
                model=model,
                total_timesteps=config["total_timesteps"],
                callback=callbacks,
                log_interval=config["log_interval"],
                reset_num_timesteps=not is_resuming,
            )
        else:
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=callbacks,
                log_interval=config["log_interval"],
                progress_bar=True,
                reset_num_timesteps=not is_resuming,
            )

        training_time = time.time() - start_time
        rospy.loginfo(f"Training completed in {training_time/3600:.2f} hours")
        
        # Get final episode count from callback
        metric_callback = None
        for callback in callbacks.callbacks:
            if isinstance(callback, WandbMetricsCallback):
                metric_callback = callback
                break
        
        # Save final model
        final_model_path = f"{config['save_dir']}/{config['checkpoint_name_prefix']}_final"
        model.save(final_model_path)
        rospy.loginfo(f"Final model saved to {final_model_path}")
        
        # Upload final model to wandb
        if config["use_wandb"]:
            rospy.loginfo("Uploading final model to wandb...")
            wandb.save(f"{final_model_path}.zip", base_path=config["save_dir"], policy="now")
        
        # Save replay buffer
        buffer_path = f"{config['save_dir']}/{config['replay_buffer_name_prefix']}_final"
        model.save_replay_buffer(buffer_path)
        rospy.loginfo(f"Replay buffer saved to {buffer_path}")
        
        # Upload final replay buffer to wandb
        if config["use_wandb"]:
            rospy.loginfo("Uploading final replay buffer to wandb...")
            wandb.save(f"{buffer_path}.pkl", base_path=config["save_dir"], policy="now")
        
        # Save training metadata
        if metric_callback:
            save_training_metadata(config["save_dir"], metric_callback.episode_count)
            rospy.loginfo(f"Training metadata saved (total episodes: {metric_callback.episode_count})")
        
    finally:
        if config["use_wandb"]:
            wandb.finish()
        if 'callbacks' in locals():
            for callback in callbacks.callbacks:
                if isinstance(callback, EvalCallback) and getattr(callback, "eval_env", None) is not None:
                    callback.eval_env.close()
        env.close()
    
    return model


def train_sac_manual_loop(model, total_timesteps, callback=None, log_interval=10, reset_num_timesteps=True):
    """
    Manual SAC training loop with explicit rollout collection and batch updates.

    This mirrors the OffPolicyAlgorithm learn flow but keeps control visible:
    1) collect environment interactions/episodes
    2) update policy and critics using replay-buffer mini-batches
    """
    total_timesteps, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name="SAC",
        progress_bar=True,
    )

    callback.on_training_start(locals(), globals())

    while model.num_timesteps < total_timesteps:
        rollout = model.collect_rollouts(
            env=model.env,
            train_freq=model.train_freq,
            action_noise=model.action_noise,
            callback=callback,
            learning_starts=model.learning_starts,
            replay_buffer=model.replay_buffer,
            log_interval=log_interval,
        )

        if not rollout.continue_training:
            break

        if model.num_timesteps > model.learning_starts:
            gradient_steps = model.gradient_steps
            if gradient_steps < 0:
                gradient_steps = rollout.episode_timesteps

            if gradient_steps > 0:
                model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)

    callback.on_training_end()
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
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Train SAC agent for Jackal Crossroad")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto/cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (without .zip)")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="Wandb run ID to resume")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording")
    parser.add_argument("--video-episode-freq", type=int, default=None, help="Record video every N episodes")
    parser.add_argument("--metrics-log-freq", type=int, default=None, help="Log rewards/losses/metrics every N timesteps")
    parser.add_argument("--eval-freq", type=int, default=None, help="Run evaluation every N timesteps (0 to disable)")
    parser.add_argument("--n-eval-episodes", type=int, default=None, help="Number of episodes per evaluation run")
    parser.add_argument("--stochastic-eval", action="store_true", help="Use stochastic actions during evaluation")
    parser.add_argument("--checkpoint-freq", type=int, default=None, help="Save model checkpoint every N timesteps")
    parser.add_argument("--checkpoint-prefix", type=str, default=None, help="Filename prefix for interval checkpoints")
    parser.add_argument("--no-checkpoint-replay-buffer", action="store_true", help="Do not save replay buffer for interval checkpoints")
    parser.add_argument("--no-metric-visualization", action="store_true", help="Disable local metric visualizations (plots/tables/csv)")
    parser.add_argument("--visualization-dir", type=str, default=None, help="Directory for metric plots and tables")
    parser.add_argument("--manual-loop", action="store_true", help="Use manual loop with explicit episode collection and batch updates")
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('sac_training', anonymous=True, log_level=rospy.INFO)
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.no_wandb:
        config["use_wandb"] = False
    if args.timesteps:
        config["total_timesteps"] = args.timesteps
    if args.seed:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if args.resume:
        config["resume_path"] = args.resume
    if args.wandb_run_id:
        config["wandb_run_id"] = args.wandb_run_id
    if args.record_video:
        config["record_video"] = True
    if args.video_episode_freq:
        config["video_episode_freq"] = args.video_episode_freq
    if args.metrics_log_freq:
        config["metrics_log_freq"] = args.metrics_log_freq
    if args.eval_freq is not None:
        config["eval_freq"] = args.eval_freq
    if args.n_eval_episodes:
        config["n_eval_episodes"] = args.n_eval_episodes
    if args.stochastic_eval:
        config["eval_deterministic"] = False
    if args.checkpoint_freq is not None:
        config["checkpoint_freq"] = args.checkpoint_freq
    if args.checkpoint_prefix:
        config["checkpoint_name_prefix"] = args.checkpoint_prefix
    if args.no_checkpoint_replay_buffer:
        config["checkpoint_save_replay_buffer"] = False
    if args.no_metric_visualization:
        config["enable_metric_visualization"] = False
    if args.visualization_dir:
        config["visualization_dir"] = args.visualization_dir
    if args.manual_loop:
        config["manual_training_loop"] = True
    
    # Set seeds for reproducibility
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    
    # Log device information
    rospy.loginfo("=" * 60)
    rospy.loginfo("Device Information")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"PyTorch version: {torch.__version__}")
    rospy.loginfo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        rospy.loginfo(f"CUDA version: {torch.version.cuda}")
        rospy.loginfo(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            rospy.loginfo(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    rospy.loginfo(f"Requested device: {config['device']}")
    rospy.loginfo("=" * 60)
    
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
