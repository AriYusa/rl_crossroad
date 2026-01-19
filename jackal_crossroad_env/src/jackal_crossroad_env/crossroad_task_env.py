#!/usr/bin/env python3
"""
Jackal Crossroad Task Environment
This class defines the task-specific logic (rewards, observations, actions, done conditions)
Based on openai_ros templates
"""

import rospy
import numpy as np
from gym import spaces
from jackal_crossroad_env.jackal_robot_env import JackalRobotEnv
from gym.envs.registration import register
import tf
import os
from datetime import datetime
from cv_bridge import CvBridge
import cv2


class CrossroadEnv(JackalRobotEnv):
    """
    Task environment for training Jackal to navigate a crossroad with traffic lights.
    This class defines:
    - Observation space
    - Action space
    - Reward function
    - Done conditions
    """

    def __init__(self):
        """
        Initialize the task environment.
        """
        rospy.logdebug("Start JackalCrossroadEnv INIT...")
        
        # Task-specific parameters
        self.goal_position = np.array([10.0, 0.0])  # Goal position (x, y)
        self.start_position = np.array([-5.0, 0.0])  # Start position
        
        # Collision detection parameters
        self.min_laser_distance = 0.3  # Minimum safe distance to obstacles (meters)
        self.collision_ray_threshold = 3  # Number of rays that must detect obstacle for collision
        self.proximity_index_offset = 5  # Number of indices on each side for averaging (center, -5, +5)
        
        # Episode parameters
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Initialize parent class
        super(CrossroadEnv, self).__init__()
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Define action space: [linear_velocity, angular_velocity]
        # Linear velocity: -1.0 to 2.0 m/s
        # Angular velocity: -2.0 to 2.0 rad/s
        self.action_space = spaces.Box(
            low=np.array([-1.0, -2.0]),
            high=np.array([2.0, 2.0]),
            dtype=np.float32
        )
        
        obs_dim = ...
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        rospy.logdebug("Finished JackalCrossroadEnv INIT...")
    
    def _set_init_pose(self):
        """
        Sets the robot in its initial pose for the task.
        """
        # Move robot to start position
        x, y = self.start_position
        self.set_robot_pose(x, y, 0.0, 0.0, 0.0, 0.0)
        # Stop the robot
        self.move_base(0.0, 0.0, epsilon=0.05, update_rate=10)
        return True
    
    def _init_env_variables(self):
        """
        Initialize variables for the episode.
        """
        self.current_step = 0
        self.cumulated_reward = 0.0
    
    def _set_action(self, action):
        """
        Apply action to the robot.
        Args:
            action: Array [linear_velocity, angular_velocity]
        """
        
        # Clip actions to be within limits
        linear_vel = np.clip(action[0], -1.0, 2.0)
        angular_vel = np.clip(action[1], -2.0, 2.0)
        
        # Publish velocity command directly
        self.move_base(linear_vel, angular_vel, epsilon=0.05, update_rate=10)

    
    def _get_obs(self):
        """
        Get current observation from the environment.
        Returns:
            numpy array: Observation vector
        """
        # Get raw sensor data from parent class
        raw_obs = super(CrossroadEnv, self)._get_obs()
        
        ...

    
    def _is_done(self, observations):
        """
        Check if episode should terminate.
        Args:
            observations: Current observation
        Returns:
            bool: True if episode is done
        """
        ...
    
    def _compute_reward(self, observations, done):
        """
        Compute reward for current step.
        Args:
            observations: Current observation
            done: Whether episode is done
        Returns:
            float: Reward value
        """
        ...

    def _is_collision(self, observations):
        """Sample laser rays, take a proximity_index_offset and average each ray with its "neighboring" rays. Check that average distances are below threshold for collision."""
        raw_laser_scan = observations['laser_scan']
        # Sample laser rays at specified indices
        ranges = np.array(raw_laser_scan.ranges)
        ranges[np.isinf(ranges)] = raw_laser_scan.range_max
        ranges[np.isnan(ranges)] = raw_laser_scan.range_max
        
        # Downsample by taking evenly spaced samples
        indices = np.linspace(0, len(ranges) - 1, self.num_laser_rays, dtype=int)
        downsampled = ranges[indices]
        
        averaged = np.zeros(len(downsampled))
        
        for i, idx in enumerate(indices):
            # Take 3 samples: idx-5, idx, idx+5
            offset = self.proximity_index_offset
            sample_indices = [
                max(0, idx - offset),
                idx,
                min(len(ranges) - 1, idx + offset)
            ]
            
            sample_values = [ranges[si] for si in sample_indices]
            averaged[i] = np.mean(sample_values)

        # Count how many averaged distances are below the minimum distance
        close_rays = np.sum(averaged < self.min_laser_distance)
        return close_rays >= self.collision_ray_threshold

    
    def step(self, action):
        """
        Execute one step in the environment.
        Args:
            action: Action to take
        Returns:
            observation, reward, done, info
        """
        rospy.logdebug("Start Step ==> Action: " + str(action))
        
        # Unpause simulation
        self.gazebo.unpauseSim()
        
        # Execute action
        self._set_action(action)
        
        # Wait for simulation to process action
        rospy.sleep(0.1)
        
        # Pause simulation
        self.gazebo.pauseSim()
        
        # Get observation
        obs = self._get_obs()
        
        # Check if done
        done = self._is_done(obs)
        
        # Compute reward
        reward = self._compute_reward(obs, done)
        
        # Update counters
        self.current_step += 1
        self.cumulated_reward += reward
        
        # Info dict
        info = {
            'cumulated_reward': self.cumulated_reward,
            'step': self.current_step
        }
        
        rospy.logdebug("End Step ==> Obs: " + str(obs))
        
        return obs, reward, done, info
    
    def reset(self):
        """
        Reset the environment.
        Returns:
            observation: Initial observation
        """
        rospy.logdebug("Start Reset")
        
        # Pause simulation
        self.gazebo.pauseSim()
        
        # Reset world or simulation
        self.gazebo.resetWorld()
        
        # Unpause simulation
        self.gazebo.unpauseSim()
        
        # Check sensors are ready
        self._check_all_sensors_ready()
        
        # Pause simulation
        self.gazebo.pauseSim()
        
        # Set robot to initial pose
        self._set_init_pose()
        
        # Initialize episode variables
        self._init_env_variables()
        
        # Reset previous distance tracking
        if hasattr(self, 'previous_distance_to_goal'):
            delattr(self, 'previous_distance_to_goal')
        
        # Unpause to get observation
        self.gazebo.unpauseSim()
        
        # Get initial observation
        obs = self._get_obs()
        
        # Pause again
        self.gazebo.pauseSim()
        
        rospy.logdebug("End Reset")
        
        return obs
