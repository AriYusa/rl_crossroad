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
        # keep the start on the pavement (matches launch default y=-4.5)
        self.start_position = np.array([-5.0, -4.5])  # Start position
        
        # Collision detection parameters
        self.min_laser_distance = 0.3  # Minimum safe distance to obstacles (meters)
        self.collision_ray_threshold = 3  # Number of rays that must detect obstacle for collision
        self.proximity_index_offset = 5  # Number of indices on each side for averaging (center, -5, +5)
        self.num_laser_rays = 20  # downsampled laser rays used in observations
        self.goal_threshold = 1  # goal threshold (meters)
        
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
        
        obs_dim = 5 + self.num_laser_rays  # [distance, heading_err, lin_vel, ang_vel, traffic] + laser rays
        
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
        laser_scan = raw_obs['laser_scan']
        odom = raw_obs['odom']
        traffic_light = raw_obs['traffic_light']

        # Pose and heading
        pos = odom.pose.pose.position
        orient = odom.pose.pose.orientation
        yaw = tf.transformations.euler_from_quaternion(
            [orient.x, orient.y, orient.z, orient.w]
        )[2]

        # Goal features
        dx = self.goal_position[0] - pos.x
        dy = self.goal_position[1] - pos.y
        # Manhattan distance for curriculum phase 1 (road crossing)
        distance = abs(dx) + abs(dy)
        heading = np.arctan2(dy, dx)
        heading_error = self._normalize_angle(heading - yaw)

        # Velocities
        lin_vel = odom.twist.twist.linear.x
        ang_vel = odom.twist.twist.angular.z

        # Laser downsample
        ranges = np.array(laser_scan.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = laser_scan.range_max
        ranges[np.isnan(ranges)] = laser_scan.range_max
        if len(ranges) == 0:
            laser = np.zeros(self.num_laser_rays, dtype=np.float32)
        else:
            indices = np.linspace(0, len(ranges) - 1, self.num_laser_rays, dtype=int)
            laser = ranges[indices].astype(np.float32)

        obs = np.concatenate((
            np.array([distance, heading_error, lin_vel, ang_vel, float(traffic_light)], dtype=np.float32),
            laser
        ))
        return obs

    
    def _is_done(self, observations):
        """
        Check if episode should terminate.
        Args:
            observations: Current observation
        Returns:
            bool: True if episode is done
        """
        if observations is None or len(observations) == 0:
            return False

        distance = observations[0]
        if self._is_collision(observations):
            return True
        if distance < self.goal_threshold:
            return True
        if self.current_step >= self.max_episode_steps:
            return True
        return False
    
    def _compute_reward(self, observations, done):
        """
        Compute reward for current step.
        Args:
            observations: Current observation
            done: Whether episode is done
        Returns:
            float: Reward value
        """
        if observations is None or len(observations) == 0:
            return 0.0

        distance = observations[0]
        if self._is_collision(observations):
            return -100.0
        if distance < self.goal_threshold:
            return 100.0

        if not hasattr(self, 'previous_distance_to_goal'):
            self.previous_distance_to_goal = distance

        progress = self.previous_distance_to_goal - distance
        self.previous_distance_to_goal = distance

        # Small step penalty to encourage efficiency
        return (progress * 10.0) - 0.01

    def _is_collision(self, observations=None):
        """Sample laser rays, take a proximity_index_offset and average each ray with its "neighboring" rays. Check that average distances are below threshold for collision."""
        if observations is not None and isinstance(observations, dict):
            raw_laser_scan = observations.get('laser_scan')
        else:
            raw_laser_scan = self.laser_scan
        if raw_laser_scan is None:
            return False
        # Sample laser rays at specified indices
        ranges = np.array(raw_laser_scan.ranges, dtype=np.float32)
        if len(ranges) == 0:
            return False
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

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    
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
        
        # Leave sim unpaused so /clock and sensors keep publishing
        rospy.logdebug("End Reset (sim unpaused)")
        
        return obs
