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
        
        # Define observation space
        # Will include: laser scan (simplified to N rays), position, velocity, traffic light state
        self.num_laser_rays = 10  # Downsample laser scan
        
        # Observation: [laser_rays (10), position_x, position_y, velocity_x, velocity_y, 
        #               yaw, angular_velocity, distance_to_goal, angle_to_goal, traffic_light_state]
        obs_dim = self.num_laser_rays + 9
        
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
        
        
    
    def _process_laser_scan(self, laser_scan):
        """
        Process laser scan data by downsampling.
        Args:
            laser_scan: LaserScan message
        Returns:
            numpy array: Processed laser ranges
        """
        if laser_scan is None:
            return np.ones(self.num_laser_rays) * 10.0
        
        ranges = np.array(laser_scan.ranges)
        
        # Replace inf and nan with max range
        ranges[np.isinf(ranges)] = laser_scan.range_max
        ranges[np.isnan(ranges)] = laser_scan.range_max
        
        # Downsample by taking evenly spaced samples
        indices = np.linspace(0, len(ranges) - 1, self.num_laser_rays, dtype=int)
        downsampled = ranges[indices]
        
        return downsampled
    
    def _save_laser_scan(self, laser_rays, raw_laser_scan, camera_image=None, averaged_laser_rays=None):
        """
        Save laser scan data and camera image to file when collision occurs.
        Args:
            laser_rays: Downsampled laser data
            raw_laser_scan: Full LaserScan message
            camera_image: Raw camera Image message (optional)
            averaged_laser_rays: Smoothed laser data used for collision check (optional)
        """
        # Create collision_data directory if it doesn't exist
        save_dir = "/tmp/collision_data"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"collision_{timestamp}.npz")
        image_filename = os.path.join(save_dir, f"collision_{timestamp}.jpg")
        
        # Save full laser scan data
        if raw_laser_scan is not None:
            full_ranges = np.array(raw_laser_scan.ranges)
            # Replace inf and nan with max range for saving
            full_ranges[np.isinf(full_ranges)] = raw_laser_scan.range_max
            full_ranges[np.isnan(full_ranges)] = raw_laser_scan.range_max
            
            # Prepare data dict
            data_dict = {
                'full_ranges': full_ranges,
                'downsampled_ranges': laser_rays,
                'angle_min': raw_laser_scan.angle_min,
                'angle_max': raw_laser_scan.angle_max,
                'angle_increment': raw_laser_scan.angle_increment,
                'range_min': raw_laser_scan.range_min,
                'range_max': raw_laser_scan.range_max,
                'timestamp': timestamp,
                'collision_distance_threshold': self.min_laser_distance,
                'collision_ray_threshold': self.collision_ray_threshold,
                'proximity_index_offset': self.proximity_index_offset,
                'num_close_rays': np.sum(
                    (averaged_laser_rays if averaged_laser_rays is not None else laser_rays)
                    < self.min_laser_distance)
            }

            if averaged_laser_rays is not None:
                data_dict['averaged_downsampled_ranges'] = averaged_laser_rays
            
            # Convert and save camera image if available
            if camera_image is not None:
                try:
                    # Convert ROS Image message to OpenCV image
                    cv_image = self.bridge.imgmsg_to_cv2(camera_image, desired_encoding='bgr8')
                    
                    # Save image as JPEG
                    cv2.imwrite(image_filename, cv_image)
                    
                    # Also store image data in npz (as numpy array)
                    data_dict['camera_image'] = cv_image
                    data_dict['image_filename'] = image_filename
                    
                except Exception as e:
                    rospy.logwarn(f"Failed to save camera image: {e}")
            
            # Save comprehensive data
            np.savez(filename, **data_dict)
    
    def _average_proximity_from_raw(self, laser_rays_indices):
        """
        For collision detection: average 3 rays at each sampled position:
        center index, center-5, and center+5.
        """
        if not hasattr(self, '_raw_laser_scan') or self._raw_laser_scan is None:
            return None
        
        ranges = np.array(self._raw_laser_scan.ranges)
        ranges[np.isinf(ranges)] = self._raw_laser_scan.range_max
        ranges[np.isnan(ranges)] = self._raw_laser_scan.range_max
        
        num_ranges = len(ranges)
        averaged = np.zeros(len(laser_rays_indices))
        
        for i, idx in enumerate(laser_rays_indices):
            # Take 3 samples: idx-5, idx, idx+5
            offset = self.proximity_index_offset
            sample_indices = [
                max(0, idx - offset),
                idx,
                min(num_ranges - 1, idx + offset)
            ]
            
            sample_values = [ranges[si] for si in sample_indices]
            averaged[i] = np.mean(sample_values)
        
        return averaged

    
    def _is_done(self, observations):
        """
        Check if episode should terminate.
        Args:
            observations: Current observation
        Returns:
            bool: True if episode is done
        """
        # Extract relevant data
        laser_rays = observations[:self.num_laser_rays]
        distance_to_goal = observations[-3]
        
        # For collision detection, average neighboring rays from full scan to reduce noise
        if hasattr(self, '_raw_laser_scan') and self._raw_laser_scan is not None:
            # Get indices used for downsampling
            full_scan_len = len(self._raw_laser_scan.ranges)
            indices = np.linspace(0, full_scan_len - 1, self.num_laser_rays, dtype=int)
            self.averaged_rays = self._average_proximity_from_raw(indices)
        else:
            self.averaged_rays = laser_rays
        
        # Check for collision using averaged values
        close_rays = np.sum(self.averaged_rays < self.min_laser_distance)
        if close_rays >= self.collision_ray_threshold:
            # Save laser scan data and camera image
            # camera_img = self._raw_camera_image if hasattr(self, '_raw_camera_image') else None
            # if hasattr(self, '_raw_laser_scan'):
            #     self._save_laser_scan(laser_rays, self._raw_laser_scan, camera_img, self.averaged_rays)
            return True
        
        if distance_to_goal < 0.5:  # Within 0.5m of goal
            rospy.loginfo("Episode done: Goal reached!")
            return True
        
        # Check max steps
        if self.current_step >= self.max_episode_steps:
            rospy.loginfo("Episode done: Max steps reached!")
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
        ...
    
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
