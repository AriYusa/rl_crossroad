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
from gazebo_msgs.srv import GetModelState


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
        self.base_goal_position = np.array([5.0, 4.5])  # Base goal position (x, y)
        # keep the start on the pavement (matches launch default y=-4.5)
        self.start_position = np.array([-5.0, -4.5])  # Start position
        self.goal_threshold = 0.3  # goal threshold (meters) - same as visulaision radius 

        # Map boundaries
        self.map_x_min = -10.0
        self.map_x_max = 10.0
        self.map_y_min = -5.5
        self.map_y_max = 5.5
        
        # Road boundaries
        self.road_y_min = -3.5
        self.road_y_max = 3.5
        
        # Goal randomization parameters
        self.spawn_goal_on_sidewalk_only = True
        self.initial_noise_level = 0.1  # Initial noise in meters
        self.max_noise_level = 4.0  # Maximum noise in meters
        self.noise_increase_rate = 0.005  # Noise increase per episode
        self.episode_count = 0  # Track total episodes
        self.current_noise_level = self.initial_noise_level

        # Initialize goal position with randomness
        self.goal_position = self._generate_random_goal()
        
        # Collision detection parameters
        self.min_laser_distance = 0.3  # Minimum safe distance to obstacles (meters)
        self.collision_ray_threshold = 3  # Number of rays that must detect obstacle for collision
        self.proximity_index_offset = 5  # Number of indices on each side for averaging (center, -5, +5)
        self.num_laser_rays = 20  # downsampled laser rays used in observations

        # Reward parameters
        self.collision_penalty = -100.0  # Penalty for collision
        self.goal_reward = 100.0  # Reward for reaching goal
        self.progress_multiplier = 10.0  # Multiplier for progress towards goal
        self.step_penalty = 0.01  # Penalty for each step (encourages efficiency)
        
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
        
        # Define observation space as a Dict
        # raw_image: camera image (height, width, channels)
        # laser_scan: laser range data
        # robot_coords: [x, y, yaw]
        # goal_coords: [x, y]
        self.observation_space = spaces.Dict({
            'raw_image': spaces.Box(
                low=0,
                high=255,
                shape=(120, 160, 3),  # Reduced resolution for memory efficiency
                dtype=np.uint8
            ),
            'laser_scan': spaces.Box(
                low=0.0,
                high=30.0,
                shape=(self.num_laser_rays,),
                dtype=np.float32
            ),
            'robot_coords': spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.pi]),
                high=np.array([np.inf, np.inf, np.pi]),
                shape=(3,),
                dtype=np.float32
            ),
            'goal_coords': spaces.Box(
                low=np.array([-np.inf, -np.inf]),
                high=np.array([np.inf, np.inf]),
                shape=(2,),
                dtype=np.float32
            )
        })
        
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
            dict: Observation dictionary with raw_image, laser_scan, robot_coords, goal_coords
        """
        # Get raw sensor data from parent class
        raw_obs = super(CrossroadEnv, self)._get_obs()
        laser_scan_msg = raw_obs['laser_scan']
        camera_image_msg = raw_obs['camera_image']
        
        # Get ground truth pose from Gazebo model state
        robot_coords = self._get_robot_ground_truth_pose()
        
        # Process camera image
        if camera_image_msg is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(camera_image_msg, "bgr8")
                # Ensure consistent size (resize if needed)
                if cv_image.shape != (120, 160, 3):
                    cv_image = cv2.resize(cv_image, (160, 120))
                raw_image = cv_image.astype(np.uint8)
            except Exception as e:
                rospy.logwarn(f"Error converting image: {e}")
                raw_image = np.zeros((120, 160, 3), dtype=np.uint8)
        else:
            raw_image = np.zeros((120, 160, 3), dtype=np.uint8)
        
        # Process laser scan
        if laser_scan_msg is not None:
            ranges = np.array(laser_scan_msg.ranges, dtype=np.float32)
            ranges[np.isinf(ranges)] = laser_scan_msg.range_max
            ranges[np.isnan(ranges)] = laser_scan_msg.range_max
            if len(ranges) > 0:
                indices = np.linspace(0, len(ranges) - 1, self.num_laser_rays, dtype=int)
                laser_scan = ranges[indices].astype(np.float32)
            else:
                laser_scan = np.zeros(self.num_laser_rays, dtype=np.float32)
        else:
            laser_scan = np.zeros(self.num_laser_rays, dtype=np.float32)
        
        # Goal coordinates
        goal_coords = self.goal_position.astype(np.float32)
        
        obs = {
            'raw_image': raw_image,
            'laser_scan': laser_scan,
            'robot_coords': robot_coords,
            'goal_coords': goal_coords
        }
        
        return obs

    
    def _is_done(self, observations):
        """
        Check if episode should terminate.
        Args:
            observations: Current observation dict
        Returns:
            bool: True if episode is done
        """
        if observations is None or not isinstance(observations, dict):
            return False
        
        # Calculate distance to goal
        robot_coords = observations.get('robot_coords', np.zeros(3))
        goal_coords = observations.get('goal_coords', np.zeros(2))
        distance = np.linalg.norm(robot_coords[:2] - goal_coords)
        
        if self._is_collision():
            rospy.loginfo("Episode done: Collision detected")
            return True
        if distance < self.goal_threshold:
            rospy.loginfo("Episode done: Goal reached")
            return True
        if self.current_step >= self.max_episode_steps:
            rospy.loginfo("Episode done: Maximum steps reached")
            return True
        return False
    
    def _compute_reward(self, observations, done):
        """
        Compute reward for current step.
        Args:
            observations: Current observation dict
            done: Whether episode is done
        Returns:
            float: Reward value
        """
        if observations is None or not isinstance(observations, dict):
            return 0.0
        
        # Calculate distance to goal
        robot_coords = observations.get('robot_coords', np.zeros(3))
        goal_coords = observations.get('goal_coords', np.zeros(2))
        distance = np.linalg.norm(robot_coords[:2] - goal_coords)
        
        if self._is_collision():
            return self.collision_penalty
        if distance < self.goal_threshold:
            return self.goal_reward

        if not hasattr(self, 'previous_distance_to_goal'):
            self.previous_distance_to_goal = distance

        progress = self.previous_distance_to_goal - distance
        self.previous_distance_to_goal = distance

        # Small step penalty to encourage efficiency
        return (progress * self.progress_multiplier) - self.step_penalty

    def _is_collision(self):
        """
        Sample laser rays, take a proximity_index_offset and average each ray with its "neighboring" rays.
        Check that average distances are below threshold for collision.
        """

        full_laser_scan = self.laser_scan

        if full_laser_scan is None:
            return False
        # Sample laser rays at specified indices
        ranges = np.array(full_laser_scan.ranges, dtype=np.float32)
        if len(ranges) == 0:
            return False
        ranges[np.isinf(ranges)] = full_laser_scan.range_max
        ranges[np.isnan(ranges)] = full_laser_scan.range_max
        
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

    def _get_robot_ground_truth_pose(self):
        """
        Get ground truth robot pose from Gazebo.
        Returns:
            np.array: [x, y, yaw]
        """
        try:
            from gazebo_msgs.srv import GetModelState
            get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = get_state('jackal', '')
            
            x = model_state.pose.position.x
            y = model_state.pose.position.y
            orient = model_state.pose.orientation
            yaw = tf.transformations.euler_from_quaternion(
                [orient.x, orient.y, orient.z, orient.w]
            )[2]
            
            return np.array([x, y, yaw], dtype=np.float32)
        except Exception as e:
            rospy.logwarn(f"Could not get ground truth pose: {e}")
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def _is_on_road(self, y_position):
        """
        Check if a y-coordinate is on the road.
        Args:
            y_position: y-coordinate to check
        Returns:
            bool: True if position is on the road
        """
        return self.road_y_min <= y_position <= self.road_y_max
    
    def _generate_random_goal(self):
        """
        Generate a random goal position with noise that increases over episodes.
        Ensures the goal is within map boundaries and optionally on sidewalk only.
        If spawn_goal_on_sidewalk_only is True, goal spawns on opposite side of road from robot.
        Returns:
            np.array: [x, y] goal position
        """
        # Goal marker has radius 0.3, so add buffer to avoid visual clipping
        marker_radius = self.goal_threshold
        
        # Generate x coordinate with noise, constrained to map boundaries with buffer
        x_min = max(self.map_x_min + marker_radius, self.base_goal_position[0] - self.current_noise_level)
        x_max = min(self.map_x_max - marker_radius, self.base_goal_position[0] + self.current_noise_level)
        goal_x = np.random.uniform(x_min, x_max)
        
        if not self.spawn_goal_on_sidewalk_only:
            # No sidewalk constraint, spawn anywhere within noise range with buffer
            y_min = max(self.map_y_min + marker_radius, self.base_goal_position[1] - self.current_noise_level)
            y_max = min(self.map_y_max - marker_radius, self.base_goal_position[1] + self.current_noise_level)
            goal_y = np.random.uniform(y_min, y_max)
        
            return np.array([goal_x, goal_y], dtype=np.float32)
        
        # If sidewalk only, determine side based on robot position
        # Determine which side of road the robot is on
        robot_y = self.start_position[1]
        
        if robot_y < self.road_y_min:
            # Robot is on lower sidewalk, spawn goal on upper sidewalk
            y_min = max(self.road_y_max + marker_radius, self.base_goal_position[1] - self.current_noise_level)
            y_max = min(self.map_y_max - marker_radius, self.base_goal_position[1] + self.current_noise_level)
        else:
            # Robot is on upper sidewalk (or on road), spawn goal on lower sidewalk
            # Use negative of base position for opposite sidewalk
            opposite_base_y = -self.base_goal_position[1]
            y_min = max(self.map_y_min + marker_radius, opposite_base_y - self.current_noise_level)
            y_max = min(self.road_y_min - marker_radius, opposite_base_y + self.current_noise_level)
        
        goal_y = np.random.uniform(y_min, y_max)
        return np.array([goal_x, goal_y], dtype=np.float32)

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
        
        # Update noise level for next episode
        self.episode_count += 1
        self.current_noise_level = min(
            self.initial_noise_level + (self.episode_count * self.noise_increase_rate),
            self.max_noise_level
        )
        rospy.loginfo(f"Episode {self.episode_count}")
        
        # Generate new random goal for this episode
        self.goal_position = self._generate_random_goal()
        
        # Get initial observation
        obs = self._get_obs()
        
        # Leave sim unpaused so /clock and sensors keep publishing
        rospy.logdebug("End Reset (sim unpaused)")
        
        return obs
