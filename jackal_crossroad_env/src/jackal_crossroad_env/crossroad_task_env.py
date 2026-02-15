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
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SpawnModel, SetModelState


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

        # Crosswalk corridor geometry (around x center line)
        self.crosswalk_center_x = 0.0
        self.crosswalk_half_width = 1.0
        self.subgoal_sidewalk_offset = 0.3
        self.waiting_zone_distance = 0.4
        
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
        self.rule_violation_penalty = -120.0  # Crossing on red or outside crosswalk
        self.goal_reward = 100.0  # Reward for reaching goal
        self.subgoal_reward = 25.0  # Reward for reaching intermediate goals
        self.progress_multiplier = 10.0  # Multiplier for progress towards goal
        self.waiting_reward = 0.05  # Reward for waiting at red light near stop line
        self.step_penalty = 0.01  # Penalty for each step (encourages efficiency)

        # Traffic light encoding for this simulation: red=-1, green=1
        self.green_light_states = {1}
        self.treat_unknown_light_as_red = True

        # Waiting detection
        self.waiting_speed_threshold = 0.05
        
        # Episode parameters
        self.max_episode_steps = 1000
        self.goal1_timeout_seconds = 60.0
        self.goal_distance_log_interval_seconds = 5.0
        self.current_step = 0
        self.done_reason = None
        self.rule_violation = False
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.goal1_deadline_time = None
        self.next_goal_distance_log_time = None

        # Multi-goal stage tracking
        self.stage_goals = []
        self.current_goal_idx = 0
        self.active_goal_position = self.goal_position.astype(np.float32)
        self.previous_distance_to_active_goal = None
        self.crossing_direction = 1.0

        # Visualization for goals in Gazebo
        self.goal_marker_names = [
            "goal_stage_1_marker",
            "goal_stage_2_marker",
            "goal_stage_3_marker",
        ]
        self.goal_marker_colors = [
            (0.15, 0.45, 1.0, 0.95),  # blue
            (1.0, 0.65, 0.1, 0.95),   # orange
            (0.2, 0.95, 0.35, 0.95),  # green
        ]
        self.goal_marker_height = 0.05
        self.goal_marker_z = self.goal_marker_height / 2.0
        self.subgoal_marker_size_x = max(2.0, 2.0 * self.crosswalk_half_width)
        self.subgoal_marker_size_y = 0.9
        self.final_goal_marker_radius = 0.25
        self.goal_marker_specs = [
            {
                "shape": "box",
                "size": (self.subgoal_marker_size_x, self.subgoal_marker_size_y, self.goal_marker_height),  # rect for crosswalk
            },
            {
                "shape": "box",
                "size": (self.subgoal_marker_size_x, self.subgoal_marker_size_y, self.goal_marker_height),  # rect for crosswalk
            },
            {
                "shape": "cylinder",    # cyl for endgoal
                "radius": self.final_goal_marker_radius,
                "length": self.goal_marker_height,
            },
        ]
        
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
        self.done_reason = None
        self.rule_violation = False
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.previous_distance_to_active_goal = None
        self.goal1_deadline_time = rospy.get_time() + self.goal1_timeout_seconds
        self.next_goal_distance_log_time = rospy.get_time() + self.goal_distance_log_interval_seconds
    
    def _set_action(self, action):
        """
        Apply action to the robot.
        Args:
            action: Array [linear_velocity, angular_velocity]
        """
        
        # Clip actions to be within limits
        linear_vel = np.clip(action[0], -1.0, 2.0)
        angular_vel = np.clip(action[1], -2.0, 2.0)
        self.last_action = np.array([linear_vel, angular_vel], dtype=np.float32)
        
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
        
        # goal coords; current goal depends on current state of goals reached (subgoals + endgoal) 
        goal_coords = self.active_goal_position.astype(np.float32)
        
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

        self.done_reason = None
        self.rule_violation = False

        if self._is_collision():
            self.done_reason = "collision"
            rospy.loginfo("Episode done: Collision detected")
            return True

        violated, reason = self._check_rule_violation(observations)
        if violated:
            self.done_reason = reason
            self.rule_violation = True
            rospy.loginfo(f"Episode done: Rule violation ({reason})")
            return True

        if self.current_goal_idx == 0 and not self._is_active_goal_reached(observations) and self._has_goal1_timed_out():   # timout for goal 1
            self.done_reason = "goal1_timeout"
            rospy.loginfo("Episode done: Goal 1 timeout (60s)")
            return True

        if self._is_active_goal_reached(observations) and self.current_goal_idx == len(self.stage_goals) - 1:
            self.done_reason = "goal_reached"
            rospy.loginfo("Episode done: Goal reached")
            return True

        if self.current_step >= self.max_episode_steps:
            self.done_reason = "max_steps"
            rospy.loginfo("Episode done: Maximum steps reached")
            return True
        return False

    def _get_robot_xy(self, observations):
        """Extract robot x,y coordinates from observation dict."""
        robot_coords = observations.get("robot_coords", np.zeros(3, dtype=np.float32))
        return np.asarray(robot_coords[:2], dtype=np.float32)

    def _manhattan_distance_to_active_goal(self, observations):
        """L1 distance to current active goal."""
        robot_xy = self._get_robot_xy(observations)
        return float(np.abs(robot_xy - self.active_goal_position).sum())

    def _euclidean_distance_to_active_goal(self, observations):
        """L2 distance to current active goal."""
        robot_xy = self._get_robot_xy(observations)
        return float(np.linalg.norm(robot_xy - self.active_goal_position))

    def _has_goal1_timed_out(self):
        """Check whether the stage-1 timeout has elapsed."""
        if self.goal1_deadline_time is None:
            return False
        return rospy.get_time() >= self.goal1_deadline_time

    def _maybe_log_goal_distance(self, observations):
        """Log distance to the active goal at a fixed interval."""
        if observations is None or not isinstance(observations, dict):
            return
        if self.next_goal_distance_log_time is None:
            self.next_goal_distance_log_time = rospy.get_time() + self.goal_distance_log_interval_seconds
            return

        now = rospy.get_time()
        if now < self.next_goal_distance_log_time:
            return

        distance = self._euclidean_distance_to_active_goal(observations)
        rospy.loginfo(
            "Distance to next goal (stage %d/%d): %.2f m",
            self.current_goal_idx + 1,
            len(self.stage_goals),
            distance,
        )
        while self.next_goal_distance_log_time <= now:
            self.next_goal_distance_log_time += self.goal_distance_log_interval_seconds

    def _is_active_goal_reached(self, observations):
        """Check if current active goal is reached.

        Subgoals (stage 1/2) use rectangular hitboxes that match marker size.
        Final goal (stage 3) uses circle.
        """
        robot_xy = self._get_robot_xy(observations)

        # Stage 1 and 2: rectangular zone
        if self.current_goal_idx in (0, 1):
            half_x = self.subgoal_marker_size_x / 2.0
            half_y = self.subgoal_marker_size_y / 2.0
            dx = abs(float(robot_xy[0]) - float(self.active_goal_position[0]))
            dy = abs(float(robot_xy[1]) - float(self.active_goal_position[1]))
            return dx <= half_x and dy <= half_y

        # Final stage: circular zone
        distance = np.linalg.norm(robot_xy - self.active_goal_position)
        return distance < self.goal_threshold

    def _is_light_green(self):  # for violation checking
        """Return True if traffic light is green according to configured states."""
        if self.traffic_light_state is None:
            return not self.treat_unknown_light_as_red
        try:
            return int(self.traffic_light_state) in self.green_light_states
        except (TypeError, ValueError):
            return False

    def _is_within_crosswalk(self, x_position):
        """Check whether x is within the crosswalk corridor."""
        return abs(float(x_position) - self.crosswalk_center_x) <= self.crosswalk_half_width

    def _check_rule_violation(self, observations):
        """
        Check if agent violates crossing rules.
        Returns:
            (bool, str): violation flag and reason.
        """
        robot_x, robot_y = self._get_robot_xy(observations)
        on_road = self._is_on_road(robot_y)

        if not on_road:
            return False, ""

        if not self._is_within_crosswalk(robot_x):
            return True, "off_crosswalk"

        if not self._is_light_green():
            return True, "red_light_violation"

        return False, ""

    def _compute_progress_reward(self, observations):
        """Reward based on Manhattan-distance progress to the active goal."""
        distance = self._manhattan_distance_to_active_goal(observations)
        if self.previous_distance_to_active_goal is None:
            self.previous_distance_to_active_goal = distance
            return 0.0

        progress = self.previous_distance_to_active_goal - distance
        self.previous_distance_to_active_goal = distance
        return progress * self.progress_multiplier

    def _is_waiting_at_red_light(self, observations):       # needs a lot of training
        """Detect waiting behavior near stop line while light is red."""
        if self.current_goal_idx != 0:
            return False
        if self._is_light_green():
            return False

        robot_x, robot_y = self._get_robot_xy(observations)
        if self._is_on_road(robot_y):
            return False
        if not self._is_within_crosswalk(robot_x):
            return False

        if self.crossing_direction >= 0:
            near_stop_line = (self.road_y_min - self.waiting_zone_distance) <= robot_y <= self.road_y_min
        else:
            near_stop_line = self.road_y_max <= robot_y <= (self.road_y_max + self.waiting_zone_distance)

        is_stationary = abs(float(self.last_action[0])) <= self.waiting_speed_threshold
        return near_stop_line and is_stationary

    def _compute_waiting_reward(self, observations):
        """Reward patient behavior while waiting for green light."""
        return self.waiting_reward if self._is_waiting_at_red_light(observations) else 0.0

    def _advance_goal_stage_if_reached(self, observations):
        """
        Move to next subgoal when current subgoal is reached.
        Returns:
            float: subgoal bonus reward.
        """
        if not self._is_active_goal_reached(observations):
            return 0.0
        if self.current_goal_idx >= len(self.stage_goals) - 1:
            return 0.0

        self.current_goal_idx += 1
        self.active_goal_position = self.stage_goals[self.current_goal_idx].astype(np.float32)
        self.previous_distance_to_active_goal = None
        rospy.loginfo(f"Advanced to goal stage {self.current_goal_idx + 1}/{len(self.stage_goals)}")
        return self.subgoal_reward

    def _compute_terminal_reward(self):
        """Compute terminal reward from the stored done reason."""
        if self.done_reason == "collision":
            return self.collision_penalty
        if self.done_reason in ("off_crosswalk", "red_light_violation"):
            return self.rule_violation_penalty
        if self.done_reason == "goal_reached":
            return self.goal_reward
        return -self.step_penalty

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

        if done:
            return self._compute_terminal_reward()

        reward = self._compute_progress_reward(observations)        # add all rewards
        reward += self._compute_waiting_reward(observations)
        reward += self._advance_goal_stage_if_reached(observations)
        reward -= self.step_penalty
        return reward

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

    def _initialize_multi_goal_path(self):
        """Create staged goals: start-side curb -> opposite curb -> final randomized goal."""
        # align crosswalk center with path midpoint.
        midpoint_x = float((self.start_position[0] + self.goal_position[0]) / 2.0)
        self.crosswalk_center_x = float(np.clip(midpoint_x, self.map_x_min, self.map_x_max))

        # determine crossing direction from start side to final goal side.
        start_y = float(self.start_position[1])
        goal_y = float(self.goal_position[1])
        self.crossing_direction = 1.0 if goal_y >= start_y else -1.0

        if self.crossing_direction >= 0:
            start_side_y = self.road_y_min - self.subgoal_sidewalk_offset
            opposite_side_y = self.road_y_max + self.subgoal_sidewalk_offset
        else:
            start_side_y = self.road_y_max + self.subgoal_sidewalk_offset
            opposite_side_y = self.road_y_min - self.subgoal_sidewalk_offset

        goal_start_side = np.array([self.crosswalk_center_x, start_side_y], dtype=np.float32)
        goal_opposite_side = np.array([self.crosswalk_center_x, opposite_side_y], dtype=np.float32)
        goal_final = self.goal_position.astype(np.float32)

        self.stage_goals = [goal_start_side, goal_opposite_side, goal_final]
        self.current_goal_idx = 0
        self.active_goal_position = self.stage_goals[self.current_goal_idx].copy()
        self.previous_distance_to_active_goal = None

    def _build_geometry_xml(self, marker_spec):
        """Method for goal markers"""
        shape = marker_spec["shape"]
        if shape == "box":
            sx, sy, sz = marker_spec["size"]
            return f"<box><size>{sx} {sy} {sz}</size></box>"
        if shape == "cylinder":
            radius = marker_spec["radius"]
            length = marker_spec["length"]
            return f"<cylinder><radius>{radius}</radius><length>{length}</length></cylinder>"
        raise ValueError(f"Unsupported marker shape: {shape}")

    def _build_goal_marker_sdf(self, model_name, color_rgba, marker_spec):
        """Method for static goal markers"""
        r, g, b, a = color_rgba
        geometry_xml = self._build_geometry_xml(marker_spec)
        return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='{model_name}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          {geometry_xml}
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <emissive>{r} {g} {b} {a}</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

    def _marker_pose(self, goal_xy):
        """Method for converting 2D goal into 3D gazebo pose"""
        return Pose(
            position=Point(float(goal_xy[0]), float(goal_xy[1]), float(self.goal_marker_z)),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        )

    def _set_goal_marker_pose(self, model_name, goal_xy):
        """Move an existing marker model. Returns True on success."""
        try:
            set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            model_state = ModelState()
            model_state.model_name = model_name
            model_state.pose = self._marker_pose(goal_xy)
            model_state.reference_frame = "world"
            response = set_model_state(model_state)
            return bool(getattr(response, "success", False))
        except rospy.ServiceException:
            return False

    def _spawn_goal_marker(self, model_name, goal_xy, color_rgba, marker_spec):
        """Spawn a single goal marker model in Gazebo."""
        marker_xml = self._build_goal_marker_sdf(model_name, color_rgba, marker_spec)
        marker_pose = self._marker_pose(goal_xy)
        try:
            spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            response = spawn_model(model_name, marker_xml, "", marker_pose, "world")
            if getattr(response, "success", True):
                return True
            status = getattr(response, "status_message", "")
            return "exist" in status.lower()
        except rospy.ServiceException:
            return False

    def _upsert_goal_marker(self, model_name, goal_xy, color_rgba, marker_spec):
        """
        Robust marker update:
        - move if model exists
        - otherwise spawn, then move
        """
        for _ in range(3):
            if self._set_goal_marker_pose(model_name, goal_xy):
                return True
            if self._spawn_goal_marker(model_name, goal_xy, color_rgba, marker_spec):
                if self._set_goal_marker_pose(model_name, goal_xy):
                    return True
            rospy.sleep(0.05)
        return False

    def _update_goal_markers_in_gazebo(self):
        """Spawn/update markers for all staged goals."""
        if len(self.stage_goals) < 3:
            return

        try:
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=2.0)
            rospy.wait_for_service("/gazebo/set_model_state", timeout=2.0)
        except rospy.ROSException as e:
            rospy.logwarn(f"Goal markers disabled: Gazebo services unavailable ({e})")
            return

        for i, marker_name in enumerate(self.goal_marker_names):
            goal_xy = self.stage_goals[i]
            color = self.goal_marker_colors[i]
            marker_spec = self.goal_marker_specs[i]
            if not self._upsert_goal_marker(marker_name, goal_xy, color, marker_spec):
                rospy.logwarn(f"Failed to place/update marker '{marker_name}'")

        rospy.loginfo(
            "Goal markers placed: g1=(%.2f, %.2f), g2=(%.2f, %.2f), g3=(%.2f, %.2f)"
            % (
                self.stage_goals[0][0], self.stage_goals[0][1],
                self.stage_goals[1][0], self.stage_goals[1][1],
                self.stage_goals[2][0], self.stage_goals[2][1],
            )
        )

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

        # Timed debug log for distance to current active goal
        self._maybe_log_goal_distance(obs)
        
        # Update counters
        self.current_step += 1
        self.cumulated_reward += reward
        
        # Info dict
        info = {
            'cumulated_reward': self.cumulated_reward,
            'step': self.current_step,
            'goal_stage': self.current_goal_idx,
            'num_goal_stages': len(self.stage_goals),
            'done_reason': self.done_reason,
            'rule_violation': self.rule_violation,
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
        
        # Update noise level for next episode
        self.episode_count += 1
        self.current_noise_level = min(
            self.initial_noise_level + (self.episode_count * self.noise_increase_rate),
            self.max_noise_level
        )
        rospy.loginfo(f"Episode {self.episode_count}")
        
        # Generate new random goal for this episode
        self.goal_position = self._generate_random_goal()
        self._initialize_multi_goal_path()
        self._update_goal_markers_in_gazebo()
        
        # Get initial observation
        obs = self._get_obs()
        
        # Leave sim unpaused so /clock and sensors keep publishing
        rospy.logdebug("End Reset (sim unpaused)")
        
        return obs
