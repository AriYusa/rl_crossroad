#!/usr/bin/env python3
"""
Jackal Robot Environment for OpenAI ROS
This class defines the robot-specific functionality (sensors, actuators, etc.)
Based on openai_ros templates
"""

import rospy
import numpy as np
from gym import spaces
from openai_ros import robot_gazebo_env
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import tf
from cv_bridge import CvBridge
import cv2


class JackalRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Jackal robot environment that interacts with ROS topics and services.
    This class handles low-level robot control and sensor data.
    """

    def __init__(self):
        """
        Initialize ROS connections and robot parameters.
        """
        rospy.logdebug("Start JackalRobotEnv INIT...")
        
        # Define robot namespace (if any)
        self.robot_name_space = ""
        
        # Controllers list - topics to check if they are running
        self.controllers_list = []
        
        # Robot sensors - topics to check
        self.robot_name_space = ""
        self.reset_controls = False
        
        # Start init from parent class
        super(JackalRobotEnv, self).__init__(
            robot_name_space=self.robot_name_space,
            controllers_list=self.controllers_list,
            reset_controls=self.reset_controls,
            start_init_physics_parameters=False,
            reset_world_or_sim="WORLD"
        )
        
        # Define publishers and subscribers
        self._setup_publishers()
        self._setup_subscribers()
        
        # Initialize variables
        self.laser_scan = None
        self.odom = None
        self.traffic_light_state = None
        self.camera_image = None
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Robot parameters
        self.max_linear_speed = 2.0  # m/s
        self.max_angular_speed = 2.0  # rad/s
        
        rospy.logdebug("Finished JackalRobotEnv INIT...")
    
    def _setup_publishers(self):
        """Setup ROS publishers for robot control."""
        self.cmd_vel_pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', 
                                           Twist, queue_size=1)
    
    def _setup_subscribers(self):
        """Setup ROS subscribers for sensor data."""
        rospy.Subscriber('/front/scan', LaserScan, self._laser_scan_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self._odom_callback)
        rospy.Subscriber('/traffic_light_left/state', Int32, self._traffic_light_callback)
        rospy.Subscriber('/front/image_raw', Image, self._camera_callback)
        
        # Wait for topics to be ready
        self._check_all_sensors_ready()
    
    def _laser_scan_callback(self, data):
        """Callback for laser scan data."""
        self.laser_scan = data
    
    def _odom_callback(self, data):
        """Callback for odometry data."""
        self.odom = data
    
    def _traffic_light_callback(self, data):
        """Callback for traffic light state."""
        self.traffic_light_state = data.data
    
    def _camera_callback(self, data):
        """Callback for camera image."""
        self.camera_image = data
    
    def _check_all_sensors_ready(self):
        """Check if all sensors are publishing data."""
        rospy.logdebug("Waiting for all sensors to be ready...")
        
        self._check_laser_scan_ready()
        # self._check_odom_ready()  # We use ground truth Gazebo coords instead of odom for simplicity
        self._check_traffic_light_ready()
        self._check_camera_ready()
        
        rospy.logdebug("All sensors are ready!")
    
    def _check_laser_scan_ready(self):
        """Wait for laser scan to be ready."""
        self.laser_scan = None
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message('/front/scan', LaserScan, timeout=5.0)
                rospy.logdebug("Laser scan is ready!")
            except:
                rospy.logdebug("Laser scan not ready yet, retrying...")
    
    def _check_odom_ready(self):
        """Wait for odometry to be ready."""
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message('/odometry/filtered', Odometry, timeout=5.0)
                rospy.logdebug("Odometry is ready!")
            except:
                rospy.logdebug("Odometry not ready yet, retrying...")
    
    def _check_traffic_light_ready(self):
        """Wait for traffic light to be ready."""
        self.traffic_light_state = None
        while self.traffic_light_state is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message('/traffic_light_left/state', Int32, timeout=5.0)
                self.traffic_light_state = msg.data
                rospy.logdebug("Traffic light is ready!")
            except:
                rospy.logdebug("Traffic light not ready yet, retrying...")
    
    def _check_camera_ready(self):
        """Wait for camera to be ready."""
        self.camera_image = None
        while self.camera_image is None and not rospy.is_shutdown():
            try:
                self.camera_image = rospy.wait_for_message('/front/image_raw', Image, timeout=5.0)
                rospy.logdebug("Camera is ready!")
            except:
                rospy.logdebug("Camera not ready yet, retrying...")
    
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True
    
    # Methods that the Task Env will need
    # ----------------------------
    
    def _set_init_pose(self):
        """
        Sets the robot in its init pose.
        This will be called in the reset of the task environment.
        """
        # Stop the robot
        self.move_base(0.0, 0.0, epsilon=0.05, update_rate=10)
        return True
    
    def _init_env_variables(self):
        """
        Inits variables needed to be initialized each time we reset at the start
        of an episode.
        """
        pass
    
    def _set_action(self, action):
        """
        Applies the given action to the simulation.
        Args:
            action: The action to apply (linear and angular velocities)
        """
        rospy.logdebug("Start Set Action ==> " + str(action))
        
        # Create Twist message
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        
        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)
        
        rospy.logdebug("END Set Action ==> " + str(action))
    
    def _get_obs(self):
        """
        Returns the observation.
        This is called in the step method of the task environment.
        """
        rospy.logdebug("Start Get Observation ==>")
        
        obs = {
            'laser_scan': self.get_laser_scan(),
            'odom': self.get_odom(),
            'traffic_light': self.get_traffic_light_state(),
            'camera_image': self.get_camera_image()
        }
        
        rospy.logdebug("Observations ==> " + str(obs))
        
        return obs
    
    def _is_done(self, observations):
        """
        Indicates whether the episode is done.
        This will be overwitten in the task environment.
        """
        return False
    
    def _compute_reward(self, observations, done):
        """
        Calculates the reward.
        This will be overwritten in the task environment.
        """
        return 0
    
    # Robot-specific methods
    # ----------------------------
    
    def get_laser_scan(self):
        """Get current laser scan data."""
        return self.laser_scan
    
    def get_odom(self):
        """Get current odometry data."""
        return self.odom
    
    def get_traffic_light_state(self):
        """Get current traffic light state."""
        return self.traffic_light_state
    
    def get_camera_image(self):
        """Get current camera image."""
        return self.camera_image
    
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("Jackal Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self.cmd_vel_pub.publish(cmd_vel_value)
        # Wait for command to be processed
        rospy.sleep(0.1)
    
    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self.cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is reset, time when backwards.
                pass
        rospy.logdebug("cmd_vel_pub Publisher Connected")
        rospy.logdebug("All Publishers READY")
    
    def set_robot_pose(self, x, y, z, roll, pitch, yaw):
        """
        Move the robot to a specific pose using Gazebo service.
        Args:
            x, y, z: Position coordinates
            roll, pitch, yaw: Orientation angles
        """
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            
            state = ModelState()
            state.model_name = 'jackal'
            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.position.z = z
            
            # Convert RPY to quaternion
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            state.pose.orientation.x = quaternion[0]
            state.pose.orientation.y = quaternion[1]
            state.pose.orientation.z = quaternion[2]
            state.pose.orientation.w = quaternion[3]
            
            # Reset velocities
            state.twist.linear.x = 0
            state.twist.linear.y = 0
            state.twist.linear.z = 0
            state.twist.angular.x = 0
            state.twist.angular.y = 0
            state.twist.angular.z = 0
            
            set_state(state)
            
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
