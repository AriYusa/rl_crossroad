#!/usr/bin/env python3
"""
Training script for Jackal Crossroad Environment with Geometric Navigation
This agent calculates movement based on coordinates to navigate toward the goal
"""

import rospy
import gym
import numpy as np
import jackal_crossroad_env  # This registers the environment


class GeometricNavigationAgent:
    """
    Simple geometric navigation agent that calculates actions based on 
    robot position, goal position, and obstacle avoidance
    """
    
    def __init__(self, max_linear_vel=1.5, max_angular_vel=1.5):
        """
        Initialize the geometric navigation agent
        
        Args:
            max_linear_vel: Maximum linear velocity (m/s)
            max_angular_vel: Maximum angular velocity (rad/s)
        """
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.obstacle_threshold = 0.8  # Distance threshold for obstacle avoidance (meters)
        self.goal_threshold = 0.3  # Distance threshold for reaching goal
        
    def calculate_action(self, observation):
        """
        Calculate action based on current observation
        
        Args:
            observation: Dictionary containing:
                - robot_coords: [x, y, yaw]
                - goal_coords: [x, y]
                - laser_scan: array of laser distances
                
        Returns:
            action: [linear_velocity, angular_velocity]
        """
        robot_coords = observation['robot_coords']
        goal_coords = observation['goal_coords']
        laser_scan = observation['laser_scan']
        
        # Current robot state
        robot_x, robot_y, robot_yaw = robot_coords
        goal_x, goal_y = goal_coords
        
        # Calculate vector to goal
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        
        # Calculate desired heading (angle to goal)
        desired_heading = np.arctan2(dy, dx)
        
        # Calculate heading error (angle difference)
        heading_error = self._normalize_angle(desired_heading - robot_yaw)
        
        # Check for obstacles in front
        front_obstacle = self._check_front_obstacle(laser_scan)
        
        # Calculate angular velocity (proportional control)
        angular_vel = np.clip(2.0 * heading_error, -self.max_angular_vel, self.max_angular_vel)
        
        # Calculate linear velocity
        if front_obstacle:
            # Obstacle detected - slow down or stop
            if abs(heading_error) > np.pi / 4:
                # Large heading error - turn in place
                linear_vel = 0.0
            else:
                # Small heading error - slow down
                linear_vel = 0.2
        else:
            # No obstacle - move forward
            if abs(heading_error) > np.pi / 3:
                # Large heading error - turn more, move slower
                linear_vel = 0.3
            elif distance_to_goal < 1.0:
                # Close to goal - slow down
                linear_vel = 0.5
            else:
                # Normal operation - move at good speed
                linear_vel = self.max_linear_vel
                
            # Reduce speed based on heading error
            linear_vel *= (1.0 - abs(heading_error) / np.pi)
        
        # Ensure velocities are within bounds
        linear_vel = np.clip(linear_vel, -1.0, 2.0)
        angular_vel = np.clip(angular_vel, -2.0, 2.0)
        
        return np.array([linear_vel, angular_vel], dtype=np.float32)
    
    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi] range
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _check_front_obstacle(self, laser_scan):
        """
        Check if there's an obstacle in front of the robot
        
        Args:
            laser_scan: Array of laser distances
            
        Returns:
            bool: True if obstacle detected in front
        """
        if len(laser_scan) == 0:
            return False
            
        # Check center rays (front of robot)
        num_rays = len(laser_scan)
        center_idx = num_rays // 2
        front_range = 3  # Check 3 rays on each side of center
        
        # Get front laser readings
        start_idx = max(0, center_idx - front_range)
        end_idx = min(num_rays, center_idx + front_range + 1)
        front_readings = laser_scan[start_idx:end_idx]
        
        # Check if any front reading is below threshold
        min_front_distance = np.min(front_readings)
        return min_front_distance < self.obstacle_threshold


def main():
    """
    Main training loop with geometric navigation agent
    """
    rospy.init_node('jackal_geometric_navigation', anonymous=True, log_level=rospy.INFO)
    
    # Create environment
    env = gym.make('JackalCrossroad-v0')
    
    rospy.loginfo("Environment created successfully!")
    rospy.loginfo("Observation space: {}".format(env.observation_space))
    rospy.loginfo("Action space: {}".format(env.action_space))
    
    # Create geometric navigation agent
    agent = GeometricNavigationAgent(max_linear_vel=1.5, max_angular_vel=1.5)
    rospy.loginfo("Geometric Navigation Agent initialized")
    
    # Training parameters
    num_episodes = 200
    max_steps_per_episode = 1000
    
    # Statistics
    episode_rewards = []
    episode_successes = []
    
    try:
        for episode in range(num_episodes):
            rospy.loginfo("=" * 50)
            rospy.loginfo("Starting Episode: {}".format(episode + 1))
            
            # Reset environment
            obs = env.reset()
            episode_reward = 0
            done = False
            step = 0
            reached_goal = False
            
            while not done and step < max_steps_per_episode:
                # Calculate action using geometric navigation
                action = agent.calculate_action(obs)
                
                # Take step in environment
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                step += 1
                
                # Check if goal reached
                robot_coords = obs['robot_coords']
                goal_coords = obs['goal_coords']
                distance_to_goal = np.linalg.norm(robot_coords[:2] - goal_coords)
                
                if distance_to_goal < agent.goal_threshold:
                    reached_goal = True
                
                # Log progress every 100 steps
                if step % 100 == 0:
                    rospy.loginfo("Step: {}, Dist to Goal: {:.2f}m, Reward: {:.2f}, Cumulated: {:.2f}".format(
                        step, distance_to_goal, reward, episode_reward))
            
            episode_rewards.append(episode_reward)
            episode_successes.append(1 if reached_goal else 0)
            
            rospy.loginfo("Episode {} finished!".format(episode + 1))
            rospy.loginfo("Total Steps: {}".format(step))
            rospy.loginfo("Episode Reward: {:.2f}".format(episode_reward))
            rospy.loginfo("Goal Reached: {}".format("YES" if reached_goal else "NO"))
            rospy.loginfo("Success Rate (last 10): {:.1f}%".format(
                np.mean(episode_successes[-10:]) * 100))
            rospy.loginfo("Average Reward (last 10): {:.2f}".format(
                np.mean(episode_rewards[-10:])))
            
    except KeyboardInterrupt:
        rospy.loginfo("Training interrupted by user")
    except Exception as e:
        rospy.logerr("Training crashed: {}".format(e))
        raise
    
    finally:
        rospy.loginfo("Training finished!")
        rospy.loginfo("Total episodes: {}".format(len(episode_rewards)))
        if episode_rewards:
            rospy.loginfo("Average reward: {:.2f}".format(np.mean(episode_rewards)))
            rospy.loginfo("Max reward: {:.2f}".format(np.max(episode_rewards)))
            rospy.loginfo("Min reward: {:.2f}".format(np.min(episode_rewards)))
            rospy.loginfo("Overall success rate: {:.1f}%".format(
                np.mean(episode_successes) * 100))
        else:
            rospy.loginfo("No episodes completed.")


if __name__ == '__main__':
    main()
