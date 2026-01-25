#!/usr/bin/env python3
"""
Training script for Jackal Crossroad Environment
This script demonstrates how to use the custom environment with a simple random agent
You can replace this with your preferred RL algorithm (PPO, SAC, DQN, etc.)
"""

import rospy
import gym
import numpy as np
import jackal_crossroad_env  # This registers the environment


def main():
    """
    Main training loop with random agent (replace with your RL algorithm)
    """
    rospy.init_node('jackal_crossroad_training', anonymous=True, log_level=rospy.INFO)
    
    # Create environment using gym.make (requires package installation)
    env = gym.make('JackalCrossroad-v0')
    
    rospy.loginfo("Environment created successfully!")
    rospy.loginfo("Observation space: {}".format(env.observation_space))
    rospy.loginfo("Action space: {}".format(env.action_space))
    
    # Training parameters
    num_episodes = 200
    max_steps_per_episode = 1000
    
    # Statistics
    episode_rewards = []
    
    try:
        for episode in range(num_episodes):
            rospy.loginfo("=" * 50)
            rospy.loginfo("Starting Episode: {}".format(episode + 1))
            
            # Reset environment
            obs = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps_per_episode:
                # REPLACE THIS with your RL agent's action selection
                # For now, we use random actions
                action = env.action_space.sample()
                
                # Take step in environment
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                step += 1
                
                # Log progress every 100 steps
                if step % 100 == 0:
                    rospy.loginfo("Step: {}, Reward: {:.2f}, Cumulated: {:.2f}".format(
                        step, reward, episode_reward))
            
            episode_rewards.append(episode_reward)
            
            rospy.loginfo("Episode {} finished!".format(episode + 1))
            rospy.loginfo("Total Steps: {}".format(step))
            rospy.loginfo("Episode Reward: {:.2f}".format(episode_reward))
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
        else:
            rospy.loginfo("No episodes completed.")


if __name__ == '__main__':
    main()
