#!/usr/bin/env python3
"""
Custom Video Recorder for Jackal Crossroad Environment
Records videos from ROS camera topics and logs them to wandb
"""

import os
import cv2
import numpy as np
import rospy
import threading
from datetime import datetime
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from stable_baselines3.common.callbacks import BaseCallback
import wandb


class ROSVideoRecorder:
    """
    Records video from ROS camera topics.
    Subscribes to both robot camera and overhead camera topics.
    """
    
    def __init__(self, robot_topic="/front/image_raw", overhead_topic="/overhead_camera/image_raw"):
        """
        Initialize the video recorder.
        
        Args:
            robot_topic: ROS topic for robot camera
            overhead_topic: ROS topic for overhead camera
        """
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # Camera topics
        self.robot_topic = robot_topic
        self.overhead_topic = overhead_topic
        
        # Latest frames
        self.robot_frame = None
        self.overhead_frame = None
        
        # Recording state
        self.is_recording = False
        self.robot_frames = []
        self.overhead_frames = []
        
        # Subscribe to camera topics
        self.robot_sub = rospy.Subscriber(
            self.robot_topic, 
            Image, 
            self._robot_camera_callback,
            queue_size=1,
            buff_size=2**24  # 16MB buffer
        )
        self.overhead_sub = rospy.Subscriber(
            self.overhead_topic, 
            Image, 
            self._overhead_camera_callback,
            queue_size=1,
            buff_size=2**24  # 16MB buffer
        )
        
        rospy.loginfo(f"Video recorder initialized. Subscribed to {robot_topic} and {overhead_topic}")
    
    def _robot_camera_callback(self, msg):
        """Callback for robot camera images."""
        # Skip processing if not recording (avoid overhead)
        if not self.is_recording:
            return
            
        try:
            # Convert ROS Image to OpenCV format (only when recording)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.lock:
                self.robot_frame = cv_image
                self.robot_frames.append(cv_image.copy())
        except Exception as e:
            rospy.logwarn(f"Error in robot camera callback: {e}")
    
    def _overhead_camera_callback(self, msg):
        """Callback for overhead camera images."""
        # Skip processing if not recording (avoid overhead)
        if not self.is_recording:
            return
            
        try:
            # Convert ROS Image to OpenCV format (only when recording)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.lock:
                self.overhead_frame = cv_image
                self.overhead_frames.append(cv_image.copy())
        except Exception as e:
            rospy.logwarn(f"Error in overhead camera callback: {e}")
    
    def start_recording(self):
        """Start recording video."""
        with self.lock:
            self.is_recording = True
            self.robot_frames = []
            self.overhead_frames = []
        rospy.logdebug("Started video recording")
    
    def stop_recording(self):
        """Stop recording video and return frames."""
        with self.lock:
            self.is_recording = False
            robot_frames = self.robot_frames.copy()
            overhead_frames = self.overhead_frames.copy()
            self.robot_frames = []
            self.overhead_frames = []
        
        rospy.logdebug(f"Stopped video recording. Robot frames: {len(robot_frames)}, Overhead frames: {len(overhead_frames)}")
        return robot_frames, overhead_frames
    
    def cleanup(self):
        """Cleanup subscribers."""
        self.robot_sub.unregister()
        self.overhead_sub.unregister()


class VideoRecorderCallback(BaseCallback):
    """
    Callback for recording and logging videos to wandb.
    Records videos every N episodes.
    """
    
    def __init__(
        self,
        video_dir="./sac_videos",
        record_freq=1000,  # Record every N episodes
        fps=10,
        use_wandb=True,
        robot_topic="/front/image_raw",
        overhead_topic="/overhead_camera/image_raw",
        verbose=0
    ):
        """
        Initialize the video recorder callback.
        
        Args:
            video_dir: Directory to save videos
            record_freq: Record video every N episodes
            fps: Frames per second for output video
            use_wandb: Whether to log videos to wandb
            robot_topic: ROS topic for robot camera
            overhead_topic: ROS topic for overhead camera
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        self.record_freq = record_freq
        self.fps = fps
        self.use_wandb = use_wandb
        
        # Episode tracking
        self.episode_count = 0
        self.should_record = False
        
        # Initialize ROS video recorder
        self.recorder = ROSVideoRecorder(robot_topic, overhead_topic)
        
        rospy.loginfo(f"VideoRecorderCallback initialized. Recording every {record_freq} episodes.")

    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        # Start recording for episode 0 if needed
        if self.episode_count == 0 and self.record_freq > 0:
            self.should_record = True
            self.recorder.start_recording()
            rospy.loginfo(f"Started recording episode {self.episode_count}")
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check for episode completion
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    # Episode ended
                    if self.should_record:
                        # Stop recording and save video
                        robot_frames, overhead_frames = self.recorder.stop_recording()
                        self._save_and_log_video(robot_frames, overhead_frames, self.episode_count)
                        self.should_record = False
                    
                    # Increment episode counter
                    self.episode_count += 1
                    
                    # Check if we should record next episode
                    if self.episode_count % self.record_freq == 0:
                        self.should_record = True
                        self.recorder.start_recording()
                        rospy.loginfo(f"Started recording episode {self.episode_count}")
        
        return True
    
    def _save_and_log_video(self, robot_frames, overhead_frames, episode_num):
        """
        Save video frames to mp4 and log to wandb.
        
        Args:
            robot_frames: List of robot camera frames
            overhead_frames: List of overhead camera frames
            episode_num: Episode number
        """
        if len(robot_frames) == 0 and len(overhead_frames) == 0:
            rospy.logwarn(f"No frames recorded for episode {episode_num}")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save robot camera video
        if len(robot_frames) > 0:
            robot_video_path = self.video_dir / f"episode_{episode_num}_robot_{timestamp}.mp4"
            self._save_video(robot_frames, robot_video_path, "Robot Camera")
            
            # Log to wandb using numpy array for browser compatibility
            if self.use_wandb and wandb.run is not None:
                try:
                    # Convert frames to numpy array in (T, C, H, W) format for wandb
                    frames_array = self._frames_to_wandb_format(robot_frames)
                    wandb.log({
                        f"video/robot": wandb.Video(
                            frames_array, 
                            fps=self.fps, 
                            format="mp4"
                        ),
                        "train/timestep": self.num_timesteps,
                        "episode/number": episode_num,
                    })
                except Exception as e:
                    rospy.logwarn(f"Failed to log robot video to wandb: {e}")
        
        # Save overhead camera video
        if len(overhead_frames) > 0:
            overhead_video_path = self.video_dir / f"episode_{episode_num}_overhead_{timestamp}.mp4"
            self._save_video(overhead_frames, overhead_video_path, "Overhead Camera")
            
            # Log to wandb using numpy array for browser compatibility
            if self.use_wandb and wandb.run is not None:
                try:
                    # Convert frames to numpy array in (T, C, H, W) format for wandb
                    frames_array = self._frames_to_wandb_format(overhead_frames)
                    wandb.log({
                        f"video/overhead": wandb.Video(
                            frames_array, 
                            fps=self.fps, 
                            format="mp4"
                        ),
                        "train/timestep": self.num_timesteps,
                        "episode/number": episode_num,
                    })
                except Exception as e:
                    rospy.logwarn(f"Failed to log overhead video to wandb: {e}")
        
        # Create side-by-side comparison video if both cameras have frames
        if len(robot_frames) > 0 and len(overhead_frames) > 0:
            combined_video_path = self.video_dir / f"episode_{episode_num}_combined_{timestamp}.mp4"
            combined_frames = self._create_combined_frames(robot_frames, overhead_frames)
            self._save_video(combined_frames, combined_video_path, "")
            
            # Log to wandb using numpy array for browser compatibility
            if self.use_wandb and wandb.run is not None:
                try:
                    # Convert frames to numpy array in (T, C, H, W) format for wandb
                    frames_array = self._frames_to_wandb_format(combined_frames)
                    wandb.log({
                        f"video/combined": wandb.Video(
                            frames_array, 
                            fps=self.fps, 
                            format="mp4"
                        ),
                        "train/timestep": self.num_timesteps,
                        "episode/number": episode_num,
                    })
                except Exception as e:
                    rospy.logwarn(f"Failed to log combined video to wandb: {e}")
        
        rospy.loginfo(f"Saved videos for episode {episode_num} to {self.video_dir}")
    
    def _save_video(self, frames, output_path, title=""):
        """
        Save frames as mp4 video.
        
        Args:
            frames: List of frames (numpy arrays)
            output_path: Path to save video
            title: Title to display on video (optional)
        """
        if len(frames) == 0:
            return
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Define codec and create VideoWriter
        # Use mp4v codec for local file saving
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in frames:
            # Add title text if provided
            if title:
                frame = frame.copy()
                cv2.putText(
                    frame, title, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )
            out.write(frame)
        
        out.release()
        rospy.logdebug(f"Saved video to {output_path}")
    
    def _frames_to_wandb_format(self, frames):
        """
        Convert list of BGR frames to numpy array in wandb format.
        wandb expects (T, C, H, W) with RGB channel order.
        
        Args:
            frames: List of BGR frames (H, W, C)
        
        Returns:
            numpy array in (T, C, H, W) format with RGB channels
        """
        # Convert BGR to RGB and transpose to (C, H, W)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for f in frames]
        # Stack into (T, C, H, W)
        return np.stack(rgb_frames, axis=0).astype(np.uint8)
    
    def _create_combined_frames(self, robot_frames, overhead_frames):
        """
        Create side-by-side combined frames from robot and overhead cameras.
        
        Args:
            robot_frames: List of robot camera frames
            overhead_frames: List of overhead camera frames
        
        Returns:
            List of combined frames
        """
        if len(robot_frames) == 0 or len(overhead_frames) == 0:
            return []
        
        # Use minimum number of frames
        num_frames = min(len(robot_frames), len(overhead_frames))
        
        # Get frame dimensions
        robot_h, robot_w = robot_frames[0].shape[:2]
        overhead_h, overhead_w = overhead_frames[0].shape[:2]
        
        # Resize frames to same height for side-by-side display
        target_height = 480
        robot_aspect = robot_w / robot_h
        overhead_aspect = overhead_w / overhead_h
        robot_target_width = int(target_height * robot_aspect)
        overhead_target_width = int(target_height * overhead_aspect)
        
        combined_frames = []
        for i in range(num_frames):
            # Resize frames
            robot_resized = cv2.resize(robot_frames[i], (robot_target_width, target_height))
            overhead_resized = cv2.resize(overhead_frames[i], (overhead_target_width, target_height))
            
            # Add labels
            cv2.putText(
                robot_resized, "Robot View", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                overhead_resized, "Overhead View", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Concatenate horizontally
            combined = np.hstack([robot_resized, overhead_resized])
            combined_frames.append(combined)
        
        return combined_frames
    
    def _on_training_end(self):
        """Called at the end of training."""
        # Cleanup recorder
        self.recorder.cleanup()
        rospy.loginfo("Video recorder cleaned up")
