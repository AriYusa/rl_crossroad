#!/usr/bin/env python3
"""
Visualize saved laser scan data from collision events.
Usage: python3 visualize_collision.py <path_to_npz_file>
"""

import numpy as np
import matplotlib
# Use non-interactive backend for Docker/headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import glob


def visualize_laser_scan(npz_file):
    """
    Load and visualize a saved laser scan from collision.
    """
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    
    full_ranges = data['full_ranges']
    downsampled_ranges = data['downsampled_ranges']
    angle_min = float(data['angle_min'])
    angle_max = float(data['angle_max'])
    angle_increment = float(data['angle_increment'])
    range_min = float(data['range_min'])
    range_max = float(data['range_max'])
    timestamp = str(data['timestamp'])
    
    # Load collision thresholds if available (for backward compatibility)
    collision_distance_threshold = float(data['collision_distance_threshold']) if 'collision_distance_threshold' in data else 0.15
    collision_ray_threshold = int(data['collision_ray_threshold']) if 'collision_ray_threshold' in data else 2
    num_close_rays = int(data['num_close_rays']) if 'num_close_rays' in data else np.sum(downsampled_ranges < collision_distance_threshold)
    
    # Check if camera image is available
    has_camera_image = 'camera_image' in data
    camera_image = data['camera_image'] if has_camera_image else None
    
    print(f"\n{'='*60}")
    print(f"Collision Laser Scan Data")
    print(f"{'='*60}")
    print(f"Timestamp: {timestamp}")
    print(f"Number of points: {len(full_ranges)}")
    print(f"Angle range: {np.rad2deg(angle_min):.1f}° to {np.rad2deg(angle_max):.1f}°")
    print(f"Range: {range_min:.2f}m to {range_max:.2f}m")
    print(f"Min distance detected: {np.min(full_ranges):.3f}m")
    print(f"Collision thresholds:")
    print(f"  - Distance threshold: {collision_distance_threshold:.2f}m")
    print(f"  - Ray threshold: {collision_ray_threshold} rays")
    print(f"  - Close rays detected: {num_close_rays}")
    print(f"Camera image: {'Available' if has_camera_image else 'Not available'}")
    if has_camera_image:
        print(f"Image shape: {camera_image.shape}")
    print(f"{'='*60}\n")
    
    # Calculate angles for full scan
    num_points = len(full_ranges)
    angles = np.linspace(angle_min, angle_max, num_points)
    
    # Calculate angles for downsampled scan
    num_downsampled = len(downsampled_ranges)
    downsampled_angles = np.linspace(angle_min, angle_max, num_downsampled)
    
    # Create figure with subplots (3x2 if camera image available, otherwise 2x2)
    if has_camera_image:
        fig = plt.figure(figsize=(15, 15))
        num_rows = 3
    else:
        fig = plt.figure(figsize=(15, 10))
        num_rows = 2
    
    # 1. Polar plot (bird's eye view)
    ax1 = plt.subplot(2, 2, 1, projection='polar')
    ax1.plot(angles, full_ranges, 'b.', markersize=2, label='Full scan', alpha=0.5)
    ax1.plot(downsampled_angles, downsampled_ranges, 'ro', markersize=8, 
             label='Downsampled (used in RL)', alpha=0.7)
    ax1.set_ylim(0, min(range_max, 10))
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title(f'Laser Scan - Bird\'s Eye View\n{timestamp}', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.grid(True)
    
    # 2. Cartesian plot (distance vs angle)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(np.rad2deg(angles), full_ranges, 'b-', linewidth=0.5, 
             label='Full scan', alpha=0.5)
    ax2.plot(np.rad2deg(downsampled_angles), downsampled_ranges, 'ro-', 
             markersize=6, linewidth=2, label='Downsampled', alpha=0.7)
    ax2.axhline(y=collision_distance_threshold, color='r', linestyle='--', 
                label=f'Collision threshold ({collision_distance_threshold:.2f}m)')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Distance (meters)')
    ax2.set_title('Distance vs Angle')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, min(range_max, 10))
    
    # 3. XY coordinates (top-down view)
    ax3 = plt.subplot(2, 2, 3)
    x_full = full_ranges * np.cos(angles)
    y_full = full_ranges * np.sin(angles)
    x_down = downsampled_ranges * np.cos(downsampled_angles)
    y_down = downsampled_ranges * np.sin(downsampled_angles)
    
    ax3.plot(x_full, y_full, 'b.', markersize=2, label='Full scan', alpha=0.5)
    ax3.plot(x_down, y_down, 'ro', markersize=8, label='Downsampled', alpha=0.7)
    ax3.plot(0, 0, 'g^', markersize=15, label='Robot')
    
    # Draw collision threshold circle
    circle = plt.Circle((0, 0), collision_distance_threshold, color='r', fill=False, 
                       linestyle='--', label=f'Collision zone ({collision_distance_threshold:.2f}m)')
    ax3.add_patch(circle)
    
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.set_title('Top-Down View (Robot Frame)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # 4. Camera image (if available)
    if has_camera_image:
        ax4 = plt.subplot(num_rows, 2, 5)
        # Convert BGR to RGB for proper display
        camera_image_rgb = camera_image[:, :, ::-1] if len(camera_image.shape) == 3 else camera_image
        ax4.imshow(camera_image_rgb)
        ax4.set_title('Camera View at Collision')
        ax4.axis('off')
        
        # Statistics in subplot 6
        ax5 = plt.subplot(num_rows, 2, 6)
    else:
        # Statistics in subplot 4 if no camera
        ax5 = plt.subplot(num_rows, 2, 4)
    
    ax5.axis('off')
    
    # Calculate statistics
    close_points_full = np.sum(full_ranges < collision_distance_threshold)
    close_points_down = np.sum(downsampled_ranges < collision_distance_threshold)
    
    stats_text = f"""
    COLLISION STATISTICS
    {'='*40}
    
    Timestamp: {timestamp}
    
    Full Scan:
      - Total points: {len(full_ranges)}
      - Min distance: {np.min(full_ranges):.3f} m
      - Points < {collision_distance_threshold:.2f}m: {close_points_full}
      - Mean distance: {np.mean(full_ranges):.3f} m
    
    Downsampled (RL Input):
      - Total points: {len(downsampled_ranges)}
      - Min distance: {np.min(downsampled_ranges):.3f} m
      - Points < {collision_distance_threshold:.2f}m: {close_points_down}
      - Mean distance: {np.mean(downsampled_ranges):.3f} m
    
    Collision Criteria:
      - Distance threshold: {collision_distance_threshold:.2f} m
      - Required rays: {collision_ray_threshold}
      - Detected rays: {num_close_rays}
      - Triggered: {'YES' if close_points_down >= collision_ray_threshold else 'NO'}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontfamily='monospace', 
            verticalalignment='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_file = npz_file.replace('.npz', '.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    print(f"Download this file to view the laser scan visualization.")
    
    # Close plot to free memory
    plt.close()


def list_collision_files(directory="/tmp/collision_data"):
    """List all collision data files."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
    
    files = sorted(glob.glob(os.path.join(directory, "collision_*.npz")))
    return files


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Visualize specific file
        npz_file = sys.argv[1]
        if os.path.exists(npz_file):
            visualize_laser_scan(npz_file)
        else:
            print(f"File not found: {npz_file}")
    else:
        # List available files
        files = list_collision_files()
        if files:
            print(f"\nFound {len(files)} collision data file(s):")
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f}")
            print(f"\nTo visualize, run:")
            print(f"  python3 visualize_collision.py <file_path>")
            print(f"\nVisualizing most recent file...")
            visualize_laser_scan(files[-1])
        else:
            print("\nNo collision data files found in /tmp/collision_data/")
            print("Run training and wait for a collision to occur.")
