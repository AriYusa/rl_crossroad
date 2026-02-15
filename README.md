# Jackal Crossroad RL Environment

This project provides a ROS-Gazebo simulation environment for training a Jackal robot to navigate through a crossroad with traffic lights using Reinforcement Learning.

## Architecture

The project uses **openai_ros** structure with two main environment layers:

1. **Robot Environment** (`jackal_robot_env.py`): Handles low-level robot interactions (sensors, actuators, ROS topics)
2. **Task Environment** (`jackal_crossroad_task_env.py`): Defines the RL task (observations, actions, rewards, done conditions)

## Setup

### 1. Build Docker Image

```bash
docker build -t ros_noetic:1 .
```

### 2. Setup X Server (for Windows)

Install an X client (e.g., XLaunch for Windows) and run it with these settings:
- Display number: 0
- Start no client
- Disable access control

### 3. Run Docker Container

Check your IP address (172.xxx....) and run:

```bash
docker run -it --rm --name ros_container \
  -e DISPLAY=<your_ip>:0 \
  -v "$(pwd)/jackal_crossroad_env:/catkin_ws/jackal_crossroad_env" \
  ros_noetic:1
```

Example: `docker run -it --rm --name ros_container -e DISPLAY=172.19.192.1:0 -v "${PWD}/jackal_crossroad_env:/catkin_ws/jackal_crossroad_env" ros_noetic:1`

### 4. Open Another Terminal with the Container

To open an additional terminal in the same running container:

```bash
docker exec -it ros_container bash
```

This allows you to run multiple commands simultaneously in the same container (e.g., one for the simulation and one for training).

## Usage

### Launch Simulation Only

Inside the Docker container:

```bash
roslaunch jackal_robot crossroad.launch
```

To check the traffic light state:
```bash
rostopic echo /traffic_light_left/state
# or
rostopic echo /traffic_light_right/state
```

#### Model Training Options
```bash
# Terminal 1: Launch simulation
roslaunch jackal_robot crossroad.launch

# Terminal 2: Run training
cd /catkin_ws/jackal_crossroad_env/scripts

# Use custom config file
python3 train_sac_agent.py --config ../config/sac_config.yaml

# Disable wandb logging
python3 train_sac_agent.py --no-wandb

# Use lightweight extractor (no image, faster)
python3 train_sac_agent.py --lightweight

# Set training timesteps
python3 train_sac_agent.py --timesteps 100000

# Set random seed
python3 train_sac_agent.py --seed 123
```

## Robot Configuration

### Sensors

- **Laser Scanner**: Front-mounted LIDAR (topic: `/front/scan`)
  - Used for obstacle detection and collision avoidance
  - 20 downsampled rays in observation space
  
- **Camera**: Point Grey Flea3 RGB camera (topic: `/front/image_raw`)
  - Resolution: 640x480 pixels, 3 channels (BGR)
  - Front-facing for visual perception
  - Configured via environment variable: `JACKAL_FLEA3=1`

### Observation Space

Dictionary observation containing:
- **`raw_image`**: RGB camera image (480×640×3 uint8 array)
- **`laser_scan`**: downsamples rays
- **`robot_coords`**: Ground truth robot pose from Gazebo [x, y, yaw] (3 float32 values)
- **`goal_coords`**: Target goal position [x, y] (2 float32 values)

### Action Space

Continuous actions:
- **Linear velocity**: -1.0 to 2.0 m/s
- **Angular velocity**: -2.0 to 2.0 rad/s

### Rewards


### Done Conditions

### SAC Architecture Details

The SAC implementation uses a custom multi-modal feature extractor with **MobileNetV3-Small** pretrained backbone:

```
Input:
├── raw_image (120×160×3) ──→ MobileNetV3-Small (frozen) ──→ 256 features
├── laser_scan (20,) ──────→ MLP ──→ 64 features
└── robot_coords (3,) + goal_coords (2,) ──→ MLP ──→ 32 features

Fusion Layer: Concatenate (352) ──→ MLP ──→ 256 combined features

SAC Networks:
├── Actor: 256 ──→ [256, 256] ──→ action mean/std
├── Critic 1: 256 + action ──→ [256, 256] ──→ Q1
└── Critic 2: 256 + action ──→ [256, 256] ──→ Q2
```