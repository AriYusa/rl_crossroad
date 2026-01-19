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
docker run -it --rm -e DISPLAY=<your_ip>:0 ros_noetic:1
```

Example: `docker run -it --rm -e DISPLAY=172.19.192.1:0 ros_noetic:1`

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

### Train with Random Agent

In a new terminal (inside Docker):

```bash
# First, install the environment package (one time only)
cd /catkin_ws/jackal_crossroad_env
pip3 install -e .

# Terminal 1: Launch simulation
roslaunch jackal_robot crossroad.launch

# Terminal 2: Run training
cd /catkin_ws/jackal_crossroad_env/scripts
python3 train_random_agent.py
```

### Train with PPO (Stable Baselines3)

First, install Stable Baselines3:

```bash
pip3 install stable-baselines3
```

Then run training:

```bash
# Terminal 1: Launch simulation
roslaunch jackal_robot crossroad.launch

# Terminal 2: Run PPO training
cd /catkin_ws/jackal_crossroad_env/scripts
python3 train_ppo.py
```

### Test Trained Model

```bash
# Terminal 1: Launch simulation
roslaunch jackal_robot crossroad.launch

# Terminal 2: Test model
cd /catkin_ws/jackal_crossroad_env/scripts
python3 test_trained_model.py
```

## Environment Details

### Observation Space

The observation is a vector containing:
- **Laser scan** (10 rays): Downsampled from full laser scan
- **Position** (x, y): Robot position in world frame
- **Velocity** (vx, vy): Linear velocities
- **Orientation** (yaw): Robot heading angle
- **Angular velocity**: Rotation rate
- **Distance to goal**: Euclidean distance to target
- **Angle to goal**: Relative angle to target
- **Traffic light state**: 0=Red, 1=Yellow, 2=Green

Total: 19 dimensions

### Action Space

Continuous actions:
- **Linear velocity**: -1.0 to 2.0 m/s
- **Angular velocity**: -2.0 to 2.0 rad/s

### Rewards

- **Reach goal**: +200
- **Collision**: -200
- **Run red light**: -100
- **Per step**: -0.1
- **Distance improvement**: +10 × (distance_decrease)

### Done Conditions

Episode ends when:
- Robot reaches goal (distance < 0.5m)
- Collision detected (laser < 0.3m)
- Max steps reached (1000 steps)

## Customization

Edit parameters in `jackal_crossroad_env/src/jackal_crossroad_env/config.py`:

```python
# Robot limits
MAX_LINEAR_SPEED = 2.0
MAX_ANGULAR_SPEED = 2.0

# Task parameters
START_POSITION = [-5.0, 0.0]
GOAL_POSITION = [10.0, 0.0]

# Rewards
REWARD_REACH_GOAL = 200.0
REWARD_COLLISION = -200.0
REWARD_RUNNING_RED_LIGHT = -100.0
```

## Using Your Own RL Algorithm

The environment follows the OpenAI Gym interface. You can use it with any RL library:

```python
import rospy
from jackal_crossroad_env import jackal_crossroad_task_env

rospy.init_node('training')
env = jackal_crossroad_task_env.JackalCrossroadEnv()

# Standard gym interface
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

Compatible with:
- Stable Baselines3 (PPO, SAC, TD3, etc.)
- Ray RLlib
- TensorFlow Agents
- PyTorch RL libraries
- Custom implementations

## Project Structure

```
rl_crossroad/
├── Dockerfile                          # Docker setup with openai_ros
├── README.md                          # This file
├── ros_entrypoint.sh                  # ROS environment setup
├── jackal_robot/                      # ROS package for simulation
│   ├── launch/crossroad.launch       # Gazebo launch file
│   ├── src/GazeboTrafficLight.*      # Traffic light plugin
│   └── worlds/crossroad.world        # Gazebo world
└── jackal_crossroad_env/              # Python RL environment package
    ├── setup.py                       # Package installation
    ├── src/jackal_crossroad_env/
    │   ├── __init__.py               # Gym registration
    │   ├── jackal_robot_env.py       # Robot environment (sensors/actuators)
    │   ├── jackal_crossroad_task_env.py  # Task environment (RL logic)
    │   └── config.py                 # Configuration parameters
    └── scripts/
        ├── train_random_agent.py     # Random agent example
        ├── train_ppo.py              # PPO training example
        └── test_trained_model.py     # Model testing script
```

## Troubleshooting

**Display issues**: Make sure X server is running and DISPLAY variable is set correctly

**Import errors**: Ensure the environment is installed: `pip3 install -e /catkin_ws/jackal_crossroad_env`

**ROS connection errors**: Make sure `roslaunch jackal_robot crossroad.launch` is running before training

**Simulation slow**: Reduce simulation rate or decrease observation frequency in the code
```
