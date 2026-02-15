#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash

# source your workspace if it exists
if [ -f /catkin_ws/devel/setup.bash ]; then
  source /catkin_ws/devel/setup.bash
fi

# Install jackal_crossroad_env in editable mode if mounted and setup.py exists
if [ -f /catkin_ws/jackal_crossroad_env/setup.py ]; then
  echo "Installing jackal_crossroad_env in editable mode..."
  pip3 install -e /catkin_ws/jackal_crossroad_env
fi

exec "$@"