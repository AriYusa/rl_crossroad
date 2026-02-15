#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash

# source your workspace if it exists
if [ -f /catkin_ws/devel/setup.bash ]; then
  source /catkin_ws/devel/setup.bash
fi

exec "$@"