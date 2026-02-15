FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

ENV JACKAL_LASER=1
ENV JACKAL_FLEA3=1
ENV JACKAL_FLEA3_TILT=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-jackal-simulator \
    ros-noetic-jackal-desktop \
    ros-noetic-teleop-twist-keyboard \
    python3-catkin-tools \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for RL
RUN pip3 install --no-cache-dir \
    gym==0.21.0 \
    numpy \
    rospkg \
    catkin_pkg \
    defusedxml \
    netifaces

WORKDIR /catkin_ws
RUN mkdir -p /catkin_ws/src

# Clone and install openai_ros
RUN cd /catkin_ws/src && \
    git clone https://bitbucket.org/theconstructcore/openai_ros.git

COPY ./jackal_robot /catkin_ws/src/jackal_robot
COPY ./jackal_crossroad_env /catkin_ws/jackal_crossroad_env

RUN bash -lc "source /opt/ros/noetic/setup.bash && catkin_make"

# Install openai_ros dependencies (skip python2 packages that don't exist in focal)
RUN bash -lc "source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && rosdep install -y --from-paths src --ignore-src --skip-keys='python-catkin-pkg'" || true

# Install Python environment package
RUN cd /catkin_ws/jackal_crossroad_env && pip3 install -e .

# Add entrypoint that sources environments
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Source ROS environment in every bash session
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
