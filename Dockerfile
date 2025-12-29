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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws
RUN mkdir -p /catkin_ws/src
COPY ./jackal_robot /catkin_ws/src/jackal_robot

RUN bash -lc "source /opt/ros/noetic/setup.bash && catkin_make"

# Add entrypoint that sources environments
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Source ROS environment in every bash session
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
