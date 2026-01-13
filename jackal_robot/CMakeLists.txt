cmake_minimum_required(VERSION 3.0.2)
project(jackal_robot)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_ros
  tf2_geometry_msgs
  dynamic_reconfigure
)

find_package(gazebo REQUIRED)
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

generate_dynamic_reconfigure_options(
  cfg/GazeboTrafficLight.cfg
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

add_library(gazebo_traffic_light
  src/GazeboTrafficLight.cpp
)
add_dependencies(gazebo_traffic_light
  ${PROJECT_NAME}_gencfg
)
target_link_libraries(gazebo_traffic_light
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)

install(
  TARGETS
    gazebo_traffic_light
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(
  DIRECTORY 
    launch 
    worlds 
    models
    yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
