cmake_minimum_required(VERSION 2.8.3)
project(acrobot_controllers)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS 
  controller_interface
  control_msgs
  urdf
  std_msgs
  controller_manager 
  joint_state_controller
  robot_state_publisher
  message_generation
)

# find_package(Boost REQUIRED)
# find_package(Eigen REQUIRED)

add_service_files(
  FILES
  moveservo.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)



## Declare a catkin package
include_directories(include ${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})

catkin_package(
  # INCLUDE_DIRS include
  #LIBRARIES ${PROJECT_NAME}
  #CATKIN_DEPENDS 
    #controller_interface
    # control_msgs
    # forward_command_controller 
    # urdf
)


## Build library
#add_library(${PROJECT_NAME})

#target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

## Install
# install(DIRECTORY include/${PROJECT_NAME}/
#  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

#install(TARGETS ${PROJECT_NAME}
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)