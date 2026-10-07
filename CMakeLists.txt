cmake_minimum_required(VERSION 3.16)
project(active_slam_information LANGUAGES CXX)

# Builds as an ament_cmake package inside a ROS 2 colcon workspace, or as a
# plain CMake project anywhere Eigen is available (no ROS needed for the
# library or its tests).

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(ament_cmake QUIET)

add_library(${PROJECT_NAME}
  src/information_gain.cpp
  src/view_geometry.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
# GCC false positives inside Eigen's own headers.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(${PROJECT_NAME} PRIVATE -Wno-maybe-uninitialized)
endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# ---- tests ------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_executable(test_information_gain test/test_information_gain.cpp)
  target_link_libraries(test_information_gain PRIVATE ${PROJECT_NAME})
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_information_gain PRIVATE -Wno-maybe-uninitialized)
  endif()
  add_test(NAME test_information_gain COMMAND test_information_gain)
endif()

# ---- ROS 2 node (only when built in a ROS workspace with the messages) -------
find_package(rclcpp QUIET)
find_package(active_slam_msgs QUIET)
if(ament_cmake_FOUND AND rclcpp_FOUND AND active_slam_msgs_FOUND)
  add_executable(fisher_ig_estimator src/ros/fisher_ig_estimator_node.cpp)
  target_link_libraries(fisher_ig_estimator ${PROJECT_NAME})
  ament_target_dependencies(fisher_ig_estimator rclcpp active_slam_msgs)
  install(TARGETS fisher_ig_estimator DESTINATION lib/${PROJECT_NAME})
endif()

# ---- ament ------------------------------------------------------------------
if(ament_cmake_FOUND)
  ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
  ament_export_dependencies(Eigen3)
  ament_package()
endif()
