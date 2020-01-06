cmake_minimum_required(VERSION 2.8.3)
project(fast_gicp)

add_definitions(-std=c++11 -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
set(CMAKE_CXX_FLAGS "-std=c++11 -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")

find_package(catkin REQUIRED COMPONENTS pcl_ros gl_test_field)
find_package(PCL REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES fast_gicp
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  include
  thirdparty/Sophus
  ${catkin_INCLUDE_DIRS}
)

add_library(fast_gicp
  src/fast_gicp/gicp/gicp_loss.cpp
)
add_dependencies(fast_gicp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(fast_gicp
  ${catkin_LIBRARIES}
)

add_executable(gicp_align src/align.cpp)
add_dependencies(gicp_align ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gicp_align
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  fast_gicp
)

add_executable(fast_test src/fast_test.cpp)
target_link_libraries(fast_test
  ${catkin_LIBRARIES}
  fast_gicp
)
