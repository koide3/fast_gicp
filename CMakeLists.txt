cmake_minimum_required(VERSION 2.8.3)
project(fast_gicp)

add_definitions(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
set(CMAKE_C_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")

set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS gl_test_field)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include thirdparty/Sophus
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
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

add_library(fast_gicp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
  src/fast_gicp/gicp/experimental/fast_gicp_mp.cpp
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
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
  ${PCL_LIBRARIES}
  fast_gicp
)
