cmake_minimum_required(VERSION 3.10.0)
project(fast_gicp LANGUAGES CXX CUDA)

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" ON)
option(BUILD_apps "Build application programs" ON)
option(BUILD_test "Build test programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

set(CMAKE_BUILD_TYPE "Release")

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
add_definitions(${PCL_DEFINITIONS})

if(DEFINED ENV{ROS_VERSION})
  set(ROS_VERSION $ENV{ROS_VERSION})
endif()

if(NOT BUILD_PYTHON_BINDINGS)
  if(${ROS_VERSION})
    if(${ROS_VERSION} EQUAL 1)
      find_package(catkin)
    elseif (${ROS_VERSION} EQUAL 2)
      find_package(ament_cmake)
    endif()
  endif()
endif()

#find_package(OpenMP)
#if (OPENMP_FOUND)
#  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

#if(BUILD_VGICP_CUDA)
#  include(${CMAKE_SOURCE_DIR}/cmake/build/glyd_cuda.cmake)
  #  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  #  link_directories(${CUDAToolkit_LIBRARY_DIR})
  #  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRS})
  #  link_directories(${CMAKE_CUDA_TOOLKIT_LIBRARY_DIR})
#endif()

###################################
## catkin specific configuration ##
###################################
if(catkin_FOUND)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES fast_gicp
  )
endif()

###########
## Build ##
###########

glyd_add_library(fast_gicp SHARED
    SOURCES
        src/fast_gicp/gicp/lsq_registration.cpp
        src/fast_gicp/gicp/fast_gicp.cpp
        src/fast_gicp/gicp/fast_gicp_st.cpp
        src/fast_gicp/gicp/fast_vgicp.cpp
        src/fast_gicp/gicp/fast_vgicp_cuda.cpp
        src/fast_gicp/ndt/ndt_cuda.cpp
    LIBRARIES
        ${PCL_LIBRARIES}
        fast_vgicp_cuda
        ${OpenMP_CXX_FLAGS}
    COMPILE_OPTIONS
        ${OpenMP_CXX_FLAGS}
)
        
#add_library(fast_gicp SHARED
#  src/fast_gicp/gicp/lsq_registration.cpp
#  src/fast_gicp/gicp/fast_gicp.cpp
#  src/fast_gicp/gicp/fast_gicp_st.cpp
#  src/fast_gicp/gicp/fast_vgicp.cpp
#)
#target_link_libraries(fast_gicp
#  ${PCL_LIBRARIES}
#)

target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

### APPS ###
if(BUILD_apps)
    glyd_add_executable(
        gicp_align
        SOURCES
            src/align.cpp
        LIBRARIES
            fast_gicp
	    ${PCL_LIBRARIES}
            ${OpenMP_CXX_FLAGS}
	COMPILE_OPTIONS
            ${OpenMP_CXX_FLAGS}
    )
  #add_executable(gicp_align src/align.cpp)
  #add_dependencies(gicp_align fast_gicp)
  #target_link_libraries(gicp_align
  #  ${PCL_LIBRARIES}
  #  fast_gicp
  #)
    glyd_add_executable(
        gicp_kitti
        SOURCES
            src/kitti.cpp
        LIBRARIES
            fast_gicp
	    ${PCL_LIBRARIES}
    )

#  add_executable(gicp_kitti src/kitti.cpp)
#  add_dependencies(gicp_kitti fast_gicp)
#  target_link_libraries(gicp_kitti
#    ${PCL_LIBRARIES}
#    fast_gicp
#  )
endif()

### Python bindings ###
#if(BUILD_PYTHON_BINDINGS)
#  add_subdirectory(thirdparty/pybind11)
#  pybind11_add_module(pygicp
#    src/python/main.cpp
#  )
#  target_include_directories(pygicp PUBLIC
#    include
#    ${PCL_INCLUDE_DIRS}
#    ${EIGEN3_INCLUDE_DIR}
#  )
#  target_link_libraries(pygicp PRIVATE
#    fast_gicp
#  )
#endif()

### CUDA ###
if(BUILD_VGICP_CUDA)
    set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    add_definitions(-DUSE_VGICP_CUDA)
    glyd_add_library(fast_vgicp_cuda SHARED
        SOURCES
            src/fast_gicp/cuda/fast_vgicp_cuda.cu
            src/fast_gicp/cuda/brute_force_knn.cu
            src/fast_gicp/cuda/covariance_estimation.cu
            src/fast_gicp/cuda/covariance_estimation_rbf.cu
            src/fast_gicp/cuda/covariance_regularization.cu
            src/fast_gicp/cuda/gaussian_voxelmap.cu
            src/fast_gicp/cuda/find_voxel_correspondences.cu
            src/fast_gicp/cuda/compute_derivatives.cu
            src/fast_gicp/cuda/compute_mahalanobis.cu
            src/fast_gicp/cuda/ndt_cuda.cu
            src/fast_gicp/cuda/ndt_compute_derivatives.cu
        LIBRARIES
            ${PCL_LIBRARIES}
            ${OpenMP_CXX_FLAGS}
            ${catkin_LIBRARIES}
            CUDA::cublas
            CUDA::curand
    ) 
    set_property(
        TARGET
	    fast_vgicp_cuda
	PROPERTY
	    CUDA_STANDARD
	    14
    )
    set_target_properties(fast_vgicp_cuda PROPERTIES CUDA_ARCHITECTURES "72;86")


#  add_library(fast_vgicp_cuda SHARED
#    src/fast_gicp/cuda/fast_vgicp_cuda.cu
#    src/fast_gicp/cuda/brute_force_knn.cu
#    src/fast_gicp/cuda/covariance_estimation.cu
#    src/fast_gicp/cuda/covariance_estimation_rbf.cu
#    src/fast_gicp/cuda/covariance_regularization.cu
#    src/fast_gicp/cuda/gaussian_voxelmap.cu
#    src/fast_gicp/cuda/find_voxel_correspondences.cu
#    src/fast_gicp/cuda/compute_derivatives.cu
#    src/fast_gicp/cuda/compute_mahalanobis.cu
#    src/fast_gicp/cuda/ndt_cuda.cu
#    src/fast_gicp/cuda/ndt_compute_derivatives.cu
#  )
#  set_target_properties(fast_vgicp_cuda PROPERTIES LINKER_LANGUAGE CUDACXX)
  target_include_directories(fast_vgicp_cuda PRIVATE
    include
    thirdparty/Eigen
    thirdparty/nvbio
    ${OpenMP_CXX_FLAGS}
    ${catkin_INCLUDE_DIRS}
  )

#  target_link_libraries(fast_vgicp_cuda
#    ${OpenMP_CXX_FLAGS}
#    ${catkin_LIBRARIES}
#    #CUDA::cublas_static
#    CUDA::cublas
#    #CUDA::curand_static
#    CUDA::curand
#  )

  # add vgicp_cuda to libfast_gicp
  #  target_sources(fast_gicp PRIVATE
  #  src/fast_gicp/gicp/fast_vgicp_cuda.cpp
  #  src/fast_gicp/ndt/ndt_cuda.cpp
  #)
#  target_link_libraries(fast_gicp
#    fast_vgicp_cuda
#    ${OpenMP_CXX_FLAGS}
#  )
#  add_dependencies(fast_gicp fast_vgicp_cuda)
#  if(catkin_FOUND)
#    install(TARGETS fast_vgicp_cuda
#      LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
#  elseif (ament_cmake_FOUND)
#    install(TARGETS fast_vgicp_cuda
#      LIBRARY DESTINATION lib)
#  endif()
endif()

### TEST ###
#if(BUILD_test)
#  find_package(GTest REQUIRED)
#
#  add_executable(gicp_test src/test/gicp_test.cpp)
#  add_dependencies(gicp_test fast_gicp)
#  target_link_libraries(gicp_test ${GTEST_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)
#  gtest_add_tests(TARGET gicp_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} EXTRA_ARGS "${CMAKE_SOURCE_DIR}/data")
#endif()
#
#if(catkin_FOUND)
#  ###################################
#  ## catkin specific configuration ##
#  ###################################
#  install(TARGETS ${PROJECT_NAME}
#    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
#
#  install(DIRECTORY include/
#    DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
#    FILES_MATCHING PATTERN "*.hpp")
#elseif (ament_cmake_FOUND)
#  ##################################
#  ## ament specific configuration ##
#  ##################################
#  ament_export_include_directories(include)
#  ament_export_libraries(fast_gicp)
#  ament_package()
#
#  install(TARGETS ${PROJECT_NAME}
#    LIBRARY DESTINATION lib)
#
#  install(
#      DIRECTORY "include/"
#      DESTINATION include
#    )
#endif()
