cmake_minimum_required(VERSION 3.10.0)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

project(fast_gicp LANGUAGES CXX CUDA)

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" ON)
option(BUILD_apps "Build application programs" ON)
option(BUILD_test "Build test programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

set(CMAKE_BUILD_TYPE "Release")

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
add_definitions(${PCL_DEFINITIONS})



###########
## Build ##
###########

glyd_add_library(fast_gicp SHARED
    SOURCES
        src/fast_gicp/gicp/lsq_registration.cpp
        #src/fast_gicp/gicp/fast_gicp.cpp
        #src/fast_gicp/gicp/fast_gicp_st.cpp
        #src/fast_gicp/gicp/fast_vgicp.cpp
        src/fast_gicp/gicp/fast_vgicp_cuda.cpp
        src/fast_gicp/ndt/ndt_cuda.cpp
    LIBRARIES
        pcl::pcl_common
        pcl::pcl_features
        pcl::pcl_filters
        pcl::pcl_io
        pcl::pcl_kdtree
        pcl::pcl_octree
        pcl::pcl_registration
        pcl::pcl_sample_consensus
        pcl::pcl_search
        fast_vgicp_cuda
    #COMPILE_OPTIONS
    #    ${OpenMP_CXX_FLAGS}
)

target_include_directories(fast_gicp PUBLIC
  include
)

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
            pcl::pcl_common
            pcl::pcl_features
            pcl::pcl_filters
            pcl::pcl_io
            pcl::pcl_kdtree
            pcl::pcl_octree
            pcl::pcl_registration
            pcl::pcl_sample_consensus
            pcl::pcl_search
            Eigen3::Eigen
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
    set_target_properties(fast_vgicp_cuda PROPERTIES CUDA_ARCHITECTURES "61;72;86;87")

    target_include_directories(fast_vgicp_cuda PRIVATE
        include
        thirdparty/nvbio
        # ${OpenMP_CXX_FLAGS}
    )
endif()

### APPS ###
if(BUILD_apps)
    glyd_add_executable(
        gicp_align
        SOURCES
            src/align.cpp
        LIBRARIES
            fast_gicp
            pcl::pcl_common
            pcl::pcl_features
            pcl::pcl_filters
            pcl::pcl_io
            pcl::pcl_kdtree
            pcl::pcl_octree
            pcl::pcl_registration
            pcl::pcl_sample_consensus
	      # COMPILE_OPTIONS
            # ${OpenMP_CXX_FLAGS}
    )
endif()
