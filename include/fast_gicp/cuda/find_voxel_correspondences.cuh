#ifndef FAST_GICP_CUDA_FIND_VOXEL_CORRESPONDENCES_CUH
#define FAST_GICP_CUDA_FIND_VOXEL_CORRESPONDENCES_CUH

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/device_vector.h>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>

namespace fast_gicp {
  namespace cuda {

void find_voxel_correspondences(const thrust::device_vector<Eigen::Vector3f>& src_points, const GaussianVoxelMap& voxelmap, const Eigen::Isometry3f& x, thrust::device_vector<int>& correspondences);

  }
}

#endif