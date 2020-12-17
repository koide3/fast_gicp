#ifndef FAST_GICP_CUDA_COMPUTE_DERIVATIVES_CUH
#define FAST_GICP_CUDA_COMPUTE_DERIVATIVES_CUH

#include <Eigen/Core>
#include <thrust/device_vector.h>

#include <fast_gicp/cuda/gaussian_voxelmap.cuh>

namespace fast_gicp {
  namespace cuda {

double compute_derivatives(
  const thrust::device_vector<Eigen::Vector3f>& src_points,
  const thrust::device_vector<Eigen::Matrix3f>& src_covs,
  const GaussianVoxelMap& voxelmap,
  const thrust::device_vector<int>& voxel_correspondences,
  const Eigen::Isometry3f& linearized_x,
  const Eigen::Isometry3f& x,
  Eigen::Matrix<double, 6, 6>* H,
  Eigen::Matrix<double, 6, 1>* b
);

  }
}  // namespace fast_gicp

#endif