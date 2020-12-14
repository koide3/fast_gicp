#ifndef FAST_GICP_CUDA_COVARIANCE_ESTIMATION_CUH
#define FAST_GICP_CUDA_COVARIANCE_ESTIMATION_CUH

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

void covariance_estimation(const thrust::device_vector<Eigen::Vector3f>& points, int k, const thrust::device_vector<int>& k_neighbors, thrust::device_vector<Eigen::Matrix3f>& covariances, RegularizationMethod method);

}

#endif