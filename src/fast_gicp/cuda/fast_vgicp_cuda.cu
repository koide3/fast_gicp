#include <fast_gicp/cuda/fast_vgicp_cuda.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <fast_gicp/cuda/covariance_estimation.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>

namespace fast_gicp {

FastVGICPCudaCore::FastVGICPCudaCore() {
  resolution = 1.0;

  // warming up GPU
  cudaDeviceSynchronize();
}
FastVGICPCudaCore ::~FastVGICPCudaCore() {}

void FastVGICPCudaCore::set_resolution(double resolution) {
  this->resolution = resolution;
}

void FastVGICPCudaCore::set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  source_points.reset(new Points(points));
}

void FastVGICPCudaCore::set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  target_points.reset(new Points(points));
}

void FastVGICPCudaCore::set_source_neighbors(int k, const std::vector<int>& neighbors) {
  assert(k * source_points->size() == neighbors.size());
  thrust::host_vector<int> k_neighbors(neighbors.begin(), neighbors.end());

  if(!source_neighbors) {
    source_neighbors.reset(new thrust::device_vector<int>());
  }

  *source_neighbors = k_neighbors;
}

void FastVGICPCudaCore::set_target_neighbors(int k, const std::vector<int>& neighbors) {
  assert(k * target_points->size() == neighbors.size());
  thrust::host_vector<int> k_neighbors(neighbors.begin(), neighbors.end());

  if(!target_neighbors) {
    target_neighbors.reset(new thrust::device_vector<int>());
  }

  *target_neighbors = k_neighbors;
}

struct untie_pair_second {
  __device__ int operator() (thrust::pair<float, int>& p) const {
    return p.second;
  }
};

void FastVGICPCudaCore::find_source_neighbors(int k) {
  assert(source_points);

  thrust::device_vector<thrust::pair<float, int>> k_neighbors;
  brute_force_knn_search(*source_points, *source_points, k, k_neighbors);

  if(!source_neighbors) {
    source_neighbors.reset(new thrust::device_vector<int>(k_neighbors.size()));
  }
  thrust::transform(k_neighbors.begin(), k_neighbors.end(), source_neighbors->begin(), untie_pair_second());
}

void FastVGICPCudaCore::find_target_neighbors(int k) {
  assert(target_points);

  thrust::device_vector<thrust::pair<float, int>> k_neighbors;
  brute_force_knn_search(*target_points, *target_points, k, k_neighbors);

  if(!target_neighbors) {
    target_neighbors.reset(new thrust::device_vector<int>(k_neighbors.size()));
  }
  thrust::transform(k_neighbors.begin(), k_neighbors.end(), target_neighbors->begin(), untie_pair_second());
}

void FastVGICPCudaCore::calculate_source_covariances() {
  assert(source_points && source_neighbors);
  int k = source_neighbors->size() / source_points->size();

  if(!source_covariances) {
    source_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(source_points->size()));
  }
  covariance_estimation(*source_points, k, *source_neighbors, *source_covariances);
}

void FastVGICPCudaCore::calculate_target_covariances() {
  assert(source_points && source_neighbors);
  int k = target_neighbors->size() / target_points->size();

  if(!target_covariances) {
    target_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(target_points->size()));
  }
  covariance_estimation(*target_points, k, *target_neighbors, *target_covariances);
}

void FastVGICPCudaCore::create_target_voxelmap() {
  assert(target_points && target_covariances);
  voxelmap.reset(new GaussianVoxelMap(resolution));
  voxelmap->create_voxelmap(*target_points, *target_covariances);
}

void FastVGICPCudaCore::test_print() {
  return;
  thrust::host_vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs = *source_covariances;

  for(int i = 0; i < 10; i++) {
    std::cout << "--- " << i << " ---" << std::endl;
    std::cout << covs[i] << std::endl;
  }
}

}  // namespace fast_gicp
