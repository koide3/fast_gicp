#ifndef FAST_GICP_FAST_VGICP_CUDA_IMPL_HPP
#define FAST_GICP_FAST_VGICP_CUDA_IMPL_HPP

#include <atomic>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <sophus/so3.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#include <fast_gicp/cuda/fast_vgicp_cuda.cuh>

namespace fast_gicp {

template<typename PointSource, typename PointTarget>
FastVGICPCuda<PointSource, PointTarget>::FastVGICPCuda() {
  reg_name_ = "FastVGICPCuda";
  max_iterations_ = 64;
  k_correspondences_ = 20;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;

  voxel_resolution_ = 1.0;
  regularization_method_ = PLANE;

  neighbor_search_method_ = CPU_PARALLEL_KDTREE;
  vgicp_cuda.reset(new FastVGICPCudaCore());
}

template<typename PointSource, typename PointTarget>
FastVGICPCuda<PointSource, PointTarget>::~FastVGICPCuda() {}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setResolution(double resolution) {
  voxel_resolution_ = resolution;
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud->size());
  std::transform(cloud->begin(), cloud->end(), points.begin(), [=](const PointSource& pt) { return pt.getVector3fMap(); });

  vgicp_cuda->set_source_cloud(points);
  switch(neighbor_search_method_) {
    case CPU_PARALLEL_KDTREE: {
      std::vector<int> neighbors = find_neighbors_parallel_kdtree(k_correspondences_, cloud, source_kdtree);
      vgicp_cuda->set_source_neighbors(k_correspondences_, neighbors);
    }
      break;
    case GPU_BRUTEFORCE:
      vgicp_cuda->find_source_neighbors(k_correspondences_);
      break;
  }
  vgicp_cuda->calculate_source_covariances();
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud->size());
  std::transform(cloud->begin(), cloud->end(), points.begin(), [=](const PointTarget& pt) { return pt.getVector3fMap(); });

  vgicp_cuda->set_target_cloud(points);
  switch(neighbor_search_method_) {
    case CPU_PARALLEL_KDTREE: {
      std::vector<int> neighbors = find_neighbors_parallel_kdtree(k_correspondences_, cloud, target_kdtree);
      vgicp_cuda->set_target_neighbors(k_correspondences_, neighbors);
    } break;
    case GPU_BRUTEFORCE:
      vgicp_cuda->find_target_neighbors(k_correspondences_);
      break;
  }
  vgicp_cuda->calculate_target_covariances();
  vgicp_cuda->create_target_voxelmap();
}

template<typename PointSource, typename PointTarget>
template<typename PointT>
std::vector<int> FastVGICPCuda<PointSource, PointTarget>::find_neighbors_parallel_kdtree(int k, const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree) const {
  kdtree.setInputCloud(cloud);
  std::vector<int> neighbors(cloud->size() * k);

#pragma omp parallel for
  for(int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k, k_indices, k_sq_distances);

    std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
  }

  return neighbors;
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Isometry3f initial_guess(guess);
  Eigen::Isometry3f estimated = Eigen::Isometry3f::Identity();

  converged_ = vgicp_cuda->optimize(initial_guess, estimated);

  final_transformation_ = estimated.matrix();
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

}  // namespace fast_gicp

#endif
