#include <fast_gicp/cuda/ndt_cuda.cuh>

#include <thrust/device_vector.h>

#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/covariance_regularization.cuh>
#include <fast_gicp/cuda/find_voxel_correspondences.cuh>
#include <fast_gicp/cuda/ndt_compute_derivatives.cuh>

#include <glk/cuda_magic_headers.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/pointcloud_buffer_cuda.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace fast_gicp {
namespace cuda {

NDTCudaCore::NDTCudaCore() {
  cudaDeviceSynchronize();
  resolution = 1.0;
  linearized_x.setIdentity();

  offsets.reset(new thrust::device_vector<Eigen::Vector3i>(1));
  (*offsets)[0] = Eigen::Vector3i::Zero().eval();

  set_neighbor_search_method(fast_gicp::NeighborSearchMethod::DIRECT7, 0.0);
}

NDTCudaCore::~NDTCudaCore() {}

void NDTCudaCore::set_resolution(double resolution) {
  this->resolution = resolution;
}

void NDTCudaCore::set_neighbor_search_method(fast_gicp::NeighborSearchMethod method, double radius) {
  thrust::host_vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> h_offsets;

  switch (method) {
    default:
      std::cerr << "here must not be reached" << std::endl;
      abort();

    case fast_gicp::NeighborSearchMethod::DIRECT1:
      h_offsets.resize(1);
      h_offsets[0] = Eigen::Vector3i::Zero();
      break;

    case fast_gicp::NeighborSearchMethod::DIRECT7:
      h_offsets.resize(7);
      h_offsets[0] = Eigen::Vector3i(0, 0, 0);
      h_offsets[1] = Eigen::Vector3i(1, 0, 0);
      h_offsets[2] = Eigen::Vector3i(-1, 0, 0);
      h_offsets[3] = Eigen::Vector3i(0, 1, 0);
      h_offsets[4] = Eigen::Vector3i(0, -1, 0);
      h_offsets[5] = Eigen::Vector3i(0, 0, 1);
      h_offsets[6] = Eigen::Vector3i(0, 0, -1);
      break;

    case fast_gicp::NeighborSearchMethod::DIRECT27:
      h_offsets.reserve(27);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            h_offsets.push_back(Eigen::Vector3i(i - 1, j - 1, k - 1));
          }
        }
      }
      break;

    case fast_gicp::NeighborSearchMethod::DIRECT_RADIUS:
      h_offsets.reserve(50);
      int range = std::ceil(radius);
      for (int i = -range; i <= range; i++) {
        for (int j = -range; j <= range; j++) {
          for (int k = -range; k <= range; k++) {
            Eigen::Vector3i offset(i, j, k);
            if (offset.cast<double>().norm() <= radius + 1e-3) {
              h_offsets.push_back(offset);
            }
          }
        }
      }

      break;
  }

  *offsets = h_offsets;
}

void NDTCudaCore::swap_source_and_target() {
  source_points.swap(target_points);
  source_voxelmap.swap(target_voxelmap);
}

void NDTCudaCore::set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  if (!source_points) {
    source_points.reset(new Points());
  }

  *source_points = points;
  source_voxelmap.reset();
}

void NDTCudaCore::set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  if (!target_points) {
    target_points.reset(new Points());
  }

  *target_points = points;
  target_voxelmap.reset();
}

void NDTCudaCore::create_voxelmaps() {
  create_source_voxelmap();
  create_target_voxelmap();
}

void NDTCudaCore::create_source_voxelmap() {
  assert(source_points);
  if (source_voxelmap) {
    return;
  }

  source_voxelmap.reset(new GaussianVoxelMap(resolution));
  source_voxelmap->create_voxelmap(*source_points);
  covariance_regularization(source_voxelmap->voxel_means, source_voxelmap->voxel_covs, fast_gicp::RegularizationMethod::MIN_EIG);
}

void NDTCudaCore::create_target_voxelmap() {
  assert(target_points);
  if (target_voxelmap) {
    return;
  }

  target_voxelmap.reset(new GaussianVoxelMap(resolution));
  target_voxelmap->create_voxelmap(*target_points);
  covariance_regularization(target_voxelmap->voxel_means, target_voxelmap->voxel_covs, fast_gicp::RegularizationMethod::MIN_EIG);
}

void NDTCudaCore::update_correspondences(const Eigen::Isometry3d& trans) {
  thrust::device_vector<Eigen::Isometry3f> trans_ptr(1);
  trans_ptr[0] = trans.cast<float>();

  if (correspondences == nullptr) {
    correspondences.reset(new Correspondences());
  }
  linearized_x = trans.cast<float>();
  find_voxel_correspondences(source_voxelmap->voxel_means, *target_voxelmap, trans_ptr.data(), *offsets, *correspondences);
}

double NDTCudaCore::compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) const {
  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("target", glk::create_point_cloud_buffer(*target_points), guik::FlatColor(Eigen::Vector4f::UnitX()));
  viewer->update_drawable("source", glk::create_point_cloud_buffer(*source_points), guik::FlatColor(Eigen::Vector4f::UnitY(), trans.cast<float>()));
  viewer->spin_until_click();

  thrust::host_vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> trans_(2);
  trans_[0] = linearized_x;
  trans_[1] = trans.cast<float>();

  thrust::device_vector<Eigen::Isometry3f> trans_ptr = trans_;
  return ndt_compute_derivatives(*source_voxelmap, *target_voxelmap, *correspondences, trans_ptr.data(), trans_ptr.data() + 1, H, b);
}

}  // namespace cuda

}  // namespace fast_gicp