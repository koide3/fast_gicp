#ifndef FAST_GICP_CUDA_COMPUTE_DERIVATIVES_CUH
#define FAST_GICP_CUDA_COMPUTE_DERIVATIVES_CUH

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <fast_gicp/cuda/gaussian_voxelmap.cuh>

namespace fast_gicp {

namespace  {

struct compute_derivatives_kernel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  compute_derivatives_kernel(const GaussianVoxelMap& voxelmap, const Eigen::Matrix<float, 6, 1>& x)
  : R(Sophus::SO3f::exp(x.head<3>()).matrix()),
    t(x.tail<3>()),
    max_bucket_scan_count(voxelmap.max_bucket_scan_count),
    voxel_resolution(voxelmap.voxel_resolution),
    num_buckets(voxelmap.buckets.size()),
    buckets_ptr(voxelmap.buckets.data()),
    voxel_num_points_ptr(voxelmap.num_points.data()),
    voxel_means_ptr(voxelmap.voxel_means.data()),
    voxel_covs_ptr(voxelmap.voxel_covs.data())
  {}

  // lookup voxel
  __host__ __device__ int lookup_voxel(const Eigen::Vector3f& x) const {
    Eigen::Vector3i coord = calc_voxel_coord(x, voxel_resolution);
    uint64_t hash = vector3i_hash(coord);

    for(int i = 0; i < max_bucket_scan_count; i++) {
      uint64_t bucket_index = (hash + i) % num_buckets;
      const thrust::pair<Eigen::Vector3i, int>& bucket = thrust::raw_pointer_cast(buckets_ptr) [bucket_index];

      if(bucket.second < 0) {
        return -1;
      }

      if(bucket.first == coord) {
          return bucket.second;
      }
    }

    return -1;
  }

  // skew symmetric matrix
  __host__ __device__ Eigen::Matrix3f skew_symmetric(const Eigen::Vector3f& x) const {
    Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
    skew(0, 1) = -x[2];
    skew(0, 2) = x[1];
    skew(1, 0) = x[2];
    skew(1, 2) = -x[0];
    skew(2, 0) = -x[1];
    skew(2, 1) = x[0];

    return skew;
  }

  // calculate derivatives
  template<typename Tuple>
  __host__ __device__ void operator() (Tuple tuple) const {
    const Eigen::Vector3f& mean_A = thrust::get<0>(tuple);
    const Eigen::Matrix3f& cov_A = thrust::get<1>(tuple);
    const Eigen::Vector3f transed_mean_A = R * mean_A + t;

    int voxel_index = lookup_voxel(transed_mean_A);
    if(voxel_index < 0) {
      return;
    }

    int num_points = thrust::raw_pointer_cast(voxel_num_points_ptr)[voxel_index];
    const Eigen::Vector3f& mean_B = thrust::raw_pointer_cast(voxel_means_ptr)[voxel_index];
    const Eigen::Matrix3f& cov_B = thrust::raw_pointer_cast(voxel_covs_ptr)[voxel_index];

    Eigen::Matrix3f RCR = R * cov_A * R.transpose();
    Eigen::Matrix3f skew_mean_A = skew_symmetric(transed_mean_A);

    Eigen::Vector3f d = mean_B - transed_mean_A;
    Eigen::Matrix3f RCR_inv = (cov_B + RCR).inverse();

    Eigen::Vector3f& loss = thrust::get<2>(tuple);
    Eigen::Matrix<float, 3, 6, Eigen::RowMajor>& J = thrust::get<3>(tuple);

    loss = num_points * (RCR_inv * d);
    J.block<3, 3>(0, 0) = RCR_inv * skew_mean_A;
    J.block<3, 3>(0, 3) = -RCR_inv;
  }

  const Eigen::Matrix3f R;
  const Eigen::Vector3f t;

  thrust::device_ptr<const Eigen::Vector3f> source_points_ptr;
  thrust::device_ptr<const Eigen::Matrix3f> source_covs_ptr;

  const int max_bucket_scan_count;
  const float voxel_resolution;

  const int num_buckets;
  thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr;

  thrust::device_ptr<const int> voxel_num_points_ptr;
  thrust::device_ptr<const Eigen::Vector3f> voxel_means_ptr;
  thrust::device_ptr<const Eigen::Matrix3f> voxel_covs_ptr;
};

struct is_nan_kernel {
  template<typename T>
  __host__ __device__ bool operator() (const T& x) const {
    return isnan(x.data()[0]);
  }
};

} // namespace


void compute_derivatives(const thrust::device_vector<Eigen::Vector3f>& src_points, const thrust::device_vector<Eigen::Matrix3f>& src_covs, const GaussianVoxelMap& voxelmap, const Eigen::Matrix<float, 6, 1>& x, thrust::device_vector<Eigen::Vector3f>& losses, thrust::device_vector<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>>& Js) {
  float nan = std::nanf("");
  losses.resize(src_points.size());
  Js.resize(src_points.size());
  thrust::fill(losses.begin(), losses.end(), Eigen::Vector3f::Constant(nan));
  thrust::fill(Js.begin(), Js.end(), Eigen::Matrix<float, 3, 6>::Constant(nan));

  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(src_points.begin(), src_covs.begin(), losses.begin(), Js.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(src_points.end(), src_covs.end(), losses.end(), Js.end())),
    compute_derivatives_kernel(voxelmap, x)
  );

  // erase invalid points
  losses.erase(thrust::remove_if(losses.begin(), losses.end(), is_nan_kernel()), losses.end());
  Js.erase(thrust::remove_if(Js.begin(), Js.end(), is_nan_kernel()), Js.end());
}

} // namespace fast_gicp


#endif