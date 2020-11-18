#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <fast_gicp/cuda/gaussian_voxelmap.cuh>

namespace fast_gicp {

namespace  {

template<typename T>
struct compute_derivatives_kernel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  compute_derivatives_kernel(const GaussianVoxelMap& voxelmap, const Eigen::Isometry3f& x_eval, const Eigen::Isometry3f& x)
  : R_eval(x_eval.linear()),
    R(x.linear()),
    t(x.translation()),
    voxel_num_points_ptr(voxelmap.num_points.data()),
    voxel_means_ptr(voxelmap.voxel_means.data()),
    voxel_covs_ptr(voxelmap.voxel_covs.data())
  {}

  // skew symmetric matrix
  __host__ __device__
  Eigen::Matrix3f skew_symmetric(const Eigen::Vector3f& x) const {
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
  __host__ __device__
  T operator() (const thrust::tuple<Eigen::Vector3f, Eigen::Matrix3f, int>& input_tuple) const {
    const Eigen::Vector3f& mean_A = thrust::get<0>(input_tuple);
    const Eigen::Matrix3f& cov_A = thrust::get<1>(input_tuple);
    const int voxel_index = thrust::get<2>(input_tuple);

    if(voxel_index < 0) {
      return thrust::make_tuple(0.0f, Eigen::Matrix<float, 6, 6>::Zero().eval(), Eigen::Matrix<float, 6, 1>::Zero().eval());
    }

    int num_points = thrust::raw_pointer_cast(voxel_num_points_ptr)[voxel_index];
    const Eigen::Vector3f& mean_B = thrust::raw_pointer_cast(voxel_means_ptr)[voxel_index];
    const Eigen::Matrix3f& cov_B = thrust::raw_pointer_cast(voxel_covs_ptr)[voxel_index];

    const Eigen::Vector3f transed_mean_A = R * mean_A + t;

    Eigen::Matrix3f RCR = R_eval * cov_A * R_eval.transpose();
    Eigen::Matrix3f RCR_inv = (cov_B + RCR).inverse();

    Eigen::Vector3f error = std::sqrt(num_points) * RCR_inv * (mean_B - transed_mean_A);

    Eigen::Matrix<float, 3, 6> dtdx0;
    dtdx0.block<3, 3>(0, 0) = skew_symmetric(transed_mean_A);
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3f::Identity();

    Eigen::Matrix<float, 3, 6> J = std::sqrt(num_points) * RCR_inv * dtdx0;

    Eigen::Matrix<float, 6, 6> H = J.transpose() * J;
    Eigen::Matrix<float, 6, 1> b = J.transpose() * error;

    return thrust::make_tuple(error.squaredNorm(), H, b);
  }

  const Eigen::Matrix3f R_eval;

  const Eigen::Matrix3f R;
  const Eigen::Vector3f t;

  thrust::device_ptr<const int> voxel_num_points_ptr;
  thrust::device_ptr<const Eigen::Vector3f> voxel_means_ptr;
  thrust::device_ptr<const Eigen::Matrix3f> voxel_covs_ptr;
};

template<>
__host__ __device__
float compute_derivatives_kernel<float>::operator() (const thrust::tuple<Eigen::Vector3f, Eigen::Matrix3f, int>& input_tuple) const {
  const Eigen::Vector3f& mean_A = thrust::get<0>(input_tuple);
  const Eigen::Matrix3f& cov_A = thrust::get<1>(input_tuple);
  const int voxel_index = thrust::get<2>(input_tuple);

  if(voxel_index < 0) {
    return 0.0f;
  }

  int num_points = thrust::raw_pointer_cast(voxel_num_points_ptr)[voxel_index];
  const Eigen::Vector3f& mean_B = thrust::raw_pointer_cast(voxel_means_ptr)[voxel_index];
  const Eigen::Matrix3f& cov_B = thrust::raw_pointer_cast(voxel_covs_ptr)[voxel_index];

  const Eigen::Vector3f transed_mean_A = R * mean_A + t;

  Eigen::Matrix3f RCR = R_eval * cov_A * R_eval.transpose();
  Eigen::Matrix3f RCR_inv = (cov_B + RCR).inverse();

  Eigen::Vector3f error = std::sqrt(num_points) * RCR_inv * (mean_B - transed_mean_A);

  return error.squaredNorm();
}

struct sum_errors_kernel {
  template<typename Tuple>
  __host__ __device__ Tuple operator() (const Tuple& lhs, const Tuple& rhs) {
    return thrust::make_tuple(
      thrust::get<0>(lhs) + thrust::get<0>(rhs),
      thrust::get<1>(lhs) + thrust::get<1>(rhs),
      thrust::get<2>(lhs) + thrust::get<2>(rhs)
    );
  }
};

template<>
__host__ __device__ float sum_errors_kernel::operator() (const float& lhs, const float& rhs) {
  return lhs + rhs;
}

} // namespace

double compute_derivatives(
  const thrust::device_vector<Eigen::Vector3f>& src_points,
  const thrust::device_vector<Eigen::Matrix3f>& src_covs,
  const GaussianVoxelMap& voxelmap,
  const thrust::device_vector<int>& voxel_correspondences,
  const Eigen::Isometry3f& linearized_x,
  const Eigen::Isometry3f& x,
  Eigen::Matrix<double, 6, 6>* H,
  Eigen::Matrix<double, 6, 1>* b
) {
  if(H == nullptr || b == nullptr) {
    float sum_errors = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(src_points.begin(), src_covs.begin(), voxel_correspondences.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(src_points.end(), src_covs.end(), voxel_correspondences.end())),
      compute_derivatives_kernel<float>(voxelmap, linearized_x, x),
      0.0f,
      sum_errors_kernel()
    );

    return sum_errors;
  }

  auto sum_errors = thrust::transform_reduce(
    thrust::make_zip_iterator(thrust::make_tuple(src_points.begin(), src_covs.begin(), voxel_correspondences.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(src_points.end(), src_covs.end(), voxel_correspondences.end())),
    compute_derivatives_kernel<thrust::tuple<float, Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>>(voxelmap, linearized_x, x),
    thrust::make_tuple(0.0f, Eigen::Matrix<float, 6, 6>::Zero().eval(), Eigen::Matrix<float, 6, 1>::Zero().eval()),
    sum_errors_kernel()
  );

  if(H && b) {
    *H = thrust::get<1>(sum_errors).cast<double>();
    *b = thrust::get<2>(sum_errors).cast<double>();
  }

  return thrust::get<0>(sum_errors);
}

}  // namespace fast_gicp
