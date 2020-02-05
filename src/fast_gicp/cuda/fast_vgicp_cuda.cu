#include <fast_gicp/cuda/fast_vgicp_cuda.cuh>

#include <sophus/so3.hpp>

#include <thrust/device_new.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <fast_gicp/cuda/covariance_estimation.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/compute_derivatives.cuh>

namespace fast_gicp {

FastVGICPCudaCore::FastVGICPCudaCore() {
  // warming up GPU
  cudaDeviceSynchronize();
  cublasCreate(&cublas_handle);

  resolution = 1.0;
  max_iterations = 64;
  rotation_epsilon = 2e-3;
  transformation_epsilon = 5e-4;
}
FastVGICPCudaCore ::~FastVGICPCudaCore() {
  cublasDestroy(cublas_handle);
}

void FastVGICPCudaCore::set_resolution(double resolution) {
  this->resolution = resolution;
}

void FastVGICPCudaCore::set_max_iterations(int itr) {
  this->max_iterations = itr;
}

void FastVGICPCudaCore::set_rotation_epsilon(double eps) {
  this->rotation_epsilon = eps;
}

void FastVGICPCudaCore::set_transformation_epsilon(double eps) {
  this->transformation_epsilon = eps;
}

void FastVGICPCudaCore::swap_source_and_target() {
  if(source_points && target_points) {
    source_points.swap(target_points);
  }
  if(source_neighbors && target_neighbors) {
    source_neighbors.swap(target_neighbors);
  }
  if(source_covariances && target_covariances) {
    source_covariances.swap(target_covariances);
  }

  if(!target_points || !target_covariances) {
    return;
  }

  if(!voxelmap) {
    voxelmap.reset(new GaussianVoxelMap(resolution));
  }
  voxelmap->create_voxelmap(*target_points, *target_covariances);
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

void FastVGICPCudaCore::calculate_source_covariances(RegularizationMethod method) {
  assert(source_points && source_neighbors);
  int k = source_neighbors->size() / source_points->size();

  if(!source_covariances) {
    source_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(source_points->size()));
  }
  covariance_estimation(*source_points, k, *source_neighbors, *source_covariances, method);
}

void FastVGICPCudaCore::calculate_target_covariances(RegularizationMethod method) {
  assert(source_points && source_neighbors);
  int k = target_neighbors->size() / target_points->size();

  if(!target_covariances) {
    target_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(target_points->size()));
  }
  covariance_estimation(*target_points, k, *target_neighbors, *target_covariances, method);
}

void FastVGICPCudaCore::create_target_voxelmap() {
  assert(target_points && target_covariances);
  if(!voxelmap) {
    voxelmap.reset(new GaussianVoxelMap(resolution));
  }
  voxelmap->create_voxelmap(*target_points, *target_covariances);

  // cudaDeviceSynchronize();
}

bool FastVGICPCudaCore::is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
  Eigen::Matrix3f R = Sophus::SO3f::exp(delta.head<3>()).matrix() - Eigen::Matrix3f::Identity();
  Eigen::Vector3f t = delta.tail<3>();

  Eigen::Matrix3f r_delta = 1.0 / rotation_epsilon * R.array().abs();
  Eigen::Vector3f t_delta = 1.0 / transformation_epsilon * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

bool FastVGICPCudaCore::optimize(Eigen::Isometry3f& estimated) {
  Eigen::Isometry3f initial_guess = Eigen::Isometry3f::Identity();
  return optimize(initial_guess, estimated);
}

bool FastVGICPCudaCore::optimize(const Eigen::Isometry3f& initial_guess, Eigen::Isometry3f& estimated) {
  assert(source_points && source_covariances && voxelmap);

  Eigen::Matrix<float, 6, 1> x0;
  x0.head<3>() = Sophus::SO3f(initial_guess.linear()).log();
  x0.tail<3>() = initial_guess.translation();

  if(x0.head<3>().norm() < 1e-2) {
    x0.head<3>() = (Eigen::Vector3f::Random()).normalized() * 1e-2;
  }

  thrust::device_vector<Eigen::Vector3f> losses;                            // 3N error vector
  thrust::device_vector<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>> Js;    // RowMajor 3Nx6 -> ColMajor 6x3N

  thrust::device_ptr<float> JJ_ptr = thrust::device_new<float>(6 * 6);
  thrust::device_ptr<float> J_loss_ptr = thrust::device_new<float>(6);

  bool converged = false;
  for(int i = 0; i < max_iterations; i++) {
    compute_derivatives(*source_points, *source_covariances, *voxelmap, x0, losses, Js);

    // gauss newton
    float alpha = 1.0f;
    float beta = 0.0f;

    int cols = 3 * losses.size();

    float* Js_ptr = thrust::reinterpret_pointer_cast<float*>(Js.data());
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 6, 6, cols, &alpha, Js_ptr, 6, Js_ptr, 6, &beta, thrust::raw_pointer_cast(JJ_ptr), 6);

    float* loss_ptr = thrust::reinterpret_pointer_cast<float*>(losses.data());
    cublasSgemv(cublas_handle, CUBLAS_OP_N, 6, cols, &alpha, Js_ptr, 6, loss_ptr, 1, &beta, thrust::raw_pointer_cast(J_loss_ptr), 1);

    Eigen::Matrix<float, 6, 6> JJ;
    cublasGetMatrix(6, 6, sizeof(float), thrust::raw_pointer_cast(JJ_ptr), 6, JJ.data(), 6);

    Eigen::Matrix<float, 6, 1> J_loss;
    cublasGetVector(6, sizeof(float), thrust::raw_pointer_cast(J_loss_ptr), 1, J_loss.data(), 1);

    Eigen::Matrix<float, 6, 1> delta = JJ.llt().solve(J_loss);

    // update parameters
    x0.head<3>() = (Sophus::SO3f::exp(-delta.head<3>()) * Sophus::SO3f::exp(x0.head<3>())).log();
    x0.tail<3>() -= delta.tail<3>();

    if(is_converged(delta)) {
      converged = true;
      break;
    }
  }

  estimated.setIdentity();
  estimated.linear() = Sophus::SO3f::exp(x0.head<3>()).matrix();
  estimated.translation() = x0.tail<3>();

  return converged;
}

}  // namespace fast_gicp
