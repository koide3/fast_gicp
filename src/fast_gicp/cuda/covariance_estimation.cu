#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <fast_gicp/gicp/gicp_settings.hpp>
#include <fast_gicp/cuda/covariance_estimation.cuh>

namespace fast_gicp {

namespace {
  struct covariance_estimation_kernel {
    covariance_estimation_kernel(const thrust::device_vector<Eigen::Vector3f>& points, int k, const thrust::device_vector<int>& k_neighbors, thrust::device_vector<Eigen::Matrix3f>& covariances)
        : k(k), points_ptr(points.data()), k_neighbors_ptr(k_neighbors.data()), covariances_ptr(covariances.data()) {}

    __host__ __device__ void operator()(int idx) const {
      // target points buffer & nn output buffer
      const Eigen::Vector3f* points = thrust::raw_pointer_cast(points_ptr);
      const int* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;
      Eigen::Matrix3f* cov = thrust::raw_pointer_cast(covariances_ptr) + idx;

      Eigen::Vector3f mean(0.0f, 0.0f, 0.0f);
      cov->setZero();
      for(int i = 0; i < k; i++) {
        const auto& pt = points[k_neighbors[i]];
        mean += pt;
        (*cov) += pt * pt.transpose();
      }
      mean /= k;
      (*cov) = (*cov) / k - mean * mean.transpose();
    }

    const int k;
    thrust::device_ptr<const Eigen::Vector3f> points_ptr;
    thrust::device_ptr<const int> k_neighbors_ptr;

    thrust::device_ptr<Eigen::Matrix3f> covariances_ptr;
  };

  struct covariance_regularization_svd{
    __host__ __device__ void operator()(Eigen::Matrix3f& cov) const {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
      eig.computeDirect(cov);

      // why this doen't work...???
      // cov = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().inverse();
      Eigen::Matrix3f values = Eigen::Vector3f(1e-3, 1, 1).asDiagonal();
      Eigen::Matrix3f v_inv = eig.eigenvectors().inverse();
      cov = eig.eigenvectors() * values * v_inv;

      // JacobiSVD is not supported on CUDA
      // Eigen::JacobiSVD(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
      // Eigen::Vector3f values(1, 1, 1e-3);
      // cov = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  };

  struct covariance_regularization_frobenius {
    __host__ __device__ void operator()(Eigen::Matrix3f& cov) const {
      float lambda = 1e-3;
      Eigen::Matrix3f C = cov + lambda * Eigen::Matrix3f::Identity();
      Eigen::Matrix3f C_inv = C.inverse();
      Eigen::Matrix3f C_norm = (C_inv / C_inv.norm()).inverse();
      cov = C_norm;
    }
  };
}

void covariance_estimation(const thrust::device_vector<Eigen::Vector3f>& points, int k, const thrust::device_vector<int>& k_neighbors, thrust::device_vector<Eigen::Matrix3f>& covariances, RegularizationMethod method) {
  thrust::device_vector<int> d_indices(points.size());
  thrust::sequence(d_indices.begin(), d_indices.end());

  covariances.resize(points.size());
  thrust::for_each(d_indices.begin(), d_indices.end(), covariance_estimation_kernel(points, k, k_neighbors, covariances));

  switch(method) {
    default:
      std::cerr << "unimplemented covariance regularization method was selected!!" << std::endl;
      abort();
    case RegularizationMethod::PLANE:
      thrust::for_each(covariances.begin(), covariances.end(), covariance_regularization_svd());
      break;
    case RegularizationMethod::FROBENIUS:
      thrust::for_each(covariances.begin(), covariances.end(), covariance_regularization_frobenius());
      break;
  }
}
}
