#include <fast_gicp/cuda/covariance_regularization.cuh>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace fast_gicp {
namespace cuda {

namespace {

struct covariance_regularization_svd {
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
}  // namespace

void covariance_regularization(thrust::device_vector<Eigen::Matrix3f>& covs, RegularizationMethod method) {
  switch(method) {
    default:
      std::cerr << "unimplemented covariance regularization method was selected!!" << std::endl;
    case RegularizationMethod::PLANE:
      thrust::for_each(covs.begin(), covs.end(), covariance_regularization_svd());
      break;
    case RegularizationMethod::FROBENIUS:
      thrust::for_each(covs.begin(), covs.end(), covariance_regularization_frobenius());
      break;
  }
}

}  // namespace cuda
}  // namespace fast_gicp