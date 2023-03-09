#include <fast_gicp/cuda/covariance_regularization.cuh>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_output_iterator.h>

namespace fast_gicp {
namespace cuda {

namespace {

struct svd_kernel {
  __host__ __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Matrix3f, int> operator()(const thrust::tuple<Eigen::Vector3f, Eigen::Matrix3f, int>& input) const {
    const auto& mean = thrust::get<0>(input);
    const auto& cov = thrust::get<1>(input);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    eig.computeDirect(cov);

    return thrust::make_tuple(mean, eig.eigenvalues(), eig.eigenvectors(), thrust::get<2>(input));
  }
};

struct eigenvalue_filter_kernel {
  __host__ __device__ bool operator()(const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Matrix3f>& input) const {
    const auto& values = thrust::get<1>(input);
    return values[1] > values[2] * 0.1;
  }
};

struct svd_reconstruction_kernel {
  svd_reconstruction_kernel(
    const thrust::device_ptr<const Eigen::Matrix3f>& values_diag,
    thrust::device_vector<Eigen::Matrix3f>& covariances)
  : values_diag_ptr(values_diag), covariances_ptr(covariances.data()) {}
  __host__ __device__ void operator()(const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Matrix3f, int>& input) const {
    const auto& mean = thrust::get<0>(input);
    const auto& values = thrust::get<1>(input);
    const auto& vecs = thrust::get<2>(input);
    const auto idx = thrust::get<3>(input);

    Eigen::Matrix3f vecs_inv = vecs.inverse();
    const auto& values_diag = *thrust::raw_pointer_cast(values_diag_ptr);
    Eigen::Matrix3f* cov = thrust::raw_pointer_cast(covariances_ptr) + idx;
    *cov = (vecs * values_diag * vecs_inv).eval();
  }
  const thrust::device_ptr<const Eigen::Matrix3f> values_diag_ptr;
  thrust::device_ptr<Eigen::Matrix3f> covariances_ptr;
};



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

struct covariance_regularization_mineig {
  __host__ __device__ void operator()(Eigen::Matrix3f& cov) const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    eig.computeDirect(cov);

    Eigen::Vector3f values = eig.eigenvalues();
    for (int i = 0; i < 3; i++) {
      values[i] = fmaxf(1e-3f, values[i]);
    }

    Eigen::Matrix3f v_diag = Eigen::Matrix3f::Zero();
    v_diag(0,0) = values.x();
    v_diag(1,1) = values.y();
    v_diag(2,2) = values.z();
    Eigen::Matrix3f v_inv = eig.eigenvectors().inverse();
    cov = eig.eigenvectors() * v_diag * v_inv;
  }
};

}  // namespace

void covariance_regularization(thrust::device_vector<Eigen::Vector3f>& means, thrust::device_vector<Eigen::Matrix3f>& covs, RegularizationMethod method) {
  if (method == RegularizationMethod::PLANE) {
    thrust::device_vector<int> d_indices(covs.size());
    thrust::sequence(d_indices.begin(), d_indices.end());
    auto first = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(means.begin(), covs.begin(), d_indices.begin())), svd_kernel());
    auto last = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(means.end(), covs.end(), d_indices.end())), svd_kernel());

    Eigen::Matrix3f diag_matrix = Eigen::Vector3f(1e-3f, 1.0f, 1.0f).asDiagonal();
    thrust::device_vector<Eigen::Matrix3f> val(1);
    val[0] = diag_matrix;
    thrust::device_ptr<Eigen::Matrix3f> diag_matrix_ptr = val.data();
    thrust::for_each(first, last, svd_reconstruction_kernel(diag_matrix_ptr, covs));

  } else if (method == RegularizationMethod::FROBENIUS) {
    thrust::for_each(covs.begin(), covs.end(), covariance_regularization_frobenius());
  } else if (method == RegularizationMethod::MIN_EIG) {
    thrust::for_each(covs.begin(), covs.end(), covariance_regularization_mineig());
  } else {
    std::cerr << "unimplemented covariance regularization method was selected!!" << std::endl;
  }
}

}  // namespace cuda
}  // namespace fast_gicp