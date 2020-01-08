#ifndef FAST_GICP_FAST_VGICP_VOXEL_HPP
#define FAST_GICP_FAST_VGICP_VOXEL_HPP

namespace fast_gicp {

class Vector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const {
    size_t seed = 0;
    boost::hash_combine(seed, x[0]);
    boost::hash_combine(seed, x[1]);
    boost::hash_combine(seed, x[2]);
    return seed;
  }
};

struct GaussianVoxel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<GaussianVoxel>;

  GaussianVoxel() {
    mean.setZero();
    cov.setZero();
  }

  void append(const Eigen::Vector4f& mean_, const Eigen::Matrix4f& cov_) {
    Eigen::Matrix4f cov_inv = cov_;
    cov_inv(3, 3) = 1;
    cov_inv = cov_inv.inverse().eval();

    cov += cov_inv;
    mean += cov_inv * mean_;
  }

  void finalize() {
    cov(3, 3) = 1;
    mean[3] = 1;

    cov = cov.inverse().eval();
    mean = (cov * mean).eval();
  }

public:
  Eigen::Vector4f mean;
  Eigen::Matrix4f cov;
};

}  // namespace fast_gicp

#endif