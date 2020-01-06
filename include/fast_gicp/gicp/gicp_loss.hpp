#ifndef FAST_GICP_GICP_LOSS_HPP
#define FAST_GICP_GICP_LOSS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fast_gicp {

/**
 * f(x) = mean_B - (R * mean_A + t)
 * x = [r00, r10, ..., r22, t0, t1, t2]
 */
Eigen::Matrix<float, 4, 12> dtransform(const Eigen::Vector4f& mean_A, const Eigen::Vector4f& mean_B) {
  Eigen::Matrix<float, 4, 12> J = Eigen::Matrix<float, 4, 12>::Zero();
  J.block<4, 4>(0, 0).diagonal().array() = -mean_A[0];
  J.block<4, 4>(0, 3).diagonal().array() = -mean_A[1];
  J.block<4, 4>(0, 6).diagonal().array() = -mean_A[2];
  J.block<4, 4>(0, 9) = -Eigen::Matrix4f::Identity();
  J.block<1, 12>(3, 0).setZero();
  return J;
}

double gicp_loss(const Eigen::Vector4f& mean_A, const Eigen::Matrix4f& cov_A, const Eigen::Vector4f& mean_B, const Eigen::Matrix4f& cov_B, const Eigen::Matrix4f& Rt, Eigen::Matrix<float, 1, 12>* J = nullptr) {
  Eigen::Vector4f d = mean_B - Rt * mean_A;

  Eigen::Matrix4f RCR = cov_B + Rt * cov_A * Rt.transpose();
  RCR(3, 3) = 1;
  Eigen::Matrix4f RCR_inv = RCR.inverse();

  Eigen::Vector4f RCRd = RCR_inv * d;
  double loss = d.dot(RCRd);

  if(!J) {
    return loss;
  }

  Eigen::Matrix<float, 4, 12> jd = dtransform(mean_A, mean_B);
  Eigen::Matrix<float, 4, 12> jRCRd = RCR_inv * jd;

  *J = RCRd.transpose() * jd + d.transpose() * jRCRd;

  return loss;
}

}  // namespace fast_gicp

#endif