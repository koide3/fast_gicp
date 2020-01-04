#ifndef FAST_GICP_GICP_LOSS_HPP
#define FAST_GICP_GICP_LOSS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fast_gicp/gicp/gicp_derivatives.hpp>

namespace gicp {

/*
double gicp_loss(const Eigen::Vector3d& mean_A, const Eigen::Matrix3d& cov_A, const Eigen::Vector3d& mean_B, const Eigen::Matrix3d& cov_B, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  Eigen::Vector3d d = mean_B - (R * mean_A - t);
  Eigen::Matrix3d C = (cov_B + R * cov_A * R.transpose()).inverse();
  return d.dot(C * d);
}
*/

double gicp_loss(const Eigen::Vector3d& mean_A, const Eigen::Matrix3d& cov_A, const Eigen::Vector3d& mean_B, const Eigen::Matrix3d& cov_B, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, Eigen::Matrix<double, 1, 12>* J = nullptr) {
  Eigen::Vector3d d = mean_B - (R * mean_A + t);

  Eigen::Matrix3d RCR = cov_B + R * cov_A * R.transpose();
  Eigen::Matrix3d RCR_inv = RCR.inverse();

  Eigen::Vector3d RCRd = RCR_inv * d;
  double loss = d.dot(RCRd);

  if(!J) {
    return loss;
  }

  Eigen::Matrix<double, 3, 12> jd = dtransform(mean_A, mean_B, R, t);
  Eigen::Matrix<double, 9, 9> jRCR_inv = dmat_inv(RCR) * dRCR(R, cov_A);

  Eigen::Matrix<double, 3, 12> jRCRd = Eigen::Matrix<double, 3, 12>::Zero();
  for(int i = 0; i < 9; i++) {
    Eigen::Matrix3d jc_block = Eigen::Map<Eigen::Matrix3d>(jRCR_inv.col(i).data());
    Eigen::Vector3d jt_block = jd.col(i);
    Eigen::Vector3d jd = jc_block * d + RCR_inv * jt_block;
    jRCRd.col(i) = jd;
  }
  jRCRd.block<3, 3>(0, 9) = RCR_inv * jd.block<3, 3>(0, 9);

  for(int i = 0; i < 12; i++) {
    (*J)[i] = jd.col(i).dot(RCRd) + d.dot(jRCRd.col(i));
  }

  return loss;
}
}  // namespace gicp

#endif