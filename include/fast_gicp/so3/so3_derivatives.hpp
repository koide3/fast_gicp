#ifndef FAST_GICP_FAST_SO3_DERIVATIVES_HPP
#define FAST_GICP_FAST_SO3_DERIVATIVES_HPP

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fast_gicp {

/**
 * skew-symmetric matrix
 */
Eigen::Matrix3f skew(const Eigen::Vector3f& x) {
  Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
  skew(0, 1) = -x[2];
  skew(0, 2) = x[1];
  skew(1, 0) = x[2];
  skew(1, 2) = -x[0];
  skew(2, 0) = -x[1];
  skew(2, 1) = x[0];

  return skew;
}

/**
 * f(x) = skew(x)
 */
Eigen::Matrix<float, 9, 3> dskew(const Eigen::Vector3f& x) {
  Eigen::Matrix<float, 9, 3> J = Eigen::Matrix<float, 9, 3>::Zero();
  J(5, 0) = 1;
  J(7, 0) = -1;
  J(2, 1) = -1;
  J(6, 1) = 1;
  J(1, 2) = 1;
  J(3, 2) = -1;
  return J;
}

/**
 * f(x) = skew(x) * skew(x)
 */
Eigen::Matrix<float, 9, 3> dskew_sq(const Eigen::Vector3f& x) {
  Eigen::Matrix<float, 9, 3> J = Eigen::Matrix<float, 9, 3>::Zero();

  J(1, 0) = x[1];
  J(2, 0) = x[2];
  J(3, 0) = x[1];
  J(4, 0) = -2 * x[0];
  J(6, 0) = x[2];
  J(8, 0) = -2 * x[0];

  J(0, 1) = -2 * x[1];
  J(1, 1) = x[0];
  J(3, 1) = x[0];
  J(5, 1) = x[2];
  J(7, 1) = x[2];
  J(8, 1) = -2 * x[1];

  J(0, 2) = -2 * x[2];
  J(2, 2) = x[0];
  J(4, 2) = -2 * x[2];
  J(5, 2) = x[1];
  J(6, 2) = x[0];
  J(7, 2) = x[1];

  return J;
}

/**
 * SO3 exponential map (vec3 -> mat3)
 * This is equivalent to Rodrigues formula
 */
Eigen::Matrix3f so3_exp(const Eigen::Vector3f& x) {
  // return Sophus::SO3d::exp(x).matrix();

  double theta_sq = x.dot(x);

  if(std::abs(theta_sq) < 1e-6) {
    Eigen::Matrix3f mat = Eigen::Matrix3f::Identity();
    return mat;
  }

  double theta = std::sqrt(theta_sq);
  Eigen::Matrix3f omega = skew(x);

  Eigen::Matrix3f mat = Eigen::Matrix3f::Identity() + std::sin(theta) / theta * omega + (1 - std::cos(theta)) / theta_sq * omega * omega;
  return mat;
}

/**
 * f(x) = exp(x)
 * x = 3D rotation parameter vector
 */
Eigen::Matrix<float, 9, 3> dso3_exp(const Eigen::Vector3f& x) {
  double theta_sq = x.dot(x);

  // dealing with singularity at theta = 0
  if(std::abs(theta_sq) < 1e-6) {
    Eigen::Matrix<float, 9, 3> J = Eigen::Matrix<float, 9, 3>::Zero();
    J(1, 2) = 1;
    J(2, 1) = -1;
    J(3, 2) = -1;
    J(5, 0) = 1;
    J(6, 1) = 1;
    J(7, 0) = -1;
    return J;
  }

  Eigen::Matrix3f omega = skew(x);
  Eigen::Matrix<float, 9, 3> domega = dskew(x);
  Eigen::Matrix<float, 9, 3> domega_sq = dskew_sq(x);

  double theta = std::sqrt(theta_sq);
  Eigen::Vector3f dtheta = x / x.norm();

  double sin = std::sin(theta);
  double cos = std::cos(theta);

  double sin_ = sin / theta;
  double cos_ = (1 - cos) / theta_sq;
  double dsin_ = cos / theta - sin / theta_sq;
  double dcos_ = sin / theta_sq - 2 * (1 - cos) / (theta_sq * theta);

  Eigen::Matrix<float, 9, 3> J = Eigen::Matrix<float, 9, 3>::Zero();
  for(int i = 0; i < 3; i++) {
    Eigen::Matrix3f dlhs = dsin_ * dtheta[i] * omega;
    Eigen::Matrix3f drhs = sin_ * Eigen::Map<Eigen::Matrix3f>(domega.col(i).data());
    Eigen::Matrix3f sub = dlhs + drhs;

    J.col(i) += Eigen::Map<Eigen::Matrix<float, 9, 1>>(sub.data());
  }

  for(int i = 0; i < 3; i++) {
    Eigen::Matrix3f dlhs = dcos_ * dtheta[i] * omega * omega;
    Eigen::Matrix3f drhs = cos_ * Eigen::Map<Eigen::Matrix3f>(domega_sq.col(i).data());
    Eigen::Matrix3f sub = dlhs + drhs;

    J.col(i) += Eigen::Map<Eigen::Matrix<float, 9, 1>>(sub.data());
  }

  return J;
}

}  // namespace gicp

#endif