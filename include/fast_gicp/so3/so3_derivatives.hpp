#ifndef FAST_GICP_SO3_DERIVATIVES_HPP
#define FAST_GICP_SO3_DERIVATIVES_HPP

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gicp {

/**
 * skew-symmetric matrix
 */
Eigen::Matrix3d skew(const Eigen::Vector3d& x) {
  Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
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
Eigen::Matrix<double, 9, 3> dskew(const Eigen::Vector3d& x) {
  Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
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
Eigen::Matrix<double, 9, 3> dskew_sq(const Eigen::Vector3d& x) {
  Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();

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
 */
Eigen::Matrix3d so3_exp(const Eigen::Vector3d& x) {
  // return Sophus::SO3d::exp(x).matrix();

  Eigen::Matrix3d omega = skew(x);

  double theta_sq = x.dot(x);
  double theta = std::sqrt(theta_sq);

  Eigen::Matrix3d mat = Eigen::Matrix3d::Identity() + std::sin(theta) / theta * omega + (1 - std::cos(theta)) / theta_sq * omega * omega;

  return mat;
}

/**
 * f(x) = exp(x)
 * x = 3D rotation parameter vector
 */
Eigen::Matrix<double, 9, 3> dso3_exp(const Eigen::Vector3d& x) {
  Eigen::Matrix3d omega = skew(x);
  Eigen::Matrix<double, 9, 3> domega = dskew(x);
  Eigen::Matrix<double, 9, 3> domega_sq = dskew_sq(x);

  double theta_sq = x.dot(x);
  double theta = std::sqrt(theta_sq);
  Eigen::Vector3d dtheta;

  double sin, cos;
  double sin_, cos_;
  double dsin_, dcos_;

  // dealing with singularity at theta = 0
  // TOTO: just returning the generators would be a better way
  if(std::abs(theta) < 1e-6) {
    dtheta.setZero();

    sin = 0;
    cos = 1;

    sin_ = 1;
    cos_ = 0.5;

    dsin_ = 0;  // this term will be canceled by dtheta = 0
    dcos_ = 0;  // 
  } else {
    dtheta = x / x.norm();

    sin = std::sin(theta);
    cos = std::cos(theta);

    sin_ = sin / theta;
    cos_ = (1 - cos) / theta_sq;

    dsin_ = cos / theta - sin / theta_sq;
    dcos_ = sin / theta_sq - 2 * (1 - cos) / (theta_sq * theta);
  }

  Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
  for(int i = 0; i < 3; i++) {
    Eigen::Matrix3d dlhs = dsin_ * dtheta[i] * omega;
    Eigen::Matrix3d drhs = sin_ * Eigen::Map<Eigen::Matrix3d>(domega.col(i).data());
    Eigen::Matrix3d sub = dlhs + drhs;

    J.col(i) += Eigen::Map<Eigen::Matrix<double, 9, 1>>(sub.data());
  }

  for(int i = 0; i < 3; i++) {
    Eigen::Matrix3d dlhs = dcos_ * dtheta[i] * omega * omega;
    Eigen::Matrix3d drhs = cos_ * Eigen::Map<Eigen::Matrix3d>(domega_sq.col(i).data());
    Eigen::Matrix3d sub = dlhs + drhs;

    J.col(i) += Eigen::Map<Eigen::Matrix<double, 9, 1>>(sub.data());
  }

  return J;
}

}

#endif