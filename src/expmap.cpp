#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>
#include <fast_gicp/so3/so3.hpp>

Eigen::Quaterniond exp_so3(const Eigen::Vector3d& omega) {
  double theta_sq = omega.dot(omega);

  double theta;
  double imag_factor;
  double real_factor;
  if(theta_sq < 1e-10) {
    theta = 0;
    double theta_quad = theta_sq * theta_sq;
    imag_factor = 0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_quad;
    real_factor = 1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_quad;
  } else {
    theta = std::sqrt(theta_sq);
    double half_theta = 0.5 * theta;
    imag_factor = std::sin(half_theta) / theta;
    real_factor = std::cos(half_theta);
  }

  return Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());
}

int main(int argc, char** argv) {
  Eigen::Vector3d omega = Eigen::Vector3d::Random();

  std::cout << "---" << std::endl << exp_so3(omega).toRotationMatrix() << std::endl;
  std::cout << "---" << std::endl << Sophus::SO3d::exp(omega).matrix() << std::endl;

  return 0;
}