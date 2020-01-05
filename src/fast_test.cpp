#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>
#include <kkl/opt/numerical.hpp>

#include <fast_gicp/gicp/gicp_loss.hpp>
#include <fast_gicp/so3/so3_derivatives.hpp>

using namespace gicp;

bool test() {
  Eigen::Matrix<double, 6, 1> x = Eigen::Matrix<double, 6, 1>::Random();

  Eigen::Vector3d mean_A = Eigen::Vector3d::Random();
  Eigen::Vector3d mean_B = Eigen::Vector3d::Random();

  Eigen::Matrix3d cov_A = Eigen::Matrix3d::Random();
  Eigen::Matrix3d cov_B = Eigen::Matrix3d::Random();

  auto f = [&](const Eigen::Matrix<double, 6, 1>& x_) {
    Eigen::Matrix3d R_ = Sophus::SO3d::exp(x_.head<3>()).matrix();
    Eigen::Vector3d t_ = x_.tail<3>();

    double loss = gicp_loss(mean_A, cov_A, mean_B, cov_B, R_, t_);
    return loss;
  };

  auto t1 = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd nj;
  for(int i = 0; i < 8192; i++) {
    nj = kkl::opt::numerical_jacobian(f, x);
  } 
  auto t2 = std::chrono::high_resolution_clock::now();

  Eigen::Matrix3d R = Sophus::SO3d::exp(x.head<3>()).matrix();
  Eigen::Vector3d t = x.tail<3>();

  Eigen::Matrix<double, 12, 6> jexp = Eigen::Matrix<double, 12, 6>::Zero();
  jexp.block<9, 3>(0, 0) = dso3_exp(x.head<3>());
  jexp.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();

  auto t3 = std::chrono::high_resolution_clock::now();
  Eigen::Matrix<double, 1, 12> jloss;
  for(int i = 0; i < 8192; i++) {
    gicp_loss(mean_A, cov_A, mean_B, cov_B, R, t, &jloss);
  } 

  auto aj = jloss * jexp;
  auto t4 = std::chrono::high_resolution_clock::now();

  double time_n = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e3;
  double time_a = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t2).count() / 1e3;
  double time_a1 = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e3;
  double time_a2 = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e3;

  std::cout << "time_n:" << time_n << " time_a:" << time_a << " (" << time_a1 << "/" << time_a2 << ")" << std::endl;

  if(((nj - aj).array().abs() < 1e-3).all()) {
    return true;
  }

  std::cerr << "JACOBIAN TEST FAILED!!" << std::endl;
  std::cerr << "--- nj ---" << std::endl << nj << std::endl;
  std::cerr << "--- aj ---" << std::endl << aj << std::endl;

  return false;
}

int main(int argc, char** argv) {
  for(int i = 0; i < 1024; i++) {
    test();
  }

  return 0;
}