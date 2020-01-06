#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>
#include <kkl/opt/numerical.hpp>

#include <fast_gicp/gicp/gicp_loss.hpp>
#include <fast_gicp/so3/so3_derivatives.hpp>

bool test() {
  return false;
}

int main(int argc, char** argv) {
  for(int i = 0; i < 1024; i++) {
    test();
  }

  return 0;
}