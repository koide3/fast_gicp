#ifndef FAST_GICP_SO3_HPP
#define FAST_GICP_SO3_HPP

#include <Eigen/Core>

namespace fast_gicp {

inline Eigen::Matrix3f skew(const Eigen::Vector3f& x) {
  Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
  skew(0, 1) = -x[2];
  skew(0, 2) = x[1];
  skew(1, 0) = x[2];
  skew(1, 2) = -x[0];
  skew(2, 0) = -x[1];
  skew(2, 1) = x[0];

  return skew;
}

}

#endif