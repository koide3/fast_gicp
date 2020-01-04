#ifndef KKL_OPT_TEST_FUNCTIONS_HPP
#define KKL_OPT_TEST_FUNCTIONS_HPP

#include <Eigen/Core>

namespace kkl {
namespace opt {

template <typename Scalar, int N>
class Sphere {
public:
  static Scalar f(const Eigen::Matrix<Scalar, N, 1>& x) { return x.array().square().sum(); }

  static Eigen::Matrix<Scalar, 1, N> j(const Eigen::Matrix<Scalar, N, 1>& x) { return 2 * x; }

  static Eigen::Matrix<Scalar, N, N> h(const Eigen::Matrix<Scalar, N, 1>& x) { return Eigen::Matrix<Scalar, N, N>::Identity() * 2; }
};

template <typename Scalar, int N>
class Rosenbrok {
public:
  static Scalar f(const Eigen::Matrix<Scalar, N, 1>& x) {
    double sum = 0.0;
    for (int i = 0; i < N - 1; i++) {
      sum += 100 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(x[i] - 1, 2);
    }
    return sum;
  }

  static Eigen::Matrix<Scalar, 1, N> j(const Eigen::Matrix<Scalar, N, 1>& x) {
    Eigen::Matrix<Scalar, 1, N> J = Eigen::Matrix<Scalar, 1, N>::Zero();
    for (int i = 0; i < N - 1; i++) {
      J[i] += 400 * std::pow(x[i], 3) - 400 * x[i] * x[i + 1] + 2 * x[i] - 2;
      J[i + 1] += 200 * (x[i + 1] - std::pow(x[i], 2));
    }

    return J;
  }
};

}  // namespace opt
}  // namespace kkl

#endif