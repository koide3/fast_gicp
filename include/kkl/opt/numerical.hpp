#ifndef KKL_OPT_NUMERICAL_HPP
#define KKL_OPT_NUMERICAL_HPP

#include <Eigen/Core>

namespace kkl {
namespace opt {

template <typename T>
int vector_size(const T& x) {
  return x.size();
}

template <>
int vector_size(const float& x) {
  return 1;
}

template <>
int vector_size(const double& x) {
  return 1;
}

template <typename Func, typename Scalar, int N>
Eigen::Matrix<Scalar, -1, -1> numerical_jacobian(const Func& f, const Eigen::Matrix<Scalar, N, 1>& x, double eps = 1e-6) {
  Eigen::Matrix<Scalar, -1, -1> j;

  for (int i = 0; i < x.size(); i++) {
    Eigen::Matrix<Scalar, N, 1> delta = Eigen::Matrix<Scalar, N, 1>::Zero();
    delta[i] = eps;

    auto y0 = f(x - delta);
    auto y1 = f(x + delta);

    if (j.size() == 0) {
      j.resize(vector_size(y0), x.size());
    }

    j.col(i).array() = (y1 - y0) / (2.0 * eps);
  }

  return j;
};

template <typename Func, typename Scalar, int N>
Eigen::Matrix<Scalar, N, N> numerical_hessian(const Func& f, const Eigen::Matrix<Scalar, N, 1>& x, double eps = 1e-6) {
  Eigen::Matrix<Scalar, N, N> h;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Eigen::Matrix<Scalar, N, 1> dx = Eigen::Matrix<Scalar, N, 1>::Zero();
      dx[i] = eps;

      auto first = [&](const Eigen::Matrix<Scalar, N, 1>& dy) {
        Scalar y0 = f(x - dx + dy);
        Scalar y1 = f(x + dx + dy);
        return (y1 - y0) / (2.0 * eps);
      };

      Eigen::Matrix<Scalar, N, 1> dy = Eigen::Matrix<Scalar, N, 1>::Zero();
      dy[j] = eps;

      Scalar dx0 = first(-dy);
      Scalar dx1 = first(dy);

      h(i, j) = (dx1 - dx0) / (2.0 * eps);
    }
  }

  return h;
}

}  // namespace opt
}  // namespace kkl

#endif