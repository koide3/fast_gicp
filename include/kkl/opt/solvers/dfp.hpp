#ifndef KKL_OPT_DFP_HPP
#define KKL_OPT_DFP_HPP

#include <iostream>

#include <Eigen/LU>
#include <Eigen/Core>
#include <kkl/opt/solvers/quasi_newton_method.hpp>

namespace kkl {
namespace opt {

template <typename Scalar, int N>
class DFP : public QuasiNewtonMethod<Scalar, N> {
public:
  using typename Optimizer<Scalar, N>::Function;
  using typename Optimizer<Scalar, N>::Jacobian;

  using typename Optimizer<Scalar, N>::VectorN;
  using typename Optimizer<Scalar, N>::MatrixN;

  DFP(const Function& f, const Jacobian& j = nullptr) : QuasiNewtonMethod<Scalar, N>(f, j) {}

  virtual MatrixN update_B_inv(const MatrixN& B_inv, const VectorN& s, const VectorN& y) const override {
    auto b_inv1 = s * s.transpose() / (y.transpose() * s)(0, 0);
    auto b_inv2 = B_inv * y * y.transpose() * B_inv.transpose() / (y.transpose() * B_inv * y);

    return B_inv + b_inv1 - b_inv2;
  }
};

}  // namespace opt
}  // namespace kkl

#endif