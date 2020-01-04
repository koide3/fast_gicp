#ifndef KKL_OPT_NEWTON_METHOD_HPP
#define KKL_OPT_NEWTON_METHOD_HPP

#include <iostream>

#include <Eigen/LU>
#include <Eigen/Core>
#include <kkl/opt/optimizer.hpp>
#include <kkl/opt/numerical.hpp>

namespace kkl {
namespace opt {

template <typename Scalar, int N>
class NewtonMethod : public Optimizer<Scalar, N> {
public:
  using typename Optimizer<Scalar, N>::Result;
  using typename Optimizer<Scalar, N>::Function;
  using typename Optimizer<Scalar, N>::Jacobian;
  using typename Optimizer<Scalar, N>::Hessian;

  using typename Optimizer<Scalar, N>::VectorN;
  using typename Optimizer<Scalar, N>::RowVectorN;

  NewtonMethod(const Function& f, const Jacobian& j = nullptr, const Hessian& h = nullptr) : Optimizer<Scalar, N>(f), jacobian(j), hessian(h) {
    if (!jacobian) {
      jacobian = [this](const VectorN& x) { return numerical_jacobian(this->function, x); };
    }
    if (!hessian) {
      hessian = [this](const VectorN& x) { return numerical_hessian(this->function, x); };
    }
  }

  virtual Result optimize(const VectorN& x0, const TerminationCriteria& criteria = TerminationCriteria()) {
    Result result;

    VectorN x = x0;
    for (int i = 0; i < criteria.max_iterations; i++) {
      result.num_iterations = i;

      auto j = jacobian(x);
      auto h = hessian(x);

      auto delta = -h.inverse() * j.transpose();
      x = x + delta;
      this->callback(x);

      if (delta.array().abs().sum() < criteria.eps) {
        result.converged = true;
        break;
      }
    }

    return result;
  }

private:
  Jacobian jacobian;
  Hessian hessian;
};

}  // namespace opt
}  // namespace kkl

#endif