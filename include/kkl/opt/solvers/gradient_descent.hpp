#ifndef KKL_OPT_GRADIENT_DESCENT_HPP
#define KKL_OPT_GRADIENT_DESCENT_HPP

#include <iostream>

#include <Eigen/LU>
#include <Eigen/Core>
#include <kkl/opt/optimizer.hpp>
#include <kkl/opt/numerical.hpp>
#include <kkl/opt/solvers/nelder_mead.hpp>
#include <kkl/opt/solvers/golden_section_search.hpp>

namespace kkl {
namespace opt {

template <typename Scalar, int N>
class GradientDescent : public Optimizer<Scalar, N> {
public:
  using typename Optimizer<Scalar, N>::Result;
  using typename Optimizer<Scalar, N>::Function;
  using typename Optimizer<Scalar, N>::Jacobian;

  using typename Optimizer<Scalar, N>::VectorN;
  using typename Optimizer<Scalar, N>::RowVectorN;

  GradientDescent(const Function& f, const Jacobian& j = nullptr, double alpha = 1e-2) : Optimizer<Scalar, N>(f), jacobian(j), alpha(alpha) {
    if (!jacobian) {
      jacobian = [this](const VectorN& x) { return numerical_jacobian(this->function, x); };
    }
  }

  virtual Result optimize(const VectorN& x0, const TerminationCriteria& criteria = TerminationCriteria()) {
    Result result;

    VectorN x = x0;
    for (int i = 0; i < criteria.max_iterations; i++) {
      result.num_iterations = i;

      auto j = jacobian(x);
      VectorN delta;

      if (alpha <= 0.0) {
        NelderMeadLineSearch<Scalar> line_search([&](Scalar a) { return this->function(x - a * j.transpose()); });
        delta = -line_search.minimize(0.0, 1e-1) * j.transpose();
      } else {
        delta = -alpha * j.transpose();
      }

      x = x + delta;

      this->callback(x);

      if (delta.array().abs().sum() < criteria.eps) {
        result.converged = true;
        break;
      }
    }

    result.x = x;
    return result;
  }

private:
  double alpha;
  Jacobian jacobian;
};

}  // namespace opt
}  // namespace kkl

#endif