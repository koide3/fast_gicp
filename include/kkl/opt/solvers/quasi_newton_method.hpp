#ifndef KKL_OPT_QUASI_NEWTON_METHOD_HPP
#define KKL_OPT_QUASI_NEWTON_METHOD_HPP

#include <iostream>

#include <Eigen/LU>
#include <Eigen/Core>
#include <kkl/opt/optimizer.hpp>
#include <kkl/opt/numerical.hpp>
#include <kkl/opt/solvers/nelder_mead.hpp>
#include <kkl/opt/solvers/backtracking_search.hpp>
#include <kkl/opt/solvers/golden_section_search.hpp>

namespace kkl {
namespace opt {

template <typename Scalar, int N>
class QuasiNewtonMethod : public Optimizer<Scalar, N> {
public:
  using typename Optimizer<Scalar, N>::Result;
  using typename Optimizer<Scalar, N>::Function;
  using typename Optimizer<Scalar, N>::Jacobian;

  using typename Optimizer<Scalar, N>::VectorN;
  using typename Optimizer<Scalar, N>::RowVectorN;
  using typename Optimizer<Scalar, N>::MatrixN;

  QuasiNewtonMethod(const Function& f, const Jacobian& j = nullptr) : Optimizer<Scalar, N>(f), jacobian(j) {
    if (!jacobian) {
      jacobian = [this](const VectorN& x) { return numerical_jacobian(this->function, x); };
    }
  }

  virtual Result optimize(const VectorN& x0, const TerminationCriteria& criteria = TerminationCriteria()) override {
    Result result;
    VectorN x = x0;

    MatrixN B_inv = MatrixN::Identity();
    for (int i = 0; i < criteria.max_iterations; i++) {
      if(this->before_optimization) {
        this->before_optimization(x);
      }

      result.num_iterations = i;
      VectorN j = jacobian(x).transpose();

      VectorN p = -B_inv * j;
      if(p.norm() > 1.0) {
        p.normalize();
      }

      Scalar f0 = this->function(x);
      Scalar jnorm = j.norm();

      BackTrackingSearch<Scalar> line_search([&](Scalar alpha) { return this->function(x + p * alpha); }, f0, jnorm);
      // GoldenSectionSearch<Scalar> line_search([&](Scalar alpha) { return this->function(x + p * alpha); });
      // NelderMeadLineSearch<Scalar> line_search([&](Scalar alpha) { return this->function(x + p * alpha); });
      Scalar alpha = line_search.minimize(0.0, 0.5, TerminationCriteria(10, 1e-3));

      VectorN s = alpha * p;
      x = x + s;

      VectorN j2 = jacobian(x).transpose();
      VectorN y = j2 - j;
      j = j2;

      B_inv = update_B_inv(B_inv, s, y);

      if(this->callback) {
        this->callback(x);
      }

      if (s.array().abs().sum() < criteria.eps) {
        result.converged = true;
        break;
      }
    }

    result.x = x;
    return result;
  }

  virtual MatrixN update_B_inv(const MatrixN& B_inv, const VectorN& s, const VectorN& y) const = 0;

private:
  double alpha;
  Jacobian jacobian;
};

}  // namespace opt
}  // namespace kkl

#endif