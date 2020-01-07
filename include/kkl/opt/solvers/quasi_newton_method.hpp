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

template<typename Scalar, int N>
class QuasiNewtonMethod : public Optimizer<Scalar, N> {
public:
  using typename Optimizer<Scalar, N>::Result;
  using typename Optimizer<Scalar, N>::Function;
  using typename Optimizer<Scalar, N>::Jacobian;
  using typename Optimizer<Scalar, N>::FunctionJacobian;

  using typename Optimizer<Scalar, N>::VectorN;
  using typename Optimizer<Scalar, N>::RowVectorN;
  using typename Optimizer<Scalar, N>::MatrixN;

  QuasiNewtonMethod(const Function& f, const Jacobian& j = nullptr) : Optimizer<Scalar, N>(f) {
    if(!jacobian) {
      jacobian = [this](const VectorN& x, RowVectorN* J) {
        double y = this->function(x);
        if(J) {
          *J = numerical_jacobian(this->function, x);
        }
        return y;
      };
    }
  }

  QuasiNewtonMethod(const FunctionJacobian& fj) : Optimizer<Scalar, N>([=](const VectorN& x) { return fj(x, nullptr); }), jacobian(fj) {}

  virtual Result optimize(const VectorN& x0, const TerminationCriteria& criteria = TerminationCriteria()) override {
    Result result;
    VectorN x = x0;

    MatrixN B_inv = MatrixN::Identity();
    for(int i = 0; i < criteria.max_iterations; i++) {
      if(this->before_optimization) {
        this->before_optimization(x);
      }

      result.num_iterations = i;
      RowVectorN j;
      double f0 = jacobian(x, &j);

      VectorN p = (-B_inv * j.transpose()).normalized();

      // BackTrackingSearch line_search([&](double alpha) { return jacobian(x + p * alpha, nullptr); }, f0, jnorm);
      // GoldenSectionSearch line_search([&](Scalar alpha) { return this->function(x + p * alpha); });
      // NelderMeadLineSearch<Scalar> line_search([&](Scalar alpha) { return this->function(x + p * alpha); });
      // double alpha = line_search.minimize(0.0, 1.0, TerminationCriteria(10, 1e-4));
      double alpha = backtracking(f0, x, p, j, 1e-3, 0.0, 0.1);

      VectorN s = alpha * p;
      x = x + s;

      RowVectorN j2;
      jacobian(x, &j2);
      VectorN y = j2 - j;

      B_inv = update_B_inv(B_inv, s, y);

      if(this->callback) {
        this->callback(x);
      }

      if(s.norm() < (x - s).norm() * criteria.eps) {
        result.converged = true;
        break;
      }
    }

    result.x = x;
    return result;
  }

  double backtracking(double f0, const VectorN& x, const VectorN& p, const RowVectorN& j, double c, double min_alpha, double max_alpha) const {
    double alpha = max_alpha;
    double rho = 0.5;
    double pj = p.dot(j);

    while(this->function(x + alpha * p) > f0 + c * alpha * pj && alpha > min_alpha) {
      alpha *= rho;
    }

    return std::max(min_alpha, alpha);
  }

  virtual MatrixN update_B_inv(const MatrixN& B_inv, const VectorN& s, const VectorN& y) const = 0;

private:
  FunctionJacobian jacobian;
};

}  // namespace opt
}  // namespace kkl

#endif