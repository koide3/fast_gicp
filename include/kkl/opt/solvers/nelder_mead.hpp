#ifndef KKL_OPT_NELDER_MEAD_HPP
#define KKL_OPT_NELDER_MEAD_HPP

#include <numeric>
#include <kkl/opt/optimizer.hpp>
#include <kkl/opt/line_search.hpp>

namespace kkl {
namespace opt {

template <typename Scalar>
class NelderMeadLineSearch : public LineSearch<Scalar> {
public:
  NelderMeadLineSearch(const std::function<Scalar(Scalar)>& f) : LineSearch<Scalar>(f), alpha(1.0), gamma(2.0), rho(0.5), sigma(0.5) {}
  virtual ~NelderMeadLineSearch() override {}

  virtual Scalar minimize(Scalar x0, Scalar x1, const TerminationCriteria& criteria = TerminationCriteria()) override {
    std::array<Scalar, 2> x = {x0, x1};
    std::array<Scalar, 2> y = {this->function(x0), this->function(x1)};

    for (int i = 0; i < criteria.max_iterations; i++) {
      if (y[1] < y[0]) {
        std::swap(x[0], x[1]);
        std::swap(y[0], y[1]);
      }

      if (is_converged(x, criteria)) {
        break;
      }

      Scalar xo = x[0];
      Scalar yo = y[0];

      Scalar xr = xo + alpha * (xo - x.back());
      Scalar yr = this->function(xr);

      if (y[0] <= yr && yr < y[0]) {
        // this never happen?
        x.back() = xr;
        y.back() = yr;
      } else if (yr < y[0]) {
        Scalar xe = xo + gamma * (xo - x.back());
        Scalar ye = this->function(xe);

        if (yr < yr) {
          x.back() = xe;
          y.back() = ye;
        } else {
          x.back() = xr;
          y.back() = yr;
        }
      } else {
        Scalar xc = xo + rho * (xo - x.back());
        Scalar yc = this->function(xc);

        if (yc < y.back()) {
          x.back() = xc;
          y.back() = yc;
        } else {
          x.back() = x[0] + rho * (x.back() - x[0]);
          y.back() = this->function(x.back());
        }
      }
    }

    return x[0];
  }

  bool is_converged(const std::array<Scalar, 2>& x, const TerminationCriteria& criteria) {
    Scalar diff = std::abs(x[0] - x[1]);
    return diff < criteria.eps;
  }

private:
  const Scalar alpha;
  const Scalar gamma;
  const Scalar rho;
  const Scalar sigma;
};

template <typename Scalar, int N>
class NelderMead : public Optimizer<Scalar, N> {
public:
  using typename Optimizer<Scalar, N>::Result;
  using typename Optimizer<Scalar, N>::Function;

  using typename Optimizer<Scalar, N>::VectorN;
  using VectorM = Eigen::Matrix<Scalar, N + 1, 1>;  // value & sample

  NelderMead(const Function& f, Scalar step = 0.1) : Optimizer<Scalar, N>(f), step(step), alpha(1.0), gamma(2.0), rho(0.5), sigma(0.5) {}

  virtual Result optimize(const VectorN& x0, const TerminationCriteria& criteria = TerminationCriteria()) {
    Result result;

    std::vector<VectorM, Eigen::aligned_allocator<VectorM>> x(1);
    x[0][0] = this->function(x0);
    x[0].template tail<N>() = x0;

    for (int i = 0; i < N; i++) {
      VectorM xi;
      xi.template tail<N>() = x0;
      xi.template tail<N>()[i] += step;

      xi[0] = this->function(xi.template tail<N>());
      x.push_back(xi);
    }

    for (int i = 0; i < criteria.max_iterations; i++) {
      result.num_iterations = i;
      std::sort(x.begin(), x.end(), [=](const VectorM& lhs, const VectorM& rhs) { return lhs[0] < rhs[0]; });
      if (is_converted(x, criteria)) {
        result.converged = true;
        break;
      }

      VectorM xo = std::accumulate(x.begin(), x.end() - 1, VectorM::Zero().eval()) / (x.size() - 1);
      xo[0] = this->function(xo.template tail<N>());

      VectorM xr = xo + alpha * (xo - x.back());
      xr[0] = this->function(xr.template tail<N>());

      if (x[0][0] <= xr[0] && xr[0] < x[N - 1][0]) {
        x.back() = xr;
      } else if (xr[0] < x[0][0]) {
        VectorM xe = xo + gamma * (xo - x.back());
        xe[0] = this->function(xe.template tail<N>());

        if (xe[0] < xr[0]) {
          x.back() = xe;
        } else {
          x.back() = xr;
        }
      } else {
        VectorM xc = xo + rho * (xo - x.back());
        xc[0] = this->function(xc.template tail<N>());

        if (xc[0] < x.back()[0]) {
          x.back() = xc;
        } else {
          for (int j = 1; j < x.size(); j++) {
            x[j] = x[0] + rho * (x[j] - x[0]);
            x[j][0] = this->function(x[j].template tail<N>());
          }
        }
      }

      if(this->callback) {
        this->callback(x[0].template tail<N>());
      }

      if (this->particles_callback) {
        std::vector<VectorN, Eigen::aligned_allocator<VectorN>> particles(x.size());
        std::transform(x.begin(), x.end(), particles.begin(), [=](const VectorM& xi) { return xi.template tail<N>(); });
        this->particles_callback(particles);
      }
    }

    result.x = x[0].template tail<N>();
    return result;
  }

private:
  bool is_converted(const std::vector<VectorM, Eigen::aligned_allocator<VectorM>>& x, const TerminationCriteria& criteria) {
    VectorM mean = std::accumulate(x.begin(), x.end(), VectorM::Zero().eval()) / x.size();
    VectorM var = VectorM::Zero();
    for (const auto& xi : x) {
      var = var.array() + (xi - mean).array().square();
    }

    return var.template tail<N>().sum() < criteria.eps;
  }

private:
  Scalar step;

  const Scalar alpha;
  const Scalar gamma;
  const Scalar rho;
  const Scalar sigma;
};
}  // namespace opt
}  // namespace kkl

#endif