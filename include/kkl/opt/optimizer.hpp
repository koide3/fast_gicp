#ifndef KKL_OPT_OPTIMIZER_HPP
#define KKL_OPT_OPTIMIZER_HPP

#include <vector>
#include <functional>
#include <Eigen/Core>

#include <kkl/opt/termination_criteria.hpp>

namespace kkl {
namespace opt {

template <typename Scalar, int N>
struct OptimizationResult {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OptimizationResult() : converged(false), num_iterations(0) {}

  bool converged;
  int num_iterations;
  Eigen::Matrix<Scalar, N, 1> x;
};

template <typename Scalar, int N>
class Optimizer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using VectorN = Eigen::Matrix<Scalar, N, 1>;
  using RowVectorN = Eigen::Matrix<Scalar, 1, N>;
  using MatrixN = Eigen::Matrix<Scalar, N, N>;

  using Function = std::function<Scalar(const VectorN&)>;
  using Jacobian = std::function<RowVectorN(const VectorN&)>;
  using Hessian = std::function<MatrixN(const VectorN&)>;

  using Result = OptimizationResult<Scalar, N>;
  using Callback = std::function<void(const VectorN&)>;
  using ParticlesCallback = std::function<void(const std::vector<VectorN, Eigen::aligned_allocator<VectorN>>&)>;

  Optimizer(const Function& function) : function(function) {}
  virtual ~Optimizer() {}

  virtual Result optimize(const VectorN& x0, const TerminationCriteria& criteria = TerminationCriteria()) { return Result(); }

  void set_callback(const Callback& callback) { this->callback = callback; }
  void set_before_optimization_callback(const Callback& callback) { this->before_optimization = callback; }
  void set_particles_callback(const ParticlesCallback& callback) { this->particles_callback = callback; }

protected:
  Function function;

  Callback callback;
  Callback before_optimization;
  ParticlesCallback particles_callback;
};

}  // namespace opt
}  // namespace kkl

#endif