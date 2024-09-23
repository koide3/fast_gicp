#ifndef FAST_GICP_LSQ_REGISTRATION_HPP
#define FAST_GICP_LSQ_REGISTRATION_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

namespace fast_gicp {

enum class LSQ_OPTIMIZER_TYPE { GaussNewton, LevenbergMarquardt };

template <int Dim>
struct OptimizationParamProcessor {
  virtual Eigen::Matrix<double, Dim, Dim> reduce_H(const Eigen::Matrix<double, 6, 6>& H_in) const = 0;
  virtual Eigen::Matrix<double, Dim, 1> reduce_b(const Eigen::Matrix<double, 6, 1>& b_in) const = 0;
  virtual Eigen::Matrix<double, 6, 1> expand_b(const Eigen::Matrix<double, Dim, 1>& b_in) const = 0;
  using Ptr = std::shared_ptr<OptimizationParamProcessor<Dim>>;
};

template <>
struct OptimizationParamProcessor<3> {
  virtual Eigen::Matrix<double, 3, 3> reduce_H(const Eigen::Matrix<double, 6, 6>& H_in) const { return H_in.template topLeftCorner<3, 3>(); }
  virtual Eigen::Matrix<double, 3, 1> reduce_b(const Eigen::Matrix<double, 6, 1>& b_in) const { return b_in.template topLeftCorner<3, 1>(); }
  virtual Eigen::Matrix<double, 6, 1> expand_b(const Eigen::Matrix<double, 3, 1>& b_in) const {
    Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
    d.topLeftCorner<3, 1>(3, 1) = b_in;
    return d;
  }
  using Ptr = std::shared_ptr<OptimizationParamProcessor<3>>;
};

template <>
struct OptimizationParamProcessor<6> {
  virtual Eigen::Matrix<double, 6, 6> reduce_H(const Eigen::Matrix<double, 6, 6>& H_in) const { return H_in; }
  virtual Eigen::Matrix<double, 6, 1> reduce_b(const Eigen::Matrix<double, 6, 1>& b_in) const { return b_in; }
  virtual Eigen::Matrix<double, 6, 1> expand_b(const Eigen::Matrix<double, 6, 1>& b_in) const { return b_in; }
  using Ptr = std::shared_ptr<OptimizationParamProcessor<6>>;
};

struct TranslationOnly : public OptimizationParamProcessor<3> {
  Eigen::Matrix<double, 3, 3> reduce_H(const Eigen::Matrix<double, 6, 6>& H_in) const override { return H_in.template bottomRightCorner<3, 3>(); }
  Eigen::Matrix<double, 3, 1> reduce_b(const Eigen::Matrix<double, 6, 1>& b_in) const override { return b_in.template bottomRightCorner<3, 1>(); }
  Eigen::Matrix<double, 6, 1> expand_b(const Eigen::Matrix<double, 3, 1>& b_in) const override {
    Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
    d.bottomRightCorner<3, 1>() = b_in;
    return d;
  }
};

template <bool B, bool... Args>
struct count_true {
  static constexpr int value = count_true<B>::value + count_true<Args...>::value;
};

template <>
struct count_true<true> {
  static constexpr int value = 1;
};
template <>
struct count_true<false> {
  static constexpr int value = 0;
};

template <bool... Args>
struct CustomDOF : public OptimizationParamProcessor<(Args + ... + 0)> {
  using KeepDOF = std::array<bool, 6>;
  static constexpr int Dim = (Args + ... + 0);
  KeepDOF keep_dof_;
  /* template <std::same_as<bool>... Ts> */
  /*   requires(sizeof...(Args) == sizeof...(Ts)) */
  CustomDOF(decltype(Args)... args) {
    static_assert(sizeof...(args) <= 6);

    int i = 0;
    ([&] { keep_dof_[i++] = args; }(), ...);

    for (; i < 6; ++i) {
      keep_dof_[i] = true;
    }
  }

  Eigen::Matrix<double, Dim, Dim> reduce_H(const Eigen::Matrix<double, 6, 6>& H_in) const { return H_in(keep_dof_, keep_dof_); }
  Eigen::Matrix<double, Dim, 1> reduce_b(const Eigen::Matrix<double, 6, 1>& b_in) const { return b_in(keep_dof_); }
  Eigen::Matrix<double, 6, 1> expand_b(const Eigen::Matrix<double, Dim, 1>& b_in) const {
    Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
    d(keep_dof_) = b_in;
    return d;
  }
};

template <typename PointSource, typename PointTarget, int Dim = 6>
class LsqRegistration : public pcl::Registration<PointSource, PointTarget, float> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr = pcl::shared_ptr<LsqRegistration<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const LsqRegistration<PointSource, PointTarget>>;
#else
  using Ptr = boost::shared_ptr<LsqRegistration<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const LsqRegistration<PointSource, PointTarget>>;
#endif

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <int D = Dim, typename std::enable_if<D == 6, bool>::type = true>
  LsqRegistration() : LsqRegistration(std::make_shared<OptimizationParamProcessor<6>>()) {}

  LsqRegistration(std::shared_ptr<OptimizationParamProcessor<Dim>> processor);

  virtual ~LsqRegistration();

  void setRotationEpsilon(double eps);
  void setInitialLambdaFactor(double init_lambda_factor);
  void setDebugPrint(bool lm_debug_print);

  void setOptimizationParamProcessor(const typename OptimizationParamProcessor<Dim>::Ptr processor);

  const Eigen::Matrix<double, 6, 6>& getFinalHessian() const;

  double evaluateCost(const Eigen::Matrix4f& relative_pose, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr);

  virtual void swapSourceAndTarget() {}
  virtual void clearSource() {}
  virtual void clearTarget() {}

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  bool is_converged(const Eigen::Isometry3d& delta) const;

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr) = 0;
  virtual double compute_error(const Eigen::Isometry3d& trans) = 0;

  bool step_optimize(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);
  bool step_gn(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);
  bool step_lm(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);

protected:
  double rotation_epsilon_;

  LSQ_OPTIMIZER_TYPE lsq_optimizer_type_;
  int lm_max_iterations_;
  double lm_init_lambda_factor_;
  double lm_lambda_;
  bool lm_debug_print_;

  typename OptimizationParamProcessor<Dim>::Ptr process_params_;

  Eigen::Matrix<double, 6, 6> final_hessian_;
};
}  // namespace fast_gicp

#endif
