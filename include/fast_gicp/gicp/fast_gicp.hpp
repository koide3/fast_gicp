#ifndef FAST_GICP_FAST_GICP_HPP
#define FAST_GICP_FAST_GICP_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

/**
 * @brief Fast GICP algorithm optimized for multi threading with OpenMP
 */
template<typename PointSource, typename PointTarget>
class FastGICP : public pcl::Registration<PointSource, PointTarget, float> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;

  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

  FastGICP();
  virtual ~FastGICP() override;

  void setNumThreads(int n);

  void setRotationEpsilon(double eps);

  void setCorrespondenceRandomness(int k);

  void setRegularizationMethod(RegularizationMethod method);

  // ```tau``` in LM optimization
  // use a small value (e.g., 1e-9) when the initial guess is expected to be accurate (e.g., odometry estimation)
  // use a large value (e.g., 1e-4) when the initial guess would be inaccurate
  void setInitialLambdaFactor(double init_lambda_factor);

  void setMaxInnerIterations(int max_iterations);

  void swapSourceAndTarget();

  void clearSource();

  void clearTarget();

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;

  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

private:
  bool is_converged(const Eigen::Isometry3d& delta) const;

  void update_correspondences(const Eigen::Isometry3d& trans);

  void update_mahalanobis(const Eigen::Isometry3d& trans);

  double compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr) const;

  bool lm_step(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);

  template<typename PointT>
  bool calculate_covariances(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);

public:
  int num_threads_;
  int k_correspondences_;
  double rotation_epsilon_;

  int lm_max_iterations_;
  double lm_init_lambda_factor_;
  double lm_lambda_;

  RegularizationMethod regularization_method_;

  std::unique_ptr<pcl::search::KdTree<PointSource>> source_kdtree;
  std::unique_ptr<pcl::search::KdTree<PointTarget>> target_kdtree;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_covs;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_covs;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis;

  std::vector<int> correspondences;
  std::vector<float> sq_distances;
};
}  // namespace fast_gicp

#endif
