#ifndef FAST_GICP_FAST_GICP_ST_HPP
#define FAST_GICP_FAST_GICP_ST_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

/**
 * @brief Fast GICP algorithm optimized for single threading
 */
template<typename PointSource, typename PointTarget>
class FastGICPSingleThread : public FastGICP<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;

  using FastGICP<PointSource, PointTarget>::target_kdtree;
  using FastGICP<PointSource, PointTarget>::correspondences;
  using FastGICP<PointSource, PointTarget>::sq_distances;
  using FastGICP<PointSource, PointTarget>::source_covs;
  using FastGICP<PointSource, PointTarget>::target_covs;
  using FastGICP<PointSource, PointTarget>::mahalanobis;

public:
  FastGICPSingleThread();
  virtual ~FastGICPSingleThread() override;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

private:
  virtual void update_correspondences(const Eigen::Isometry3d& trans) override;

  virtual void update_mahalanobis(const Eigen::Isometry3d& trans) override;

  virtual double compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr) const override;

private:
  std::vector<float> second_sq_distances;
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> anchors;
};
}  // namespace fast_gicp

#endif
