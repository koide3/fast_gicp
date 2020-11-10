#ifndef FAST_GICP_FAST_VGICP_HPP
#define FAST_GICP_FAST_VGICP_HPP

#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <sophus/so3.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp_voxel.hpp>

namespace fast_gicp {

/**
 * @brief Fast Voxelized GICP algorithm boosted with OpenMP
 */
template<typename PointSource, typename PointTarget>
class FastVGICP : public FastGICP<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  using Ptr = boost::shared_ptr<FastVGICP<PointSource, PointTarget>>;

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;

  using FastGICP<PointSource, PointTarget>::num_threads_;
  using FastGICP<PointSource, PointTarget>::source_kdtree;
  using FastGICP<PointSource, PointTarget>::target_kdtree;
  using FastGICP<PointSource, PointTarget>::source_covs;
  using FastGICP<PointSource, PointTarget>::target_covs;

public:
  FastVGICP();
  virtual ~FastVGICP() override;

  void setResolution(double resolution);

  void setVoxelAccumulationMode(VoxelAccumulationMode mode);

  void setNeighborSearchMethod(NeighborSearchMethod method);

  virtual void swapSourceAndTarget() override;

  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

private:
  void create_voxelmap(const PointCloudTargetConstPtr& cloud);
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> neighbor_offsets() const;

  Eigen::Vector3i voxel_coord(const Eigen::Vector4d& x) const;
  Eigen::Vector4d voxel_origin(const Eigen::Vector3i& coord) const;
  GaussianVoxel::Ptr lookup_voxel(const Eigen::Vector3i& x) const;

  virtual void update_correspondences(const Eigen::Isometry3d& trans) override;

  virtual void update_mahalanobis(const Eigen::Isometry3d& trans) override;

  virtual double compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr) const override;

private:
  double voxel_resolution_;
  NeighborSearchMethod search_method_;
  VoxelAccumulationMode voxel_mode_;

  using VoxelMap = std::unordered_map<Eigen::Vector3i, GaussianVoxel::Ptr, Vector3iHash, std::equal_to<Eigen::Vector3i>, Eigen::aligned_allocator<std::pair<Eigen::Vector3i, GaussianVoxel::Ptr>>>;
  VoxelMap voxels;

  std::vector<std::pair<int, GaussianVoxel::Ptr>> voxel_correspondences;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> voxel_mahalanobis;
};
}  // namespace fast_gicp

#endif
