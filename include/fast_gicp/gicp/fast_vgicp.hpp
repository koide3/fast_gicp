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
#include <fast_gicp/gicp/fast_vgicp_voxel.hpp>

namespace fast_gicp {

/**
 * @brief Fast Voxelized GICP algorithm boosted with OpenMP
 */
template<typename PointSource, typename PointTarget>
class FastVGICP : public pcl::Registration<PointSource, PointTarget, float> {
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

  FastVGICP();
  virtual ~FastVGICP() override;

  void setNumThreads(int n);

  void setResolution(double resolution);

  void setCorrespondenceRandomness(int k);

  void setRegularizationMethod(RegularizationMethod method);

  void setNeighborSearchMethod(NeighborSearchMethod method);

  void setVoxelAccumulationMode(VoxelAccumulationMode mode);

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;

  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

private:
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> neighbor_offsets() const;

  Eigen::Vector3i voxel_coord(const Eigen::Vector4f& x) const;
  Eigen::Vector4f voxel_origin(const Eigen::Vector3i& coord) const;
  GaussianVoxel::Ptr lookup_voxel(const Eigen::Vector3i& x) const;

  bool is_converged(const Eigen::Matrix<float, 6, 1>& delta) const;

  Eigen::VectorXf loss_ls(const Eigen::Matrix<float, 6, 1>& x, Eigen::MatrixXf* J) const;

  template<typename PointT>
  bool calculate_covariances(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>>& covariances);

private:
  int num_threads_;
  int k_correspondences_;
  double rotation_epsilon_;

  pcl::search::KdTree<PointSource> source_kdtree;
  pcl::search::KdTree<PointTarget> target_kdtree;

  std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>> source_covs;
  std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>> target_covs;

  double voxel_resolution_;
  NeighborSearchMethod search_method_;
  RegularizationMethod regularization_method_;
  VoxelAccumulationMode voxel_mode_;

  using VoxelMap = std::unordered_map<Eigen::Vector3i, GaussianVoxel::Ptr, Vector3iHash, std::equal_to<Eigen::Vector3i>, Eigen::aligned_allocator<std::pair<Eigen::Vector3i, GaussianVoxel::Ptr>>>;
  VoxelMap voxels;
};
}  // namespace fast_gicp

#endif
