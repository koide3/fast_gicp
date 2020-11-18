#ifndef FAST_GICP_FAST_VGICP_CUDA_HPP
#define FAST_GICP_FAST_VGICP_CUDA_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

class FastVGICPCudaCore;

enum class NearestNeighborMethod { CPU_PARALLEL_KDTREE, GPU_BRUTEFORCE };

/**
 * @brief Fast Voxelized GICP algorithm boosted with CUDA
 */
template<typename PointSource, typename PointTarget>
class FastVGICPCuda : public FastVGICP<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  using Ptr = boost::shared_ptr<FastVGICPCuda<PointSource, PointTarget>>;

  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;

  using FastGICP<PointSource, PointTarget>::k_correspondences_;
  using FastVGICP<PointSource, PointTarget>::voxel_resolution_;
  using FastVGICP<PointSource, PointTarget>::regularization_method_;

  /*
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;

  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;
  */

  FastVGICPCuda();
  virtual ~FastVGICPCuda() override;

  void setNearesetNeighborSearchMethod(NearestNeighborMethod method);

  virtual void swapSourceAndTarget() override;

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;

  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual void update_correspondences(const Eigen::Isometry3d& trans) override;

  virtual void update_mahalanobis(const Eigen::Isometry3d& trans) override;

  virtual double compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr) const override;

  template<typename PointT>
  std::vector<int> find_neighbors_parallel_kdtree(int k, const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree) const;

private:
  pcl::search::KdTree<PointSource> source_kdtree;
  pcl::search::KdTree<PointTarget> target_kdtree;

  NearestNeighborMethod neighbor_search_method_;

  std::unique_ptr<FastVGICPCudaCore> vgicp_cuda;
};

}  // namespace fast_gicp

#endif
