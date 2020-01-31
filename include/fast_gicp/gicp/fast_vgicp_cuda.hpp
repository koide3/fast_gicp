#ifndef FAST_GICP_FAST_VGICP_CUDA_HPP
#define FAST_GICP_FAST_VGICP_CUDA_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <sophus/so3.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

class FastVGICPCudaCore;

enum NearestNeighborMethod { CPU_PARALLEL_KDTREE, GPU_BRUTEFORCE };

/**
 * @brief Fast Voxelized GICP algorithm boosted with CUDA
 */
template<typename PointSource, typename PointTarget>
class FastVGICPCuda : public pcl::Registration<PointSource, PointTarget, float> {
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

  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;

  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

  FastVGICPCuda();
  virtual ~FastVGICPCuda() override;

  void setResolution(double resolution);

  void setCorrespondenceRandomness(int k);

  void setRegularizationMethod(RegularizationMethod method);

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;

  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  template<typename PointT>
  std::vector<int> find_neighbors_parallel_kdtree(int k, const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree) const;

private:
private:
  int k_correspondences_;
  double rotation_epsilon_;

  pcl::search::KdTree<PointSource> source_kdtree;
  pcl::search::KdTree<PointTarget> target_kdtree;

  double voxel_resolution_;
  RegularizationMethod regularization_method_;
  NearestNeighborMethod neighbor_search_method_;

  std::unique_ptr<FastVGICPCudaCore> vgicp_cuda;
  };
}  // namespace fast_gicp

#endif
