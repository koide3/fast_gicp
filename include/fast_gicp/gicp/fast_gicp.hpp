#ifndef FAST_GICP_FAST_GICP_HPP
#define FAST_GICP_FAST_GICP_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <sophus/so3.hpp>

#include <kkl/opt/solvers/bfgs.hpp>

#include <fast_gicp/gicp/gicp_loss.hpp>
#include <fast_gicp/so3/so3_derivatives.hpp>

namespace fast_gicp {

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

  FastGICP() {
    num_threads_ = omp_get_max_threads();

    reg_name_ = "FastGICP";
    max_iterations_ = 64;
    rotation_epsilon_ = 2e-3;
    transformation_epsilon_ = 5e-4;
    // corr_dist_threshold_ = 1.0;
    corr_dist_threshold_ = std::numeric_limits<float>::max();
  }
  virtual ~FastGICP() override {}

  void setNumThreads(int n) {
    num_threads_ = n;
  }

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override {
    pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
    calculate_covariances(cloud, source_kdtree, source_covs);
  }

  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override {
    pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
    calculate_covariances(cloud, target_kdtree, target_covs);
  }

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override {
    Eigen::Matrix<float, 6, 1> x0;
    x0.head<3>() = Sophus::SO3f(guess.template block<3, 3>(0, 0)).log();
    x0.tail<3>() = guess.template block<3, 1>(0, 3);

    auto callback = [this](const Eigen::Matrix<float, 6, 1>& x) { update_correspondences(x); };

    converged_ = false;
    update_correspondences(x0);
    for(int i = 0; i < max_iterations_; i++) {
      nr_iterations_ = i;

      Eigen::MatrixXf J;
      Eigen::VectorXf loss = loss_ls(x0, &J);

      Eigen::MatrixXf JJ = J.transpose() * J;
      Eigen::VectorXf delta = JJ.inverse() * J.transpose() * loss;

      x0 -= delta;

      if(is_converged(delta)) {
        converged_ = true;
        break;
      }
      callback(x0);
    }

    final_transformation_.setIdentity();
    final_transformation_.template block<3, 3>(0, 0) = Sophus::SO3f::exp(x0.head<3>()).matrix();
    final_transformation_.template block<3, 1>(0, 3) = x0.tail<3>();

    pcl::transformPointCloud(*input_, output, final_transformation_);
  }

private:
  bool is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
    double accum = 0.0;
    Eigen::Matrix3f R = Sophus::SO3f::exp(delta.head<3>()).matrix() - Eigen::Matrix3f::Identity();
    Eigen::Vector3f t = delta.tail<3>();

    Eigen::Matrix3f r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
    Eigen::Vector3f t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

    return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
  }

  void update_correspondences(const Eigen::Matrix<float, 6, 1>& x) {
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(x.head<3>()).matrix();
    trans.block<3, 1>(0, 3) = x.tail<3>();

    correspondences.resize(input_->size());
    sq_distances.resize(input_->size());

#pragma omp parallel for num_threads(num_threads_)
    for(int i = 0; i < input_->size(); i++) {
      PointTarget pt;
      pt.getVector4fMap() = trans * input_->at(i).getVector4fMap();

      std::vector<int> k_indices;
      std::vector<float> k_sq_dists;
      target_kdtree.nearestKSearch(pt, 1, k_indices, k_sq_dists);

      correspondences[i] = k_indices[0];
      sq_distances[i] = k_sq_dists[0];
    }
  }

  Eigen::VectorXf loss_ls(const Eigen::Matrix<float, 6, 1>& x, Eigen::MatrixXf* J) const {
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(x.head<3>()).matrix();
    trans.block<3, 1>(0, 3) = x.tail<3>();

    Eigen::VectorXf loss(input_->size() * 3);
    J->resize(input_->size() * 3, 6);

#pragma omp parallel for num_threads(num_threads_)
    for(int i = 0; i < input_->size(); i++) {
      int target_index = correspondences[i];
      float sq_dist = sq_distances[i];

      if(sq_dist > corr_dist_threshold_ * corr_dist_threshold_) {
        continue;
      }

      const auto& mean_A = input_->at(i).getVector4fMap();
      const auto& cov_A = source_covs[i];

      const auto& mean_B = target_->at(target_index).getVector4fMap();
      const auto& cov_B = target_covs[target_index];

      Eigen::Vector4f transed_mean_A = trans * mean_A;
      Eigen::Vector4f d = mean_B - transed_mean_A;
      Eigen::Matrix4f RCR = cov_B + trans * cov_A * trans.transpose();
      RCR(3, 3) = 1;

      Eigen::Matrix4f RCR_inv = RCR.inverse();
      Eigen::Vector4f RCRd = RCR_inv * d;

      Eigen::Matrix<float, 4, 6> dtdx0 = Eigen::Matrix<float, 4, 6>::Zero();
      dtdx0.block<3, 3>(0, 0) = fast_gicp::skew(transed_mean_A.head<3>());
      dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3f::Identity();

      Eigen::Matrix<float, 4, 6> jlossexp = RCR_inv * dtdx0;

      loss.block<3, 1>(i * 3, 0) = RCRd.head<3>();
      J->block<3, 6>(3 * i, 0) = jlossexp.block<3, 6>(0, 0);
    }

    return loss;
  }

  template<typename PointT>
  bool calculate_covariances(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>>& covariances) {
    kdtree.setInputCloud(cloud);
    covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_)
    for(int i = 0; i < cloud->size(); i++) {
      std::vector<int> k_indices;
      std::vector<float> k_sq_distances;
      kdtree.nearestKSearch(cloud->at(i), 20, k_indices, k_sq_distances);

      Eigen::Matrix<float, 4, 20> data;

      for(int j = 0; j < k_indices.size(); j++) {
        data.col(j) = cloud->at(k_indices[j]).getVector4fMap();
      }

      data.colwise() -= data.rowwise().mean().eval();
      Eigen::Matrix4f cov = data * data.transpose();

      Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      // Eigen::Vector3d values = svd.singularValues();
      Eigen::Vector3f values(1, 1, 1e-2);
      // Eigen::Vector3d values = svd.singularValues().array().max(1e-2);
      // Eigen::Vector3d values = svd.singularValues().normalized().array().max(1e-2);

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }

    return true;
  }

private:
  int num_threads_;
  double rotation_epsilon_;

  pcl::search::KdTree<PointSource> source_kdtree;
  pcl::search::KdTree<PointTarget> target_kdtree;

  std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>> source_covs;
  std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>> target_covs;

  std::vector<int> correspondences;
  std::vector<float> sq_distances;
};
}  // namespace fast_gicp

#endif
