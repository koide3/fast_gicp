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

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

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
    reg_name_ = "FastGICP";
  }
  virtual ~FastGICP() override {
    max_iterations_ = 64;
    transformation_epsilon_ = 5e-4;
    // corr_dist_threshold_ = 1.0;
    corr_dist_threshold_ = std::numeric_limits<float>::max();
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

    update_correspondences(x0);

    auto viewer = guik::LightViewer::instance();
    viewer->update_drawable("target", std::make_shared<glk::PointCloudBuffer>(target_), guik::ShaderSetting().add("color_mode", 1).add("material_color", Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)));

    auto f = [this](const Eigen::Matrix<float, 6, 1>& x) { return loss(x); };
    auto fj = [this](const Eigen::Matrix<float, 6, 1>& x, Eigen::Matrix<float, 1, 6>* J) { return loss(x, J); };
    auto callback = [this](const Eigen::Matrix<float, 6, 1>& x) {
      update_correspondences(x);
      final_transformation_.setIdentity();
      final_transformation_.template block<3, 3>(0, 0) = Sophus::SO3f::exp(x.head<3>()).matrix();
      final_transformation_.template block<3, 1>(0, 3) = x.tail<3>();

      auto viewer = guik::LightViewer::instance();
      viewer->update_drawable("source", std::make_shared<glk::PointCloudBuffer>(input_), guik::ShaderSetting().add("color_mode", 1).add("material_color", Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f)).add("model_matrix", final_transformation_));

      bool kill_switch = false;

      viewer->register_ui_callback("kill_switch", [&]() { kill_switch = ImGui::Button("kill_switch"); });
      viewer->spin_once();
    };

    kkl::opt::BFGS<float, 6> optimizer(f);
    optimizer.set_callback(callback);
    auto result = optimizer.optimize(x0, kkl::opt::TerminationCriteria(64, 1e-6));
    nr_iterations_ = result.num_iterations;

    /*
    for(int i = 0; i < 8192; i++) {
      Eigen::Matrix<float, 1, 6> J;
      loss(x0, &J);
      x0 -= J.normalized() * 1e-3;
      callback(x0);
    }
    */

    viewer->spin();

    final_transformation_.setIdentity();
    final_transformation_.template block<3, 3>(0, 0) = Sophus::SO3f::exp(x0.head<3>()).matrix();
    final_transformation_.template block<3, 1>(0, 3) = x0.tail<3>();

    pcl::transformPointCloud(*input_, output, final_transformation_);
  }

private:
  void update_correspondences(const Eigen::Matrix<float, 6, 1>& x) {
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(x.head<3>()).matrix();
    trans.block<3, 1>(0, 3) = x.tail<3>();

    correspondences.resize(input_->size());
    sq_distances.resize(input_->size());

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

  double loss(const Eigen::Matrix<float, 6, 1>& x, Eigen::Matrix<float, 1, 6>* J = nullptr) const {
    std::cout << "loss" << std::endl;

    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(x.head<3>()).matrix();
    trans.block<3, 1>(0, 3) = x.tail<3>();

    double loss = 0.0;
    Eigen::Matrix<double, 1, 12> sum_jloss = Eigen::Matrix<double, 1, 12>::Zero();

    for(int i = 0; i < input_->size(); i++) {
      int target_index = correspondences[i];
      float sq_dist = sq_distances[i];

      if(sq_dist > corr_dist_threshold_ * corr_dist_threshold_) {
        continue;
      }

      Eigen::Vector4f mean_A = input_->at(i).getVector4fMap();
      const auto& cov_A = source_covs[i];

      Eigen::Vector4f mean_B = target_->at(target_index).getVector4fMap();
      const auto& cov_B = target_covs[target_index];

      if(J) {
        Eigen::Matrix<float, 1, 12> jloss = Eigen::Matrix<float, 1, 12>::Zero();
        loss += fast_gicp::gicp_loss(mean_A, cov_A, mean_B, cov_B, trans, &jloss);
        sum_jloss += jloss.cast<double>();
      } else {
        loss += fast_gicp::gicp_loss(mean_A, cov_A, mean_B, cov_B, trans);
      }
    }

    if(J) {
      Eigen::Matrix<float, 12, 6> jexp = Eigen::Matrix<float, 12, 6>::Zero();
      jexp.block<9, 3>(0, 0) = fast_gicp::dso3_exp(x.head<3>());
      jexp.block<3, 3>(9, 3) = Eigen::Matrix3f::Identity();

      (*J) = sum_jloss.cast<float>() * jexp;
    }

    return loss;
  }

  template<typename PointT>
  bool calculate_covariances(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>>& covariances) {
    kdtree.setInputCloud(cloud);
    covariances.resize(cloud->size());

    for(int i = 0; i < cloud->size(); i++) {
      std::vector<int> k_indices;
      std::vector<float> k_sq_distances;
      kdtree.nearestKSearch(cloud->at(i), 20, k_indices, k_sq_distances);

      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

      Eigen::Matrix<double, 3, 20> data;

      for(int j = 0; j < k_indices.size(); j++) {
        const auto& pt = cloud->at(k_indices[j]);
        data.col(j) = pt.getVector3fMap().template cast<double>();
      }

      data.colwise() -= data.rowwise().mean();
      cov = data * data.transpose();

      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
      // Eigen::Vector3d values = svd.singularValues();
      Eigen::Vector3d values(1, 1, 1e-2);
      // Eigen::Vector3d values = svd.singularValues().array().max(1e-2);
      // Eigen::Vector3d values = svd.singularValues().normalized().array().max(1e-2);

      cov = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = cov.template cast<float>();
    }

    return true;
  }

private:
  pcl::search::KdTree<PointSource> source_kdtree;
  pcl::search::KdTree<PointTarget> target_kdtree;

  std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>> source_covs;
  std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>> target_covs;

  std::vector<int> correspondences;
  std::vector<float> sq_distances;
};
}  // namespace fast_gicp

#endif
