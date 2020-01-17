#ifndef FAST_GICP_FAST_GICP_IMPL_HPP
#define FAST_GICP_FAST_GICP_IMPL_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <sophus/so3.hpp>
#include <fast_gicp/so3/so3.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/opt/gauss_newton.hpp>

namespace fast_gicp {

template<typename PointSource, typename PointTarget>
FastGICP<PointSource, PointTarget>::FastGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  reg_name_ = "FastGICP";
  max_iterations_ = 64;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;
  // corr_dist_threshold_ = 1.0;
  regularization_method_ = PLANE;
  corr_dist_threshold_ = std::numeric_limits<float>::max();
}

template<typename PointSource, typename PointTarget>
FastGICP<PointSource, PointTarget>::~FastGICP() {}

template<typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setNumThreads(int n) {
  num_threads_ = n;

#ifdef _OPENMP
  if(n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template<typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template<typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  calculate_covariances(cloud, source_kdtree, source_covs);
}

template<typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  calculate_covariances(cloud, target_kdtree, target_covs);
}

template<typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Matrix<float, 6, 1> x0;
  x0.head<3>() = Sophus::SO3f(guess.template block<3, 3>(0, 0)).log();
  x0.tail<3>() = guess.template block<3, 1>(0, 3);

  if(x0.head<3>().norm() < 1e-1) {
    x0.head<3>() = (Eigen::Vector3f::Random()).normalized() * 1e-1;
  }

  converged_ = false;
  GaussNewton<double, 6> solver;

  for(int i = 0; i < max_iterations_; i++) {
    nr_iterations_ = i;

    update_correspondences(x0);
    Eigen::MatrixXf J;
    Eigen::VectorXf loss = loss_ls(x0, &J);

    Eigen::Matrix<float, 6, 1> delta = solver.delta(loss.cast<double>(), J.cast<double>()).cast<float>();

    x0.head<3>() = (Sophus::SO3f::exp(-delta.head<3>()) * Sophus::SO3f::exp(x0.head<3>())).log();
    x0.tail<3>() -= delta.tail<3>();

    if(is_converged(delta)) {
      converged_ = true;
      break;
    }
  }

  final_transformation_.setIdentity();
  final_transformation_.template block<3, 3>(0, 0) = Sophus::SO3f::exp(x0.head<3>()).matrix();
  final_transformation_.template block<3, 1>(0, 3) = x0.tail<3>();

  pcl::transformPointCloud(*input_, output, final_transformation_);
}

template<typename PointSource, typename PointTarget>
bool FastGICP<PointSource, PointTarget>::is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
  double accum = 0.0;
  Eigen::Matrix3f R = Sophus::SO3f::exp(delta.head<3>()).matrix() - Eigen::Matrix3f::Identity();
  Eigen::Vector3f t = delta.tail<3>();

  Eigen::Matrix3f r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3f t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template<typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Matrix<float, 6, 1>& x) {
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

template<typename PointSource, typename PointTarget>
Eigen::VectorXf FastGICP<PointSource, PointTarget>::loss_ls(const Eigen::Matrix<float, 6, 1>& x, Eigen::MatrixXf* J) const {
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
    dtdx0.block<3, 3>(0, 0) = skew(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3f::Identity();

    Eigen::Matrix<float, 4, 6> jlossexp = RCR_inv * dtdx0;

    loss.block<3, 1>(i * 3, 0) = RCRd.head<3>();
    J->block<3, 6>(3 * i, 0) = jlossexp.block<3, 6>(0, 0);
  }

  return loss;
}

template<typename PointSource, typename PointTarget>
template<typename PointT>
bool FastGICP<PointSource, PointTarget>::calculate_covariances(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>>& covariances) {
  kdtree.setInputCloud(cloud);
  covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_)
  for(int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), 50, k_indices, k_sq_distances);

    Eigen::Matrix<float, 4, 50> data;

    for(int j = 0; j < k_indices.size(); j++) {
      data.col(j) = cloud->at(k_indices[j]).getVector4fMap();
    }

    data.colwise() -= data.rowwise().mean().eval();
    Eigen::Matrix4f cov = data * data.transpose();

    if(regularization_method_ == FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse().cast<float>();
    } else {
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3f values;

      switch(regularization_method_) {
        case PLANE:
          values = Eigen::Vector3f(1, 1, 1e-3);
          break;
        case MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
      }

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }

  return true;
}

}  // namespace fast_gicp

#endif
