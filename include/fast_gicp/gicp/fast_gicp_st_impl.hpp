#ifndef FAST_GICP_FAST_GICP_ST_IMPL_HPP
#define FAST_GICP_FAST_GICP_ST_IMPL_HPP

#include <sophus/so3.hpp>

#include <fast_gicp/so3/so3.hpp>
#include <fast_gicp/opt/gauss_newton.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>

namespace fast_gicp {

template<typename PointSource, typename PointTarget>
FastGICPSingleThread<PointSource, PointTarget>::FastGICPSingleThread() {
  reg_name_ = "FastGICPSingleThread";
  max_iterations_ = 64;
  k_correspondences_ = 20;
  transformation_epsilon_ = 5e-4;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;
  // corr_dist_threshold_ = 1.0;
  regularization_method_ = PLANE;
  corr_dist_threshold_ = std::numeric_limits<float>::max();
}

template<typename PointSource, typename PointTarget>
FastGICPSingleThread<PointSource, PointTarget>::~FastGICPSingleThread() {}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  calculate_covariances(cloud, source_kdtree, source_covs);
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  calculate_covariances(cloud, target_kdtree, target_covs);
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  anchors.clear();

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
bool FastGICPSingleThread<PointSource, PointTarget>::is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
  double accum = 0.0;
  Eigen::Matrix3f R = Sophus::SO3f::exp(delta.head<3>()).matrix() - Eigen::Matrix3f::Identity();
  Eigen::Vector3f t = delta.tail<3>();

  Eigen::Matrix3f r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3f t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::update_correspondences(const Eigen::Matrix<float, 6, 1>& x) {
  Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
  trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(x.head<3>()).matrix();
  trans.block<3, 1>(0, 3) = x.tail<3>();

  bool is_first = anchors.empty();

  correspondences.resize(input_->size());
  sq_distances.resize(input_->size());
  second_sq_distances.resize(input_->size());
  anchors.resize(input_->size());

  std::vector<int> k_indices;
  std::vector<float> k_sq_dists;

  for(int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    pt.getVector4fMap() = trans * input_->at(i).getVector4fMap();

    if(!is_first) {
      double d = (pt.getVector4fMap() - anchors[i]).norm();
      double max_first = std::sqrt(sq_distances[i]) + d;
      double min_second = std::sqrt(second_sq_distances[i]) - d;

      if(max_first < min_second) {
        continue;
      }
    }

    target_kdtree.nearestKSearch(pt, 2, k_indices, k_sq_dists);

    correspondences[i] = k_indices[0];
    sq_distances[i] = k_sq_dists[0];
    second_sq_distances[i] = k_sq_dists[1];
    anchors[i] = pt.getVector4fMap();
  }
}

template<typename PointSource, typename PointTarget>
Eigen::VectorXf FastGICPSingleThread<PointSource, PointTarget>::loss_ls(const Eigen::Matrix<float, 6, 1>& x, Eigen::MatrixXf* J) const {
  Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
  trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(x.head<3>()).matrix();
  trans.block<3, 1>(0, 3) = x.tail<3>();

  int count = 0;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> losses(input_->size());
  // use row-major arrangement for ease of repacking
  std::vector<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>>> Js(input_->size());

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
    losses[count] = (RCR_inv * d).eval().head<3>();
    Js[count].block<3, 3>(0, 0) = RCR_inv.block<3, 3>(0, 0) * skew(transed_mean_A.head<3>());
    Js[count].block<3, 3>(0, 3) = -RCR_inv.block<3, 3>(0, 0);
    count++;
  }

  int final_size = count;
  Eigen::VectorXf loss = Eigen::Map<Eigen::VectorXf>(losses.front().data(), final_size * 3);
  *J = Eigen::Map<Eigen::MatrixXf>(Js.front().data(), 6, final_size * 3).transpose();
  return loss;
}

template<typename PointSource, typename PointTarget>
template<typename PointT>
bool FastGICPSingleThread<PointSource, PointTarget>::calculate_covariances(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>>& covariances) {
  kdtree.setInputCloud(cloud);
  covariances.resize(cloud->size());

  std::vector<int> k_indices;
  std::vector<float> k_sq_distances;
  Eigen::Matrix<float, 4, -1> data(4, k_correspondences_);
  Eigen::JacobiSVD<Eigen::Matrix3f> svd;

  for(int i = 0; i < cloud->size(); i++) {
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    for(int j = 0; j < k_indices.size(); j++) {
      data.col(j) = cloud->at(k_indices[j]).getVector4fMap();
    }

    data.colwise() -= data.rowwise().mean().eval();
    Eigen::Matrix4f cov = data * data.transpose();

    if(regularization_method_ == FROBENIUS) {
      double lambda = 1e-6;
      Eigen::Matrix3f C = cov.block<3, 3>(0, 0) + lambda * Eigen::Matrix3f::Identity();
      Eigen::Matrix3f C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3f values;

      switch(regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case PLANE:
          values = Eigen::Vector3f(1, 1, 1e-2);
          break;
        case MIN_EIG:
          values = svd.singularValues().array().max(1e-2);
          break;
        case NORMALIZED_MIN_EIG:
          values = svd.singularValues().normalized().array().max(1e-2);
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
