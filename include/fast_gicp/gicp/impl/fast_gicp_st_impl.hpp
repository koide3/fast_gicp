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
  regularization_method_ = PLANE;

  corr_dist_threshold_ = std::numeric_limits<float>::max();

  source_kdtree.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree.reset(new pcl::search::KdTree<PointTarget>);

  lm_max_iterations_ = 10;
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;
}

template<typename PointSource, typename PointTarget>
FastGICPSingleThread<PointSource, PointTarget>::~FastGICPSingleThread() {}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_kdtree.swap(target_kdtree);
  source_covs.swap(target_covs);

  correspondences.clear();
  sq_distances.clear();
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::clearSource() {
  input_.reset();
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::clearTarget() {
  target_.reset();
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if(input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  calculate_covariances(cloud, *source_kdtree, source_covs);
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if(target_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  calculate_covariances(cloud, *target_kdtree, target_covs);
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Isometry3d x0(guess.template cast<double>().matrix());

  anchors.clear();

  lm_lambda_ = -1.0;
  converged_ = false;

  for(int i = 0; i < max_iterations_ && !converged_; i++) {
    nr_iterations_ = i;

    update_correspondences(x0);
    update_mahalanobis(x0);

    Eigen::Isometry3d delta;
    if(!lm_step(x0, delta)) {
      std::cerr << "lm not converged!!" << std::endl;
      break;
    }

    converged_ = is_converged(delta);
  }

  final_transformation_ = x0.cast<float>().matrix();
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

template<typename PointSource, typename PointTarget>
bool FastGICPSingleThread<PointSource, PointTarget>::lm_step(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double y0 = compute_error(x0, &H, &b);

  if(lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * H.diagonal().array().abs().maxCoeff();
  }

  double nu = 2.0;
  for(int i = 0; i < lm_max_iterations_; i++) {
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H + lm_lambda_ * Eigen::Matrix<double, 6, 6>::Identity());
    Eigen::Matrix<double, 6, 1> d = solver.solve(-b);

    delta.setIdentity();
    delta.linear() = Sophus::SO3d::exp(d.head<3>()).matrix();
    delta.translation() = d.tail<3>();

    Eigen::Isometry3d xi = delta * x0;
    double yi = compute_error(xi);
    double rho = (y0 - yi) / (d.dot(lm_lambda_ * d - b));

    if(rho < 0) {
      lm_lambda_ = nu * lm_lambda_;
      nu = 2 * nu;
      continue;
    }

    x0 = xi;
    lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    return true;
  }

  return false;
}

template<typename PointSource, typename PointTarget>
bool FastGICPSingleThread<PointSource, PointTarget>::is_converged(const Eigen::Isometry3d& delta) const {
  double accum = 0.0;
  Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& x) {
  Eigen::Isometry3f trans = x.template cast<float>();

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

    target_kdtree->nearestKSearch(pt, 2, k_indices, k_sq_dists);

    correspondences[i] = k_indices[0];
    sq_distances[i] = k_sq_dists[0];
    second_sq_distances[i] = k_sq_dists[1];
    anchors[i] = pt.getVector4fMap();
  }
}

template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::update_mahalanobis(const Eigen::Isometry3d& trans) {
  assert(source_covs.size() == input_->size());
  assert(target_covs.size() == target_->size());
  assert(correspondences.size() == input_->size());

  Eigen::Matrix4d trans_matrix = trans.matrix();
  mahalanobis.resize(input_->size());

  for(int i = 0; i < input_->size(); i++) {
    int target_index = correspondences[i];
    if(target_index < 0) {
      continue;
    }

    const auto& cov_A = source_covs[i];
    const auto& cov_B = target_covs[target_index];

    Eigen::Matrix4d RCR = cov_B + trans_matrix * cov_A * trans_matrix.transpose();
    RCR(3, 3) = 1.0;

    mahalanobis[i] = RCR.inverse();
  }
}

template<typename PointSource, typename PointTarget>
double FastGICPSingleThread<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) const {
  double sum_errors = 0.0;

  if(H && b) {
    H->setZero();
    b->setZero();
  }

  for(int i = 0; i < input_->size(); i++) {
    int target_index = correspondences[i];
    if(target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mahalanobis[i] * (mean_B - transed_mean_A);

    sum_errors += error.squaredNorm();

    if(H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = mahalanobis[i] * dtdx0;

    (*H) += jlossexp.transpose() * jlossexp;
    (*b) += jlossexp.transpose() * error;
  }

  return sum_errors;
}

template<typename PointSource, typename PointTarget>
template<typename PointT>
bool FastGICPSingleThread<PointSource, PointTarget>::calculate_covariances(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances) {
  kdtree.setInputCloud(cloud);
  covariances.resize(cloud->size());

  std::vector<int> k_indices;
  std::vector<float> k_sq_distances;
  Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences_);
  Eigen::JacobiSVD<Eigen::Matrix3d> svd;

  for(int i = 0; i < cloud->size(); i++) {
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    for(int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;

    if(regularization_method_ == NONE) {
      covariances[i] = cov;
    } else if(regularization_method_ == FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;

      switch(regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
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
