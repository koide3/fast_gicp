#ifndef FAST_GICP_FAST_GICP_ST_IMPL_HPP
#define FAST_GICP_FAST_GICP_ST_IMPL_HPP

#include <fast_gicp/so3/so3.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>

namespace fast_gicp {

template<typename PointSource, typename PointTarget>
FastGICPSingleThread<PointSource, PointTarget>::FastGICPSingleThread()
: FastGICP<PointSource, PointTarget>()
{
  this->reg_name_ = "FastGICPSingleThread";
  this->num_threads_ = 1;
}

template<typename PointSource, typename PointTarget>
FastGICPSingleThread<PointSource, PointTarget>::~FastGICPSingleThread() {}


template<typename PointSource, typename PointTarget>
void FastGICPSingleThread<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  anchors.clear();

  FastGICP<PointSource, PointTarget>::computeTransformation(output, guess);
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
  if(H && b) {
    H->setZero();
    b->setZero();
  }

  double sum_errors = 0.0;
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

}  // namespace fast_gicp

#endif
