#ifndef FAST_GICP_FAST_VGICP_IMPL_HPP
#define FAST_GICP_FAST_VGICP_IMPL_HPP

#include <atomic>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <sophus/so3.hpp>
#include <fast_gicp/so3/so3.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

namespace fast_gicp {

template<typename PointSource, typename PointTarget>
FastVGICP<PointSource, PointTarget>::FastVGICP() : FastGICP<PointSource, PointTarget>() {
  this->reg_name_ = "FastVGICP";

  voxel_resolution_ = 1.0;
  search_method_ = DIRECT1;
  voxel_mode_ = ADDITIVE;
}

template<typename PointSource, typename PointTarget>
FastVGICP<PointSource, PointTarget>::~FastVGICP() {}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setResolution(double resolution) {
  voxel_resolution_ = resolution;
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setNeighborSearchMethod(NeighborSearchMethod method) {
  search_method_ = method;
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setVoxelAccumulationMode(VoxelAccumulationMode mode) {
  voxel_mode_ = mode;
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_kdtree.swap(target_kdtree);
  source_covs.swap(target_covs);

  if(target_) {
    create_voxelmap(target_);
  }
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if(target_ == cloud) {
    return;
  }

  FastGICP<PointSource, PointTarget>::setInputTarget(cloud);
  create_voxelmap(cloud);
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::create_voxelmap(const PointCloudTargetConstPtr& cloud) {
  voxels.clear();
  for(int i = 0; i < cloud->size(); i++) {
    Eigen::Vector3i coord = voxel_coord(cloud->at(i).getVector4fMap().template cast<double>());

    auto found = voxels.find(coord);
    if(found == voxels.end()) {
      GaussianVoxel::Ptr voxel;
      switch(voxel_mode_) {
        case ADDITIVE:
        case ADDITIVE_WEIGHTED:
          voxel = std::make_shared<AdditiveGaussianVoxel>();
          break;
        case MULTIPLICATIVE:
          voxel = std::make_shared<MultiplicativeGaussianVoxel>();
          break;
      }
      found = voxels.insert(found, std::make_pair(coord, voxel));
    }

    auto& voxel = found->second;
    voxel->append(cloud->at(i).getVector4fMap().template cast<double>(), target_covs[i]);
  }

  for(auto& voxel : voxels) {
    voxel.second->finalize();
  }
}

template<typename PointSource, typename PointTarget>
std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> FastVGICP<PointSource, PointTarget>::neighbor_offsets() const {
  switch(search_method_) {
    // clang-format off
    default:
      std::cerr << "here must not be reached" << std::endl;
      abort();
    case DIRECT1:
      return std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>{
        Eigen::Vector3i(0, 0, 0)
      };
    case DIRECT7:
      return std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>{
        Eigen::Vector3i(0, 0, 0),
        Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(-1, 0, 0),
        Eigen::Vector3i(0, 1, 0),
        Eigen::Vector3i(0, -1, 0),
        Eigen::Vector3i(0, 0, 1),
        Eigen::Vector3i(0, 0, -1)
      };
    // clang-format on
  }

  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> offsets27;
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      for(int k = 0; k < 3; k++) {
        offsets27.push_back(Eigen::Vector3i(i, j, k));
      }
    }
  }
  return offsets27;
}

template<typename PointSource, typename PointTarget>
Eigen::Vector3i FastVGICP<PointSource, PointTarget>::voxel_coord(const Eigen::Vector4d& x) const {
  return (x.array() / voxel_resolution_ - 0.5).floor().template cast<int>().template head<3>();
}

template<typename PointSource, typename PointTarget>
Eigen::Vector4d FastVGICP<PointSource, PointTarget>::voxel_origin(const Eigen::Vector3i& coord) const {
  Eigen::Vector3d origin = (coord.cast<double>().array() + 0.5) * voxel_resolution_;
  return Eigen::Vector4d(origin[0], origin[1], origin[2], 1.0f);
}

template<typename PointSource, typename PointTarget>
GaussianVoxel::Ptr FastVGICP<PointSource, PointTarget>::lookup_voxel(const Eigen::Vector3i& x) const {
  auto found = voxels.find(x);
  if(found == voxels.end()) {
    return nullptr;
  }

  return found->second;
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Isometry3d x0 = Eigen::Isometry3d(guess.template cast<double>());

  this->lm_lambda_ = -1.0;
  this->converged_ = false;

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> voxel_means;
  for(const auto& voxel : voxels) {
    voxel_means.push_back(voxel.second->mean.cast<float>().head<3>());
  }

  for(int i = 0; i < this->max_iterations_ && !this->converged_; i++) {
    this->nr_iterations_ = i;

    update_correspondences(x0);
    update_mahalanobis(x0);

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> corrs;
    for(const auto& corr: voxel_correspondences) {
      corrs.push_back(x0.cast<float>() * input_->at(corr.first).getVector3fMap());
      corrs.push_back(corr.second->mean.head<3>().cast<float>());
    }

    Eigen::Isometry3d delta;
    if(!this->lm_step(x0, delta)) {
      std::cerr << "lm not converged!!" << std::endl;
      break;
    }

    this->converged_ = this->is_converged(delta);
  }

  this->final_transformation_ = x0.cast<float>().matrix();
  pcl::transformPointCloud(*input_, output, this->final_transformation_);
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  voxel_correspondences.clear();
  auto offsets = neighbor_offsets();

  std::vector<std::vector<std::pair<int, GaussianVoxel::Ptr>>> corrs(num_threads_);
  for(auto& c: corrs) {
    c.reserve((input_->size() * offsets.size()) / num_threads_);
  }

#pragma omp parallel for num_threads(num_threads_)
  for(int i = 0; i < input_->size(); i++) {
    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    Eigen::Vector4d transed_mean_A = trans * mean_A;
    Eigen::Vector3i coord = voxel_coord(transed_mean_A);

    for(const auto& offset : offsets) {
      auto voxel = lookup_voxel(coord + offset);
      if(voxel != nullptr) {
        corrs[omp_get_thread_num()].push_back(std::make_pair(i, voxel));
      }
    }
  }

  voxel_correspondences.reserve(input_->size() * offsets.size());
  for(const auto& c : corrs) {
    voxel_correspondences.insert(voxel_correspondences.end(), c.begin(), c.end());
  }
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::update_mahalanobis(const Eigen::Isometry3d& trans) {
  assert(source_covs.size() == input_->size());

  Eigen::Matrix4d trans_matrix = trans.matrix();
  voxel_mahalanobis.resize(voxel_correspondences.size());

#pragma omp parallel for num_threads(num_threads_)
  for(int i = 0; i < voxel_correspondences.size(); i++) {
    const auto& corr = voxel_correspondences[i];
    const auto& cov_A = source_covs[corr.first];
    const auto& cov_B = corr.second->cov;

    Eigen::Matrix4d RCR = cov_B + trans_matrix * cov_A * trans_matrix.transpose();
    RCR(3, 3) = 1.0;

    voxel_mahalanobis[i] = RCR.inverse();
  }
}

template<typename PointSource, typename PointTarget>
double FastVGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) const {
  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for(int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors)
  for(int i = 0; i < voxel_correspondences.size(); i++) {
    const auto& corr = voxel_correspondences[i];
    auto target_voxel = corr.second;

    const Eigen::Vector4d mean_A = input_->at(corr.first).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs[corr.first];

    const Eigen::Vector4d mean_B = corr.second->mean;
    const auto& cov_B = corr.second->cov;

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = std::sqrt(target_voxel->num_points) * voxel_mahalanobis[i] * (mean_B - transed_mean_A);

    const double w = voxel_mode_ == VoxelAccumulationMode::ADDITIVE_WEIGHTED ? std::sqrt(target_voxel->num_points) : 1.0;

    sum_errors += error.head<3>().squaredNorm();

    if(H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = std::sqrt(target_voxel->num_points) * voxel_mahalanobis[i] * dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * error;

    int thread_num = omp_get_thread_num();
    Hs[thread_num] += Hi;
    bs[thread_num] += bi;
  }

  if(H && b) {
    H->setZero();
    b->setZero();
    for(int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

}  // namespace fast_gicp

#endif
