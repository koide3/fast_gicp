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
#include <fast_gicp/opt/gauss_newton.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

namespace fast_gicp {

template<typename PointSource, typename PointTarget>
FastVGICP<PointSource, PointTarget>::FastVGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  reg_name_ = "FastVGICP";
  max_iterations_ = 64;
  k_correspondences_ = 20;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;
  // corr_dist_threshold_ = 1.0;
  regularization_method_ = PLANE;
  corr_dist_threshold_ = std::numeric_limits<float>::max();

  voxel_resolution_ = 1.0;
  search_method_ = DIRECT1;
  voxel_mode_ = ADDITIVE;

  source_kdtree.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree.reset(new pcl::search::KdTree<PointTarget>);
}

template<typename PointSource, typename PointTarget>
FastVGICP<PointSource, PointTarget>::~FastVGICP() {}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setNumThreads(int n) {
  num_threads_ = n;

#ifdef _OPENMP
  if(n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setResolution(double resolution) {
  voxel_resolution_ = resolution;
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
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
void FastVGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if(input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  calculate_covariances(cloud, *source_kdtree, source_covs);
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if(target_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  calculate_covariances(cloud, *target_kdtree, target_covs);
  create_voxelmap(cloud);
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::create_voxelmap(const PointCloudTargetConstPtr& cloud) {
  voxels.clear();
  for(int i = 0; i < cloud->size(); i++) {
    Eigen::Vector3i coord = voxel_coord(cloud->at(i).getVector4fMap());

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
    voxel->append(cloud->at(i).getVector4fMap(), target_covs[i]);
  }

  for(auto& voxel : voxels) {
    voxel.second->finalize();
  }
}

template<typename PointSource, typename PointTarget>
void FastVGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Matrix<float, 6, 1> x0;
  x0.head<3>() = Sophus::SO3f(guess.template block<3, 3>(0, 0)).log();
  x0.tail<3>() = guess.template block<3, 1>(0, 3);

  // prevent stacking at zero
  if(x0.head<3>().norm() < 1e-2) {
    x0.head<3>() = (Eigen::Vector3f::Random()).normalized() * 1e-2;
  }

  converged_ = false;
  GaussNewton<double, 6> solver;

  for(int i = 0; i < max_iterations_; i++) {
    nr_iterations_ = i;

    Eigen::MatrixXf J;
    Eigen::VectorXf loss = loss_ls(x0, &J);

    Eigen::Matrix<float, 6, 1> delta = solver.delta(loss.cast<double>(), J.cast<double>()).cast<float>();

    Eigen::Isometry3f x0_ = Eigen::Isometry3f::Identity();
    x0_.linear() = Sophus::SO3f::exp(x0.head<3>()).matrix();
    x0_.translation() = x0.tail<3>();

    Eigen::Isometry3f delta_ = Eigen::Isometry3f::Identity();
    delta_.linear() = Sophus::SO3f::exp(delta.head<3>()).matrix();
    delta_.translation() = delta.tail<3>();

    Eigen::Isometry3f x1_ = delta_.inverse() * x0_;

    x0.head<3>() = Sophus::SO3f(x1_.linear()).log();
    x0.tail<3>() = x1_.translation();

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
Eigen::Vector3i FastVGICP<PointSource, PointTarget>::voxel_coord(const Eigen::Vector4f& x) const {
  return (x.array() / voxel_resolution_ - 0.5).floor().template cast<int>().template head<3>();
}

template<typename PointSource, typename PointTarget>
Eigen::Vector4f FastVGICP<PointSource, PointTarget>::voxel_origin(const Eigen::Vector3i& coord) const {
  Eigen::Vector3f origin = (coord.cast<float>().array() + 0.5) * voxel_resolution_;
  return Eigen::Vector4f(origin[0], origin[1], origin[2], 1.0f);
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
bool FastVGICP<PointSource, PointTarget>::is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
  Eigen::Matrix3f R = Sophus::SO3f::exp(delta.head<3>()).matrix() - Eigen::Matrix3f::Identity();
  Eigen::Vector3f t = delta.tail<3>();

  Eigen::Matrix3f r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3f t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template<typename PointSource, typename PointTarget>
Eigen::VectorXf FastVGICP<PointSource, PointTarget>::loss_ls(const Eigen::Matrix<float, 6, 1>& x, Eigen::MatrixXf* J) const {
  Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
  trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(x.head<3>()).matrix();
  trans.block<3, 1>(0, 3) = x.tail<3>();

  auto offsets = neighbor_offsets();

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> losses(input_->size() * offsets.size());
  // use row-major arrangement for ease of repacking
  std::vector<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>>> Js(input_->size() * offsets.size());

  std::atomic_int count(0);

#pragma omp parallel for num_threads(num_threads_)
  for(int i = 0; i < input_->size(); i++) {
    const auto& mean_A = input_->at(i).getVector4fMap();
    const auto& cov_A = source_covs[i];

    Eigen::Vector4f transed_mean_A = trans * mean_A;
    Eigen::Vector3i coord = voxel_coord(transed_mean_A);

    bool is_RCR_computed = false;
    Eigen::Matrix4f RCR;
    Eigen::Matrix4f skew_mean_A;

    for(const auto& offset : offsets) {
      auto voxel = lookup_voxel(coord + offset);

      if(voxel == nullptr) {
        continue;
      }

      if(!is_RCR_computed) {
        RCR = trans * cov_A * trans.transpose();
        RCR(3, 3) = 1;
        skew_mean_A.setZero();
        skew_mean_A.block<3, 3>(0, 0) = skew(transed_mean_A.head<3>());
        is_RCR_computed = true;
      }

      const auto& mean_B = voxel->mean;
      const auto& cov_B = voxel->cov;

      Eigen::Vector4f d = mean_B - transed_mean_A;
      Eigen::Matrix4f RCR_inv = (cov_B + RCR).inverse();

      int n = count++;
      losses[n] = (RCR_inv * d).head<3>();
      Js[n].block<3, 3>(0, 0) = (RCR_inv * skew_mean_A).block<3, 3>(0, 0);
      Js[n].block<3, 3>(0, 3) = -RCR_inv.block<3, 3>(0, 0);

      if(voxel_mode_ == ADDITIVE_WEIGHTED) {
        losses[n] *= voxel->num_points;
      }
    }
  }

  int final_size = count;
  *J = Eigen::Map<Eigen::MatrixXf>(Js.front().data(), 6, final_size * 3).transpose();

  return Eigen::Map<Eigen::VectorXf>(losses.front().data(), final_size * 3);
}

template<typename PointSource, typename PointTarget>
template<typename PointT>
bool FastVGICP<PointSource, PointTarget>::calculate_covariances(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Matrix4, Eigen::aligned_allocator<Matrix4>>& covariances) {
  kdtree.setInputCloud(cloud);
  covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_)
  for(int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    Eigen::Matrix<float, 4, -1> data(4, k_correspondences_);

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
