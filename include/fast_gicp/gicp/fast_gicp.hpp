#ifndef FAST_GICP_FAST_GICP_HPP
#define FAST_GICP_FAST_GICP_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>

#include <sophus/so3.hpp>

#include <kkl/opt/solvers/bfgs.hpp>
#include <kkl/opt/solvers/nelder_mead.hpp>

#include <fast_gicp/gicp/gicp_loss.hpp>
#include <fast_gicp/so3/so3_derivatives.hpp>

namespace gicp {

class FastGeneralizedIterativeClosestPoint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FastGeneralizedIterativeClosestPoint() {}
  virtual ~FastGeneralizedIterativeClosestPoint() {}

  void set_input_target(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) {
    tgt_cloud = cloud;
    calculate_covariances(cloud, tgt_kdtree, tgt_covs);
  }

  void set_input_source(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) {
    src_cloud = cloud;
    calculate_covariances(cloud, src_kdtree, src_covs);
  }

  Eigen::Matrix4d align() {
    Eigen::Matrix4d initial_guess = Eigen::Matrix4d::Identity();
    return align(initial_guess);
  }

  Eigen::Matrix4d align(Eigen::Matrix4d& initial_guess) {
    Eigen::Matrix<double, 6, 1> x0;
    x0.head<3>() = Sophus::SO3d(initial_guess.block<3, 3>(0, 0)).log();
    x0.tail<3>() = initial_guess.block<3, 1>(0, 3);

    auto f = [this](const Eigen::Matrix<double, 6, 1>& x) { return loss(x); };
    auto j = [this](const Eigen::Matrix<double, 6, 1>& x) {
      Eigen::Matrix<double, 1, 6> J;
      loss(x, &J);
      return J;
    };
    auto before_optimization = [this](const Eigen::Matrix<double, 6, 1>& x) {
      std::cout << "update correspondences" << std::endl;
      update_correspondences(x);
    };
    auto callback = [this](const Eigen::Matrix<double, 6, 1>& x) { std::cout << loss(x) << ":" << x.transpose() << std::endl; };

    kkl::opt::BFGS<double, 6> solver(f, j);
    solver.set_callback(callback);
    solver.set_before_optimization_callback(before_optimization);

    update_correspondences(x0);
    auto result = solver.optimize(x0, kkl::opt::TerminationCriteria(32, 1e-8));

    Eigen::Matrix4d estimated = Eigen::Matrix4d::Identity();
    estimated.block<3, 3>(0, 0) = Sophus::SO3d::exp(result.x.head<3>()).matrix();
    estimated.block<3, 1>(0, 3) = result.x.tail<3>();

    return estimated;
  }

private:
  void update_correspondences(const Eigen::Matrix<double, 6, 1>& x) {
    Eigen::Matrix3d R = Sophus::SO3d::exp(x.head<3>()).matrix();
    Eigen::Vector3d t = x.tail<3>();

    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3, 3>(0, 0) = R.cast<float>();
    trans.block<3, 1>(0, 3) = t.cast<float>();

    correspondences.resize(src_cloud->size());
    sq_distances.resize(src_cloud->size());

    for(int i = 0; i < src_cloud->size(); i++) {
      pcl::PointXYZI pt;
      pt.getVector4fMap() = trans * src_cloud->at(i).getVector4fMap();

      std::vector<int> k_indices;
      std::vector<float> k_sq_dists;
      tgt_kdtree.nearestKSearch(pt, 1, k_indices, k_sq_dists);

      correspondences[i] = k_indices[0];
      sq_distances[i] = k_sq_dists[0];
    }
  }

  double loss(const Eigen::Matrix<double, 6, 1>& x, Eigen::Matrix<double, 1, 6>* J = nullptr) const {
    Eigen::Matrix3d R = Sophus::SO3d::exp(x.head<3>()).matrix();
    Eigen::Vector3d t = x.tail<3>();

    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3, 3>(0, 0) = R.cast<float>();
    trans.block<3, 1>(0, 3) = t.cast<float>();

    double loss = 0.0;

    Eigen::Matrix<double, 1, 12> sum_jloss = Eigen::Matrix<double, 1, 12>::Zero();

    for(int i = 0; i < src_cloud->size(); i++) {
      int tgt_index = correspondences[i];
      float sq_dist = sq_distances[i];

      if(sq_dist > 1.0) {
        continue;
      }

      Eigen::Vector3d mean_A = src_cloud->at(i).getVector3fMap().cast<double>();
      Eigen::Matrix3d cov_A = src_covs[i];

      Eigen::Vector3d mean_B = tgt_cloud->at(tgt_index).getVector3fMap().cast<double>();
      Eigen::Matrix3d cov_B = tgt_covs[tgt_index];

      if(J) {
        Eigen::Matrix<double, 1, 12> jloss = Eigen::Matrix<double, 1, 12>::Zero();
        loss += gicp_loss(mean_A, cov_A, mean_B, cov_B, R, t, &jloss);
        sum_jloss += jloss;
      } else {
        loss += gicp_loss(mean_A, cov_A, mean_B, cov_B, R, t);
      }
    }

    if(J) {
      Eigen::Matrix<double, 12, 6> jexp = Eigen::Matrix<double, 12, 6>::Zero();
      jexp.block<9, 3>(0, 0) = dso3_exp(x.head<3>());
      jexp.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();

      (*J) = sum_jloss * jexp;
    }

    return loss;
  }

  bool calculate_covariances(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, pcl::search::KdTree<pcl::PointXYZI>& kdtree, std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& covariances) {
    kdtree.setInputCloud(cloud);
    covariances.resize(cloud->size());

    for(int i = 0; i < cloud->size(); i++) {
      std::vector<int> k_indices;
      std::vector<float> k_sq_distances;
      kdtree.nearestKSearch(cloud->at(i), 20, k_indices, k_sq_distances);

      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      auto& cov = covariances[i];
      cov.setZero();

      Eigen::Matrix<double, 3, 20> data;

      for(int j = 0; j < k_indices.size(); j++) {
        const auto& pt = cloud->at(k_indices[j]);
        data.col(j) = pt.getVector3fMap().cast<double>();

        // mean += pt.getVector3fMap().cast<double>();
        // cov += (pt.getVector3fMap() * pt.getVector3fMap().transpose()).cast<double>();
      }

      // mean /= k_indices.size();
      // cov /= k_indices.size();
      // cov -= mean * mean.transpose();

      data.colwise() -= data.rowwise().mean();
      cov = data * data.transpose();

      /*
      double lambda = 1e-6;
      Eigen::Matrix3d inv_C = (cov + lambda * Eigen::Matrix3d::Identity()).inverse();
      cov = (inv_C / inv_C.norm()).inverse();
      */

      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
      // Eigen::Vector3d values = svd.singularValues();
      Eigen::Vector3d values(1, 1, 1e-2);
      // Eigen::Vector3d values = svd.singularValues().array().max(1e-2);
      // Eigen::Vector3d values = svd.singularValues().normalized().array().max(1e-2);

      cov = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }

    return true;
  }

private:
  pcl::PointCloud<pcl::PointXYZI>::ConstPtr tgt_cloud;
  pcl::PointCloud<pcl::PointXYZI>::ConstPtr src_cloud;

  pcl::search::KdTree<pcl::PointXYZI> tgt_kdtree;
  pcl::search::KdTree<pcl::PointXYZI> src_kdtree;

  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> tgt_covs;
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> src_covs;

  std::vector<int> correspondences;
  std::vector<float> sq_distances;
};

}  // namespace gicp

#endif