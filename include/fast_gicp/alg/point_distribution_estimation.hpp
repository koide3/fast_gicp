#ifndef FAST_GICP_POINT_DISTRIBUTION_ESTIMATION_HPP
#define FAST_GICP_POINT_DISTRIBUTION_ESTIMATION_HPP

#include <random>
#include <pcl/search/kdtree.h>
#include <fast_gicp/gicp/gicp_settings.hpp>
#include <fast_gicp/alg/scan_line_interval_estimation.hpp>

namespace fast_gicp {

template<typename PointT>
class PointDistributionEstimation {
public:
  PointDistributionEstimation(RegularizationMethod regularization)
  : regularization(regularization)
  {}
  virtual ~PointDistributionEstimation() {}

  virtual void estimate(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud) = 0;

protected:
  Eigen::Matrix4f regularize(Eigen::Matrix4f& cov) const {
    if(regularization == FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();

      Eigen::Matrix4f covariance = Eigen::Matrix4f::Zero();
      covariance.block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse().cast<float>();
      return covariance;
    }

      Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3f values;

      switch(regularization) {
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

      Eigen::Matrix4f covariance = Eigen::Matrix4f::Zero();
      covariance.block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
      return covariance;
  }

public:
  RegularizationMethod regularization;

  pcl::search::KdTree<PointT> kdtree;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> covariances;
};

template<typename PointT>
class KNNDistributionEstimation : public PointDistributionEstimation<PointT> {
public:
  KNNDistributionEstimation(int k = 20, RegularizationMethod regularization = PLANE, int num_threads_ = 0) : PointDistributionEstimation<PointT>(regularization), k(k), num_threads(num_threads_) {
    if(num_threads == 0) {
      num_threads = omp_get_max_threads();
    }
  }

  virtual void estimate(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud) override {
    this->kdtree.setInputCloud(cloud);
    this->covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < cloud->size(); i++) {
      std::vector<int> k_indices;
      std::vector<float> k_sq_distances;
      this->kdtree.nearestKSearch(cloud->at(i), k, k_indices, k_sq_distances);

      Eigen::Matrix<float, 4, -1> data(4, k);

      for(int j = 0; j < k_indices.size(); j++) {
        data.col(j) = cloud->at(k_indices[j]).getVector4fMap();
      }

      data.colwise() -= data.rowwise().mean().eval();
      Eigen::Matrix4f cov = data * data.transpose();

      this->covariances[i] = this->regularize(cov);
    }
  }

private:
  int k;
  int num_threads;
};

template<typename PointT>
class RadiusDistributionEstimation : public PointDistributionEstimation<PointT> {
public:
  RadiusDistributionEstimation(int min_k = 10, int max_k=30, RegularizationMethod regularization = PLANE, int num_threads_ = 0) : PointDistributionEstimation<PointT>(regularization), min_k(min_k), max_k(max_k), num_threads(num_threads_) {
    min_radius = 0.15;
    radius_factor = 5.0;

    if(num_threads == 0) {
      num_threads = omp_get_max_threads();
    }
  }

  virtual void estimate(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud) override {
    if(!scan_line_interval) {
      scan_line_interval.reset(new ScanLineIntervalEstimation<PointT>());
      scan_line_interval->estimate(cloud);
    }

    this->kdtree.setInputCloud(cloud);
    this->covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < cloud->size(); i++) {
      std::vector<int> k_indices;
      std::vector<float> k_sq_distances;

      double theta = scan_line_interval->line_interval();
      double dist = cloud->at(i).getVector3fMap().norm();
      double search_radius = std::max(dist * std::tan(theta) * radius_factor, min_radius);

      this->kdtree.radiusSearch(cloud->at(i), search_radius, k_indices, k_sq_distances);

      if(k_indices.size() > max_k) {
        std::shuffle(k_indices.begin(), k_indices.end(), mt);
        k_indices.erase(k_indices.begin() + max_k, k_indices.end());
      } else if(k_indices.size() < min_k) {
        this->kdtree.nearestKSearch(cloud->at(i), min_k, k_indices, k_sq_distances);
      }

      Eigen::Matrix<float, 4, -1> data(4, k_indices.size());

      for(int j = 0; j < k_indices.size(); j++) {
        data.col(j) = cloud->at(k_indices[j]).getVector4fMap();
      }

      data.colwise() -= data.rowwise().mean().eval();
      Eigen::Matrix4f cov = data * data.transpose();

      this->covariances[i] = this->regularize(cov);
    }
  }

private:
  std::mt19937 mt;

  int min_k;
  int max_k;
  int num_threads;

  double min_radius;
  double radius_factor;

  std::unique_ptr<ScanLineIntervalEstimation<PointT>> scan_line_interval;
};
}

#endif