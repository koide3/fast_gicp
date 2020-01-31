#include <chrono>
#include <fstream>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>

pcl::PointCloud<pcl::PointXYZ>::Ptr load_cloud_txt(const std::string& filename) {
  std::ifstream ifs(filename);
  if(!ifs) {
    std::cerr << "failed to open " << filename << std::endl;
    abort();
  }

  int num_points;
  ifs >> num_points;

  std::cout << "# points:" << num_points << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  cloud->resize(num_points);
  for(int i = 0; i < num_points; i++) {
    ifs >> cloud->at(i).x >> cloud->at(i).y >> cloud->at(i).z;
  }

  return cloud;
}

std::vector<int> find_neighbors(int k, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud) {
  pcl::search::KdTree<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud);

  std::vector<int> neighbors(cloud->size() * k);

#pragma omp parallel for
  for(int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k, k_indices, k_sq_distances);

    std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
  }

  return neighbors;
}

int main(int argc, char** argv) {
  auto source_cloud = load_cloud_txt("/home/koide/catkin_ws/src/fast_gicp/data/source_cloud.txt");
  auto target_cloud = load_cloud_txt("/home/koide/catkin_ws/src/fast_gicp/data/target_cloud.txt");

  fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>::Ptr vgicp(new fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>());
  vgicp->setResolution(1.0);
  vgicp->setCorrespondenceRandomness(20);

  for(int i = 0; i < 32; i++) {
    auto t1_ = std::chrono::high_resolution_clock::now();
    vgicp->setInputSource(source_cloud);
    vgicp->setInputTarget(target_cloud);
    auto t2_ = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    vgicp->align(*aligned);
    auto t3_ = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t2_ - t1_).count() / 1e6 << "[msec]" << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t3_ - t2_).count() / 1e6 << "[msec]" << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t3_ - t1_).count() / 1e6 << "[msec]" << std::endl;

    std::cout << "--- estimated ---" << std::endl << vgicp->getFinalTransformation() << std::endl;
  }

  return 0;
}