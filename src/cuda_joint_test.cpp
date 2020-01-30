#include <chrono>
#include <fstream>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/cuda/fast_vgicp_cuda.cuh>

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

int main(int argc, char** argv) {
  auto source_cloud = load_cloud_txt("/home/koide/catkin_ws/src/fast_gicp/data/source_cloud.txt");
  auto target_cloud = load_cloud_txt("/home/koide/catkin_ws/src/fast_gicp/data/target_cloud.txt");

  /*
  fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp;
  vgicp.setRegularizationMethod(fast_gicp::FROBENIUS);
  vgicp.setResolution(1.0);
  vgicp.setCorrespondenceRandomness(50);
  vgicp.setInputSource(source_cloud);
  vgicp.setInputTarget(target_cloud);
  */


  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> source_points(source_cloud->size());
  std::transform(source_cloud->begin(), source_cloud->end(), source_points.begin(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap(); });

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> target_points(source_cloud->size());
  std::transform(target_cloud->begin(), target_cloud->end(), target_points.begin(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap(); });

  std::unique_ptr<fast_gicp::FastVGICPCudaCore> vgicp_core(new fast_gicp::FastVGICPCudaCore());

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "set clouds" << std::endl;
  vgicp_core->set_source_cloud(source_points);
  vgicp_core->set_target_cloud(target_points);

  std::cout << "find neighbors" << std::endl;
  vgicp_core->find_source_neighbors(20);
  vgicp_core->find_target_neighbors(20);

  std::cout << "calc covariances" << std::endl;
  vgicp_core->calculate_source_covariances();
  vgicp_core->calculate_target_covariances();

  std::cout << "create voxelmap" << std::endl;
  vgicp_core->create_target_voxelmap();

  std::cout << "test" << std::endl;
  vgicp_core->test_print();
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;

  return 0;
}