#include <chrono>
#include <iostream>

#include <sophus/so3.hpp>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

void test(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& src_cloud, const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& tgt_cloud, pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>::Ptr reg) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());

  auto t1 = std::chrono::high_resolution_clock::now();
  reg->setInputSource(src_cloud);
  reg->setInputTarget(tgt_cloud);
  reg->align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();

  auto t3 = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 32; i++) {
    reg->setInputSource(src_cloud);
    reg->setInputTarget(tgt_cloud);
    reg->align(*aligned);
  }
  auto t4 = std::chrono::high_resolution_clock::now();

  std::cout << "--- " << reg->getClassName() << " ---" << std::endl;
  double single_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  double tenth_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e6;

  std::cout << "fitness_score:" << reg->getFitnessScore() << " single:" << single_time << "[msec] 10times:" << tenth_time << "[msec]" << std::endl;
}

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZI>());

  pcl::io::loadPCDFile("/home/koide/catkin_ws/src/fast_gicp/data/251370668.pcd", *tgt_cloud);
  pcl::io::loadPCDFile("/home/koide/catkin_ws/src/fast_gicp/data/251371071.pcd", *src_cloud);

  pcl::VoxelGrid<pcl::PointXYZI> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
  voxelgrid.setInputCloud(tgt_cloud);
  voxelgrid.filter(*filtered);
  tgt_cloud = filtered;

  filtered.reset(new pcl::PointCloud<pcl::PointXYZI>());
  voxelgrid.setInputCloud(src_cloud);
  voxelgrid.filter(*filtered);
  src_cloud = filtered;

  auto ndt = boost::make_shared<pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>>();
  ndt->setResolution(0.5);

  test(src_cloud, tgt_cloud, ndt);
  test(src_cloud, tgt_cloud, boost::make_shared<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>>());
  test(src_cloud, tgt_cloud, boost::make_shared<fast_gicp::FastGICPSingleThread<pcl::PointXYZI, pcl::PointXYZI>>());
  test(src_cloud, tgt_cloud, boost::make_shared<fast_gicp::FastGICP<pcl::PointXYZI, pcl::PointXYZI>>());

  auto vgicp = boost::make_shared<fast_gicp::FastVGICP<pcl::PointXYZI, pcl::PointXYZI>>();
  vgicp->setNumThreads(1);
  vgicp->setNeighborSearchMethod(fast_gicp::DIRECT1);
  test(src_cloud, tgt_cloud, vgicp);
  vgicp->setNumThreads(0);
  test(src_cloud, tgt_cloud, vgicp);

  return 0;
}