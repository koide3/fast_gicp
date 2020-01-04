#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <fast_gicp/gicp/fast_gicp.hpp>

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

  gicp::FastGeneralizedIterativeClosestPoint icp;
  icp.set_input_target(tgt_cloud);
  icp.set_input_source(src_cloud);

  Eigen::Matrix4d estimated = icp.align();

  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::transformPointCloud(*src_cloud, *aligned, estimated.cast<float>());

  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned2(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp2;
  gicp2.setInputTarget(tgt_cloud);
  gicp2.setInputSource(src_cloud);
  gicp2.align(*aligned2);

  std::cout << "--- matrix1 ---" << std::endl << estimated << std::endl;
  std::cout << "--- matrix2 ---" << std::endl << gicp2.getFinalTransformation() << std::endl;

  pcl::visualization::PCLVisualizer vis;

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> tgt_handler(tgt_cloud, 255, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> src_handler(src_cloud, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> aligned_handler(aligned, 0, 0, 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> aligned2_handler(aligned, 0, 255, 255);

  vis.addPointCloud(tgt_cloud, tgt_handler, "tgt");
  vis.addPointCloud(src_cloud, src_handler, "src");
  vis.addPointCloud(aligned, aligned_handler, "aligned");
  vis.addPointCloud(aligned2, aligned2_handler, "aligned2");
  vis.spin();

  return 0;
}