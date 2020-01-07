#include <chrono>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <fast_gicp/gicp/fast_gicp.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

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

  auto t1 = std::chrono::high_resolution_clock::now();
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp2;
  gicp2.setInputSource(src_cloud);
  gicp2.setInputTarget(tgt_cloud);

  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
  gicp2.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();

  Eigen::Matrix4f model_matrix = gicp2.getFinalTransformation();

  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("pclgicp", std::make_shared<glk::PointCloudBuffer>(aligned), guik::ShaderSetting().add("color_mode", 1).add("material_color", Eigen::Vector4f(0.0f, 0.0f, 1.0f, 1.0f)));

  auto t3 = std::chrono::high_resolution_clock::now();
  fast_gicp::FastGICP<pcl::PointXYZI, pcl::PointXYZI> gicp;
  gicp.setInputSource(src_cloud);
  gicp.setInputTarget(tgt_cloud);

  aligned.reset(new pcl::PointCloud<pcl::PointXYZI>());
  gicp.align(*aligned);
  auto t4 = std::chrono::high_resolution_clock::now();

  double elapsed1 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  double elapsed2 = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e6;

  std::cout << elapsed1 << " " << elapsed2 << std::endl;

  viewer->update_drawable("fast_gicp", std::make_shared<glk::PointCloudBuffer>(aligned), guik::ShaderSetting().add("color_mode", 1).add("material_color", Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f)));
  viewer->spin();

  // Eigen::Matrix4d estimated = icp.align();


  return 0;
}