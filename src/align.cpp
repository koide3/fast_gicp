#include <chrono>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/gicp.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZI>());

  pcl::io::loadPCDFile("/home/koide/fails/failed_target_640.pcd", *tgt_cloud);
  pcl::io::loadPCDFile("/home/koide/fails/failed_source_640.pcd", *src_cloud);

  pcl::ApproximateVoxelGrid<pcl::PointXYZI> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
  voxelgrid.setInputCloud(tgt_cloud);
  voxelgrid.filter(*filtered);
  tgt_cloud = filtered;

  filtered.reset(new pcl::PointCloud<pcl::PointXYZI>());
  voxelgrid.setInputCloud(src_cloud);
  voxelgrid.filter(*filtered);
  src_cloud = filtered;

  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("source", std::make_shared<glk::PointCloudBuffer>(src_cloud), guik::ShaderSetting().add("color_mode", 1).add("material_color", Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)).add("point_scale", 3.0f));
  viewer->update_drawable("target", std::make_shared<glk::PointCloudBuffer>(tgt_cloud), guik::ShaderSetting().add("color_mode", 1).add("material_color", Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f)).add("point_scale", 3.0f));

  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned;

  // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> vgicp;
  fast_gicp::FastGICP<pcl::PointXYZI, pcl::PointXYZI> vgicp;
  aligned.reset(new pcl::PointCloud<pcl::PointXYZI>());
  auto t1 = std::chrono::high_resolution_clock::now();
  vgicp.setInputSource(src_cloud);
  vgicp.setInputTarget(tgt_cloud);
  vgicp.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "vgicp:" << elapsed << "[msec]" << std::endl;
  viewer->update_drawable("source", std::make_shared<glk::PointCloudBuffer>(src_cloud), guik::ShaderSetting().add("color_mode", 1).add("material_color", Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)).add("point_scale", 3.0f).add("model_matrix", vgicp.getFinalTransformation()));
  // viewer->update_drawable("vgicp", std::make_shared<glk::PointCloudBuffer>(aligned), guik::ShaderSetting().add("color_mode", 1).add("material_color", Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f)).add("point_scale", 3.0f));

  viewer->spin();

  // Eigen::Matrix4d estimated = icp.align();

  return 0;
}