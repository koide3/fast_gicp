#include <chrono>
#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/circular_buffer.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

class KittiLoader {
public:
  KittiLoader(const std::string& dataset_path) : dataset_path(dataset_path) {
    for(num_frames = 0;; num_frames++) {
      std::string filename = (boost::format("%s/%06d.bin") % dataset_path % num_frames).str();
      if(!boost::filesystem::exists(filename)) {
        break;
      }
    }

    if(num_frames == 0) {
      std::cerr << "error: no files in " << dataset_path << std::endl;
    }
  }
  ~KittiLoader() {}

  size_t size() const {
    return num_frames;
  }

  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud(size_t i) const {
    std::string filename = (boost::format("%s/%06d.bin") % dataset_path % i).str();
    FILE* file = fopen(filename.c_str(), "rb");
    if(!file) {
      std::cerr << "error: failed to load " << filename << std::endl;
      return nullptr;
    }

    std::vector<float> buffer(1000000);
    size_t num_points = fread(reinterpret_cast<char*>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
    fclose(file);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->resize(num_points);

    for(int i = 0; i < num_points; i++) {
      auto& pt = cloud->at(i);
      pt.x = buffer[i * 4];
      pt.y = buffer[i * 4 + 1];
      pt.z = buffer[i * 4 + 2];
      // pt.intensity = buffer[i * 4 + 3];
    }

    return cloud;
  }

private:
  int num_frames;
  std::string dataset_path;
};

int main(int argc, char** argv) {
  if(argc < 2) {
    std::cout << "usage: gicp_kitti /your/kitti/path/sequences/00/velodyne" << std::endl;
    return 0;
  }

  KittiLoader kitti(argv[1]);

  // use downsample_resolution=1.0 for fast registration
  double downsample_resolution = 0.25;
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);

  // registration method
  // you should fine-tune hyper-parameters (e.g., voxel resolution, max correspondence distance) for the best result
  // fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  // fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setResolution(0.5);
  gicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
  // gicp.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
  gicp.setMaxCorrespondenceDistance(1.0);

  // set initial frame as target
  voxelgrid.setInputCloud(kitti.cloud(0));
  pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
  voxelgrid.filter(*target);
  gicp.setInputTarget(target);

  // sensor pose sequence
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses(kitti.size());
  poses[0].setIdentity();

  // trajectory for visualization
  pcl::PointCloud<pcl::PointXYZ>::Ptr trajectory(new pcl::PointCloud<pcl::PointXYZ>);
  trajectory->push_back(pcl::PointXYZ(0.0f, 0.0f, 0.0f));

  pcl::visualization::PCLVisualizer vis;
  vis.addPointCloud<pcl::PointXYZ>(trajectory, "trajectory");

  // for calculating FPS
  boost::circular_buffer<std::chrono::high_resolution_clock::time_point> stamps(30);
  stamps.push_back(std::chrono::high_resolution_clock::now());

  for(int i = 1; i < kitti.size(); i++) {
    // set the current frame as source
    voxelgrid.setInputCloud(kitti.cloud(i));
    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid.filter(*source);
    gicp.setInputSource(source);

    // align and swap source and target cloud for next registration
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
    gicp.align(*aligned);
    gicp.swapSourceAndTarget();

    // accumulate pose
    poses[i] = poses[i - 1] * gicp.getFinalTransformation().cast<double>();

    // FPS display
    stamps.push_back(std::chrono::high_resolution_clock::now());
    std::cout << stamps.size() / (std::chrono::duration_cast<std::chrono::nanoseconds>(stamps.back() - stamps.front()).count() / 1e9) << "fps" << std::endl;

    // visualization
    trajectory->push_back(pcl::PointXYZ(poses[i](0, 3), poses[i](1, 3), poses[i](2, 3)));
    vis.updatePointCloud<pcl::PointXYZ>(trajectory, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(trajectory, 255.0, 0.0, 0.0), "trajectory");
    vis.spinOnce();
  }

  // save the estimated poses
  std::ofstream ofs("/tmp/traj.txt");
  for(const auto& pose : poses) {
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 4; j++) {
        if(i || j) {
          ofs << " ";
        }

        ofs << pose(i, j);
      }
    }
    ofs << std::endl;
  }

  return 0;
}