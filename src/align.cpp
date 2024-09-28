#include <chrono>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include "fast_gicp/gicp/gicp_settings.hpp"

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

constexpr double TRANSFORMATION_EPSILON = 1e-4;

// benchmark for PCL's registration methods
template <typename Registration>
void test_pcl(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  double fitness_score = 0.0;

  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  reg.setTransformationEpsilon(TRANSFORMATION_EPSILON);
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = reg.getFitnessScore();
  std::cout << "Transform: \n" << reg.getFinalTransformation().template cast<double>() << "\n";
  std::cout << "Fitness score: " << fitness_score << "\n";

  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "single:" << single << "[msec] " << std::flush;

  // 100 times
  // t1 = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 100; i++) {
  //   reg.setInputTarget(target);
  //   reg.setInputSource(source);
  //   reg.align(*aligned);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  // std::cout << "100times:" << multi << "[msec] fitness_score:" << fitness_score << std::endl;
}

// benchmark for fast_gicp registration methods
template <typename Registration>
void test(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  double fitness_score = 0.0;

  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  // fast_gicp reuses calculated covariances if an input cloud is the same as the previous one
  // to prevent this for benchmarking, force clear source and target clouds
  reg.clearTarget();
  reg.clearSource();
  reg.setTransformationEpsilon(TRANSFORMATION_EPSILON);
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = reg.getFitnessScore();
  auto has_converged = reg.hasConverged();
  auto H = reg.getFinalHessian();
  std::cout << "Transform: \n" << reg.getFinalTransformation().template cast<double>() << "\n";
  std::cout << "Fitness score: " << fitness_score << " Has converged? " << has_converged << "\n";
  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

  std::cout << "single:" << single << "[msec] " << std::flush;

  // 100 times
  // t1 = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 10; i++) {
  //   reg.clearTarget();
  //   reg.clearSource();
  //   reg.setInputTarget(target);
  //   reg.setInputSource(source);
  //   reg.align(*aligned);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  // std::cout << "10times:" << multi / 10.0 << "[msec] " << std::flush;

  // for some tasks like odometry calculation,
  // you can reuse the covariances of a source point cloud in the next registration
  // t1 = std::chrono::high_resolution_clock::now();
  // pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_ = target;
  // pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_ = source;
  // for (int i = 0; i < 10; i++) {
  //   reg.swapSourceAndTarget();
  //   reg.clearSource();

  //   reg.setInputTarget(target_);
  //   reg.setInputSource(source_);
  //   reg.align(*aligned);

  //   target_.swap(source_);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double reuse = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

  // std::cout << "10times_reuse:" << reuse / 10.0 << "[msec] fitness_score:" << fitness_score << std::endl;
}

/**
 * @brief main
 */
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "usage: gicp_align target_pcd source_pcd" << std::endl;
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if (pcl::io::loadPCDFile(argv[1], *target_cloud)) {
    std::cerr << "failed to open " << argv[1] << std::endl;
    return 1;
  }
  if (pcl::io::loadPCDFile(argv[2], *source_cloud)) {
    std::cerr << "failed to open " << argv[2] << std::endl;
    return 1;
  }

  // remove invalid points around origin
  source_cloud->erase(
    std::remove_if(source_cloud->begin(), source_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
    source_cloud->end());
  target_cloud->erase(
    std::remove_if(target_cloud->begin(), target_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
    target_cloud->end());

  // downsampling
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  float voxel_leaf_size = 0.1;
  voxelgrid.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*filtered);
  target_cloud = filtered;

  filtered.reset(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*filtered);
  source_cloud = filtered;

  std::cout << "target:" << target_cloud->size() << "[pts] source:" << source_cloud->size() << "[pts]" << std::endl;

  std::cout << "--- pcl_gicp ---" << std::endl;
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> pcl_gicp;
  test_pcl(pcl_gicp, target_cloud, source_cloud);

  std::cout << "--- pcl_ndt ---" << std::endl;
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> pcl_ndt;
  pcl_ndt.setResolution(1.0);
  test_pcl(pcl_ndt, target_cloud, source_cloud);
  std::cout << "Num iter: " << pcl_ndt.getFinalNumIteration() << "\n\n";

  std::cout << "--- fgicp_st ---" << std::endl;
  fast_gicp::FastGICPSingleThread<pcl::PointXYZ, pcl::PointXYZ> fgicp_st;
  test(fgicp_st, target_cloud, source_cloud);

  std::cout << "--- fgicp_mt ---" << std::endl;
  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> fgicp_mt;
  // fast_gicp uses all the CPU cores by default
  // fgicp_mt.setNumThreads(8);
  test(fgicp_mt, target_cloud, source_cloud);

  std::cout << "--- vgicp_st ---" << std::endl;
  fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp;
  vgicp.setResolution(1.0);
  vgicp.setNumThreads(1);
  test(vgicp, target_cloud, source_cloud);

  std::cout << "--- vgicp_mt ---" << std::endl;
  vgicp.setNumThreads(omp_get_max_threads());
  test(vgicp, target_cloud, source_cloud);

#ifdef USE_VGICP_CUDA
  std::cout << "--- ndt_cuda (P2D) ---" << std::endl;
  fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ> ndt_cuda;
  ndt_cuda.setResolution(1.0);
  ndt_cuda.setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
  test(ndt_cuda, target_cloud, source_cloud);
  std::cout << "Num iter: " << ndt_cuda.getFinalNumIterations() << "\n\n";

  std::cout << "--- ndt_cuda (D2D - DIRECT7) ---" << std::endl;
  // std::cout << "getMaxCorrespondenceDistance: " << ndt_cuda.getMaxCorrespondenceDistance() << "\n";
  // std::cout << "getRANSACOutlierRejectionThreshold: " << ndt_cuda.getRANSACOutlierRejectionThreshold() << "\n";
  // std::cout << "getRANSACIterations: " << ndt_cuda.getRANSACIterations() << "\n";
  // std::cout << "getEuclideanFitnessEpsilon: " << ndt_cuda.getEuclideanFitnessEpsilon() << "\n";
  // std::cout << "getMaximumIterations: " << ndt_cuda.getMaximumIterations() << "\n";
  // std::cout << "getTransformationEpsilon: " << ndt_cuda.getTransformationEpsilon() << "\n";
  // std::cout << "getTransformationRotationEpsilon: " << ndt_cuda.getTransformationRotationEpsilon() << "\n";
  ndt_cuda.setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
  ndt_cuda.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
  ndt_cuda.setTransformationEpsilon(1e-4);
  ndt_cuda.setMaximumIterations(64);
  // ndt_cuda.setEuclideanFitnessEpsilon();
  test(ndt_cuda, target_cloud, source_cloud);
  std::cout << "Num iter: " << ndt_cuda.getFinalNumIterations() << "\n\n";

  std::cout << "--- ndt_cuda (D2D - DIRECT1) ---" << std::endl;
  ndt_cuda.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
  test(ndt_cuda, target_cloud, source_cloud);

  std::cout << "--- ndt_cuda (D2D - DIRECT27) ---" << std::endl;
  ndt_cuda.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT27);
  test(ndt_cuda, target_cloud, source_cloud);

  std::cout << "--- ndt_cuda (D2D - RADIUS) ---" << std::endl;
  ndt_cuda.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, 1.0);
  test(ndt_cuda, target_cloud, source_cloud);

  std::cout << "--- vgicp_cuda (parallel_kdtree) ---" << std::endl;
  fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp_cuda;
  vgicp_cuda.setResolution(1.0);
  // vgicp_cuda uses CPU-based parallel KDTree in covariance estimation by default
  // on a modern CPU, it is faster than GPU_BRUTEFORCE
  // vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
  test(vgicp_cuda, target_cloud, source_cloud);
  std::cout << "Num iter: " << vgicp_cuda.getFinalNumIterations() << "\n\n";

  std::cout << "--- vgicp_cuda (gpu_bruteforce) ---" << std::endl;
  // use GPU-based bruteforce nearest neighbor search for covariance estimation
  // this would be a good choice if your PC has a weak CPU and a strong GPU (e.g., NVIDIA Jetson)
  vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  test(vgicp_cuda, target_cloud, source_cloud);
  std::cout << "Num iter: " << vgicp_cuda.getFinalNumIterations() << "\n\n";

  std::cout << "--- vgicp_cuda (gpu_rbf_kernel) ---" << std::endl;
  // use RBF-kernel-based covariance estimation
  // extremely fast but maybe a bit inaccurate
  vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
  // kernel width (and distance threshold) need to be tuned
  vgicp_cuda.setKernelWidth(0.5);
  test(vgicp_cuda, target_cloud, source_cloud);
  std::cout << "Num iter: " << vgicp_cuda.getFinalNumIterations() << "\n\n";
#endif

  return 0;
}
