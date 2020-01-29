#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <fast_gicp/cuda/fast_vgicp_cuda.cuh>

int main(int argc, char ** argv) {
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(1024);
  for(auto& pt: points) {
    pt.setRandom();
  }

  fast_gicp::FastVGICPCudaCore vgicp_core;
  vgicp_core.set_source_cloud(points);

  std::cout << "hello" << std::endl;
  return 0;
}