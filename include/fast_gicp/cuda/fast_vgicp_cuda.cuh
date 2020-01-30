#ifndef FAST_GICP_FAST_VGICP_CUDA_CUH
#define FAST_GICP_FAST_VGICP_CUDA_CUH

#include <memory>
#include <vector>
#include <Eigen/Core>

namespace thrust {
template<typename T>
class device_allocator;

template<typename T, typename Alloc>
class device_vector;
}

namespace fast_gicp {

class GaussianVoxelMap;

class FastVGICPCudaCore {
public:
  using Points = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
  using Indices = thrust::device_vector<int, thrust::device_allocator<int>>;
  using Matrices = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;

  FastVGICPCudaCore();
  ~FastVGICPCudaCore();

  void set_resolution(double resolution);
  void set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud);
  void set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud);

  void set_source_neighbors(int k, const std::vector<int>& neighbors);
  void set_target_neighbors(int k, const std::vector<int>& neighbors);
  void find_source_neighbors(int k);
  void find_target_neighbors(int k);

  void calculate_source_covariances();
  void calculate_target_covariances();

  void create_target_voxelmap();

  void test_print();

private:
  double resolution;

  std::unique_ptr<Points> source_points;
  std::unique_ptr<Points> target_points;

  std::unique_ptr<Indices> source_neighbors;
  std::unique_ptr<Indices> target_neighbors;

  std::unique_ptr<Matrices> source_covariances;
  std::unique_ptr<Matrices> target_covariances;

  std::unique_ptr<GaussianVoxelMap> voxelmap;
};

} // namespace fast_gicp


#endif