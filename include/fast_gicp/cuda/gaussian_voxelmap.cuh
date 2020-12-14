#ifndef FAST_GICP_CUDA_GAUSSIAN_VOXELMAP_CUH
#define FAST_GICP_CUDA_GAUSSIAN_VOXELMAP_CUH

#include <Eigen/Core>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace fast_gicp {

  class GaussianVoxelMap {
  public:
    GaussianVoxelMap(float resolution, int init_num_buckets = 8192 * 4, int max_bucket_scan_count = 10);
    ~GaussianVoxelMap();

    void create_voxelmap(const thrust::device_vector<Eigen::Vector3f>& points, const thrust::device_vector<Eigen::Matrix3f>& covariances);

  private:
    void create_hashtable(const thrust::device_vector<Eigen::Vector3i>& voxel_coords);

    const thrust::pair<Eigen::Vector3i, int>* find(const Eigen::Vector3i& coord, const thrust::host_vector<thrust::pair<Eigen::Vector3i, int>>& buckets) const;

    bool insert(const thrust::pair<Eigen::Vector3i, int>& voxel, thrust::host_vector<thrust::pair<Eigen::Vector3i, int>>& buckets);

    void rehash(thrust::host_vector<thrust::pair<Eigen::Vector3i, int>>& buckets);

  public:
    const int init_num_buckets;
    const int max_bucket_scan_count;
    const float voxel_resolution;

    int num_voxels;
    thrust::device_vector<thrust::pair<Eigen::Vector3i, int>> buckets;

    // voxel data
    thrust::device_vector<int> num_points;
    thrust::device_vector<Eigen::Vector3f> voxel_means;
    thrust::device_vector<Eigen::Matrix3f> voxel_covs;
  };

} // namespace fast_gicp


#endif