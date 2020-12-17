#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/device_vector.h>
#include <fast_gicp/cuda/vector3_hash.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/find_voxel_correspondences.cuh>

namespace fast_gicp {
namespace cuda {

namespace {

struct find_voxel_correspondences_kernel {
  find_voxel_correspondences_kernel(const GaussianVoxelMap& voxelmap, const Eigen::Isometry3f& x)
      : R(x.linear()),
        t(x.translation()),
        voxelmap_info_ptr(voxelmap.voxelmap_info_ptr.data()),
        buckets_ptr(voxelmap.buckets.data()),
        voxel_num_points_ptr(voxelmap.num_points.data()),
        voxel_means_ptr(voxelmap.voxel_means.data()),
        voxel_covs_ptr(voxelmap.voxel_covs.data()) {}

  // lookup voxel
  __host__ __device__ int lookup_voxel(const Eigen::Vector3f& x) const {
    const VoxelMapInfo& voxelmap_info = *thrust::raw_pointer_cast(voxelmap_info_ptr);

    Eigen::Vector3i coord = calc_voxel_coord(x, voxelmap_info.voxel_resolution);
    uint64_t hash = vector3i_hash(coord);

    for(int i = 0; i < voxelmap_info.max_bucket_scan_count; i++) {
      uint64_t bucket_index = (hash + i) % voxelmap_info.num_buckets;
      const thrust::pair<Eigen::Vector3i, int>& bucket = thrust::raw_pointer_cast(buckets_ptr)[bucket_index];

      if(bucket.second < 0) {
        return -1;
      }

      if(bucket.first == coord) {
        return bucket.second;
      }
    }

    return -1;
  }

  __host__ __device__ int operator()(const Eigen::Vector3f& pt) const {
    return lookup_voxel(R * pt + t);
  }

  const Eigen::Matrix3f R;
  const Eigen::Vector3f t;

  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
  thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr;

  thrust::device_ptr<const int> voxel_num_points_ptr;
  thrust::device_ptr<const Eigen::Vector3f> voxel_means_ptr;
  thrust::device_ptr<const Eigen::Matrix3f> voxel_covs_ptr;
};
}  // namespace

void find_voxel_correspondences(const thrust::device_vector<Eigen::Vector3f>& src_points, const GaussianVoxelMap& voxelmap, const Eigen::Isometry3f& x, thrust::device_vector<int>& correspondences) {
  correspondences.resize(src_points.size());
  thrust::transform(src_points.begin(), src_points.end(), correspondences.begin(), find_voxel_correspondences_kernel(voxelmap, x));
}

}  // namespace cuda
}  // namespace fast_gicp
