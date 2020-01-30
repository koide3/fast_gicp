#ifndef FAST_GICP_CUDA_GAUSSIAN_VOXELMAP_CUH
#define FAST_GICP_CUDA_GAUSSIAN_VOXELMAP_CUH

#include <chrono>
#include <Eigen/Core>
#include <thrust/unique.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <fast_gicp/cuda/vector3_hash.cuh>

namespace fast_gicp {

namespace {
  // point coord -> voxel coord conversion
  struct voxel_coord_kernel {
    voxel_coord_kernel(float resolution) : resolution(resolution) {}

    __host__ __device__ Eigen::Vector3i operator()(const Eigen::Vector3f& x) const {
      return calc_voxel_coord(x, resolution);
    }

    float resolution;
  };


  struct voxel_accumulation_kernel {
    voxel_accumulation_kernel(float voxel_resolution, int max_bucket_scan_count, const thrust::device_vector<thrust::pair<Eigen::Vector3i, int>>& buckets, thrust::device_vector<int>& num_points, thrust::device_vector<Eigen::Vector3f>& voxel_means, thrust::device_vector<Eigen::Matrix3f>& voxel_covs)
        : voxel_resolution(voxel_resolution),
          max_bucket_scan_count(max_bucket_scan_count),
          num_buckets(buckets.size()),
          buckets_ptr(buckets.data()),
          num_points_ptr(num_points.data()),
          voxel_means_ptr(voxel_means.data()),
          voxel_covs_ptr(voxel_covs.data()) {}

    // accumulation
    template<typename Tuple>
    __device__ void operator()(const Tuple& tuple) const {
      // input data
      const Eigen::Vector3i& coord = thrust::get<0>(tuple);
      const Eigen::Vector3f& x = thrust::get<1>(tuple);
      const Eigen::Matrix3f& cov = thrust::get<2>(tuple);

      const thrust::pair<Eigen::Vector3i, int>* buckets = thrust::raw_pointer_cast(buckets_ptr);

      uint64_t hash = vector3i_hash(coord);
      for(int i = 0; i < max_bucket_scan_count; i++) {
        uint64_t bucket_index = (hash + i) % num_buckets;
        if(thrust::get<1>(buckets[bucket_index]) < 0) {
          return;
        }

        if(thrust::get<0>(buckets[bucket_index]) == coord) {
          size_t voxel_index = thrust::get<1>(buckets[bucket_index]);
          int* num_points = thrust::raw_pointer_cast(num_points_ptr) + voxel_index;
          Eigen::Vector3f* voxel_mean = thrust::raw_pointer_cast(voxel_means_ptr) + voxel_index;
          Eigen::Matrix3f* voxel_cov = thrust::raw_pointer_cast(voxel_covs_ptr) + voxel_index;

          atomicAdd(num_points, 1);
          for(int j = 0; j < 3; j++) {
            atomicAdd(voxel_mean->data() + j, *(x.data() + j));
          }
          for(int j = 0; j < 9; j++) {
            atomicAdd(voxel_cov->data() + j, *(cov.data() + j));
          }

          return;
        }
      }
    }

    const float voxel_resolution;
    const int max_bucket_scan_count;

    const int num_buckets;
    thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr;

    thrust::device_ptr<int> num_points_ptr;
    thrust::device_ptr<Eigen::Vector3f> voxel_means_ptr;
    thrust::device_ptr<Eigen::Matrix3f> voxel_covs_ptr;
  };

  struct voxel_normalization_kernel {
    template<typename Tuple>
    __device__ void operator()(Tuple tuple) const {
      thrust::get<1>(tuple) = thrust::get<1>(tuple) / thrust::get<0>(tuple);
      thrust::get<2>(tuple) = thrust::get<2>(tuple) / thrust::get<0>(tuple);
    }
  };
  }  // namespace

  class GaussianVoxelMap {
  public:
    GaussianVoxelMap(float resolution, int init_num_buckets = 8192 * 2, int max_bucket_scan_count = 10) : init_num_buckets(init_num_buckets), max_bucket_scan_count(max_bucket_scan_count), voxel_resolution(resolution) {}

    void create_voxelmap(const thrust::device_vector<Eigen::Vector3f>& points, const thrust::device_vector<Eigen::Matrix3f>& covariances) {
      thrust::device_vector<Eigen::Vector3i> voxel_coords(points.size());
      thrust::transform(points.begin(), points.end(), voxel_coords.begin(), voxel_coord_kernel(voxel_resolution));

      create_hashtable(voxel_coords);

      num_points.resize(num_voxels, 0);
      voxel_means.resize(num_voxels, Eigen::Vector3f::Zero());
      voxel_covs.resize(num_voxels, Eigen::Matrix3f::Zero());

      thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(voxel_coords.begin(), points.begin(), covariances.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(voxel_coords.end(), points.end(), covariances.end())),
        voxel_accumulation_kernel(voxel_resolution, max_bucket_scan_count, buckets, num_points, voxel_means, voxel_covs)
      );

      thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(num_points.begin(), voxel_means.begin(), voxel_covs.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(num_points.end(), voxel_means.end(), voxel_covs.end())),
        voxel_normalization_kernel()
      );
    }

  private:
    void create_hashtable(const thrust::device_vector<Eigen::Vector3i>& voxel_coords) {
      thrust::host_vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> coords = voxel_coords;

      num_voxels = 0;
      thrust::host_vector<thrust::pair<Eigen::Vector3i, int>> buckets(init_num_buckets);
      thrust::fill(buckets.begin(), buckets.end(), thrust::make_pair(Eigen::Vector3i(0, 0, 0), -1));

      for(int i = 0; i < coords.size(); i++) {
        auto voxel_data = thrust::make_pair(coords[i], -1);
        while(!insert(voxel_data, buckets)) {
          rehash(buckets);
        }
      }

      this->buckets = buckets;
    }

    const thrust::pair<Eigen::Vector3i, int>* find(const Eigen::Vector3i& coord, const thrust::host_vector<thrust::pair<Eigen::Vector3i, int>>& buckets) const {
      uint64_t hash = vector3i_hash(coord);
      for(int i = 0; i < max_bucket_scan_count; i++) {
        size_t bucket_index = (hash + i) % buckets.size();

        if(buckets[bucket_index].second < 0) {
          return nullptr;
        }
        if(buckets[bucket_index].first == coord) {
          return &buckets[bucket_index];
        }
      }

      return nullptr;
    }

    bool insert(const thrust::pair<Eigen::Vector3i, int>& voxel, thrust::host_vector<thrust::pair<Eigen::Vector3i, int>>& buckets) {
      const auto& coord = voxel.first;
      uint64_t hash = vector3i_hash(coord);

      for(int i = 0; i < max_bucket_scan_count; i++) {
        size_t bucket_index = (hash + i) % buckets.size();

        // insert voxel in the bucket
        if(buckets[bucket_index].second < 0) {
          if(voxel.second < 0) {
            buckets[bucket_index] = voxel;
            buckets[bucket_index].second = num_voxels;
            num_voxels++;
          } else {
            buckets[bucket_index] = voxel;
          }

          return true;
        }

        // voxel found in the bucket
        if(buckets[bucket_index].first == coord) {
          return true;
        }
      }

      return false;
    }

    void rehash(thrust::host_vector<thrust::pair<Eigen::Vector3i, int>>& buckets) {
      // the bigger is the better...
      int new_num_buckets = buckets.size() * 2 - 1;
      thrust::host_vector<thrust::pair<Eigen::Vector3i, int>> new_buckets(new_num_buckets);
      thrust::fill(new_buckets.begin(), new_buckets.end(), thrust::make_pair(Eigen::Vector3i(0, 0, 0), -1));
      std::cout << "rehash:" << new_buckets.size() << std::endl;

      for(int i = 0; i < buckets.size(); i++) {
        if(buckets[i].second < 0) {
          continue;
        }

        while(!insert(buckets[i], new_buckets)) {
          rehash(new_buckets);
        }
      }

      buckets.swap(new_buckets);
    }

  private:
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