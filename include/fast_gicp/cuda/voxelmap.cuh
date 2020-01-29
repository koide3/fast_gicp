#ifndef FAST_GICP_CUDA_VOXELMAP_CUH
#define FAST_GICP_CUDA_VOXELMAP_CUH

#include <chrono>
#include <Eigen/Core>
#include <thrust/unique.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace fast_gicp {

namespace {
  // taken from boost/hash.hpp
  __host__ __device__ void hash_combine(uint64_t& h, uint64_t k) {
    const uint64_t m = UINT64_C(0xc6a4a7935bd1e995);
    const int r = 47;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;

    h += 0xe6546b64;
  }

  __host__ __device__ uint64_t vector3i_hash(const Eigen::Vector3i& x) {
    uint64_t seed = 0;
    hash_combine(seed, x[0]);
    hash_combine(seed, x[1]);
    hash_combine(seed, x[2]);
    return seed;
  }

  struct voxel_coord {
    voxel_coord(float resolution) : resolution(resolution) {}

    __host__ __device__ Eigen::Vector3i operator()(const Eigen::Vector3f& x) {
      Eigen::Vector3i coord = (x.array() / resolution - 0.5).floor().cast<int>();
      return coord;
    }

    float resolution;
  };

  }  // namespace

  class VoxelMapCuda {
  public:
    VoxelMapCuda(float resolution, int init_num_buckets = 8192 * 2, int max_bucket_scan_count=10) : init_num_buckets(init_num_buckets), max_bucket_scan_count(max_bucket_scan_count), voxel_resolution(resolution) {}

    void create_hashtable(const thrust::device_vector<Eigen::Vector3f>& points) {
      thrust::device_vector<Eigen::Vector3i> voxel_coords(points.size());
      thrust::transform(points.begin(), points.end(), voxel_coords.begin(), voxel_coord(voxel_resolution));

      thrust::host_vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> coords = voxel_coords;

      bucket_contents.reserve(8192);

      buckets.resize(init_num_buckets);
      thrust::fill(buckets.begin(), buckets.end(), -1);

      for(int i = 0; i<coords.size(); i++) {
        auto voxel_data = thrust::make_tuple(vector3i_hash(coords[i]), coords[i], -1);
        while(!insert(voxel_data, buckets)) {
          rehash(buckets);
        }
      }
    }

    const thrust::tuple<uint64_t, Eigen::Vector3i, int>* find(const Eigen::Vector3i& coord) const {
      uint64_t hash = vector3i_hash(coord);
      for(int i = 0; i < max_bucket_scan_count; i++) {
        size_t bucket_index = (hash + i) % buckets.size();
        int bucket_content_id = buckets[bucket_index];

        if(bucket_content_id >= 0) {
          return &bucket_contents[bucket_content_id];
        }
        if(bucket_content_id < 0) {
          return nullptr;
        }
      }

      return nullptr;
    }

    bool insert(const thrust::tuple<uint64_t, Eigen::Vector3i, int>& voxel, thrust::host_vector<int>& buckets) {
      uint64_t hash = thrust::get<0>(voxel);
      const auto& coord = thrust::get<1>(voxel);

      for(int i = 0; i < max_bucket_scan_count; i++) {
        size_t bucket_index = (hash + i) % buckets.size();
        int bucket_content_id = buckets[bucket_index];

        // insert voxel in the bucket
        if(bucket_content_id < 0) {
          if(thrust::get<2>(voxel) < 0) {
            buckets[bucket_index] = bucket_contents.size();
            bucket_contents.push_back(thrust::make_tuple(thrust::get<0>(voxel), thrust::get<1>(voxel), bucket_contents.size()));
          } else {
            buckets[bucket_index] = thrust::get<2>(voxel);
          }

          return true;
        }

        // voxel found in the bucket
        if(thrust::get<1>(bucket_contents[bucket_content_id]) == coord) {
          return true;
        }
      }

      return false;
    }

    void rehash(thrust::host_vector<int>& buckets) {
      // the bigger is the better...
      int new_num_buckets = buckets.size() * 2 - 1;
      thrust::host_vector<int> new_buckets(new_num_buckets);
      thrust::fill(new_buckets.begin(), new_buckets.end(), -1);
      std::cout << "rehash:" << new_buckets.size() << std::endl;

      for(int i = 0; i < bucket_contents.size(); i++) {
        while(!insert(bucket_contents[i], new_buckets)) {
          rehash(new_buckets);
        }
      }

      buckets.swap(new_buckets);
    }

  private:
    const int init_num_buckets;
    const int max_bucket_scan_count;
    const float voxel_resolution;

    thrust::host_vector<int> buckets;                                                    // backet content ids
    thrust::host_vector<thrust::tuple<uint64_t, Eigen::Vector3i, int>> bucket_contents;  // hash, coord, voxel id
  };

} // namespace fast_gicp


#endif