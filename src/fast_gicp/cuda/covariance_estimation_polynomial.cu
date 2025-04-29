#include <fast_gicp/cuda/covariance_estimation.cuh>

#include <thrust/device_vector.h>

#include <thrust/async/for_each.h>
#include <thrust/async/transform.h>

namespace fast_gicp {
namespace cuda {

struct NormalDistribution {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  __host__ __device__ NormalDistribution() {}

  static __host__ __device__ NormalDistribution zero() {
    NormalDistribution dist;
    dist.sum_weights = 0.0f;
    dist.mean.setZero();
    dist.cov.setZero();
    return dist;
  }

  __host__ __device__ NormalDistribution operator+(const NormalDistribution& rhs) const {
    NormalDistribution sum;
    sum.sum_weights = sum_weights + rhs.sum_weights;
    sum.mean = mean + rhs.mean;
    sum.cov = cov + rhs.cov;
    return sum;
  }

  __host__ __device__ NormalDistribution& operator+=(const NormalDistribution& rhs) {
    sum_weights += rhs.sum_weights;
    mean += rhs.mean;
    cov += rhs.cov;
    return *this;
  }

  __host__ __device__ void accumulate(const float w, const Eigen::Vector3f& x) {
    sum_weights += w;
    mean += w * x;
    cov += w * x * x.transpose();
  }

  __host__ __device__ NormalDistribution& finalize() {
    Eigen::Vector3f sum_pt = mean;
    mean /= sum_weights;
    cov = (cov - mean * sum_pt.transpose()) / sum_weights;

    return *this;
  }

  float sum_weights;
  Eigen::Vector3f mean;
  Eigen::Matrix3f cov;
};

struct covariance_estimation_kernel {
  static const int BLOCK_SIZE = 512;

  covariance_estimation_kernel(thrust::device_ptr<const float> exp_factor_ptr, thrust::device_ptr<const float> max_dist_ptr, thrust::device_ptr<const Eigen::Vector3f> points_ptr)
  : exp_factor_ptr(exp_factor_ptr),
    max_dist_ptr(max_dist_ptr),
    points_ptr(points_ptr) {}

  __host__ __device__ NormalDistribution operator()(const Eigen::Vector3f& x) const {
    const float exp_factor = *thrust::raw_pointer_cast(exp_factor_ptr);
    const float max_dist = *thrust::raw_pointer_cast(max_dist_ptr);
    const float max_dist_sq = max_dist * max_dist;
    const Eigen::Vector3f* points = thrust::raw_pointer_cast(points_ptr);

    NormalDistribution dist = NormalDistribution::zero();
    for (int i = 0; i < BLOCK_SIZE; i++) {
      float sq_d = (x - points[i]).squaredNorm();
      if (sq_d > max_dist_sq) {
        continue;
      }

      float w = expf(-exp_factor * sq_d);
      dist.accumulate(w, points[i]);
    }

    return dist;
  }

  thrust::device_ptr<const float> exp_factor_ptr;
  thrust::device_ptr<const float> max_dist_ptr;
  thrust::device_ptr<const Eigen::Vector3f> points_ptr;
};

struct finalization_kernel_polynomial {
  finalization_kernel_polynomial(const int stride, const thrust::device_vector<double>& accumulated_dists)
  : stride(stride),
    accumulated_dists_first(accumulated_dists.data()),
    accumulated_dists_last(accumulated_dists.data() + accumulated_dists.size()) {}

  __host__ __device__ Eigen::Matrix3f operator()(int index) const {
    const double* dists = thrust::raw_pointer_cast(accumulated_dists_first);
    const double* dists_last = thrust::raw_pointer_cast(accumulated_dists_last);
    const int num_dists = dists_last - dists;

    double sum = dists[index];
    for (int dist_index = index + stride; dist_index < num_dists; dist_index += stride) {
      sum += dists[dist_index];
    }

    return sum * Eigen::Matrix3f::Identity(); // Since we used a polynomial kernel, we only accumulate the distances.
  }

  const int stride;
  thrust::device_ptr<const double> accumulated_dists_first;
  thrust::device_ptr<const double> accumulated_dists_last;
};

// 수정된 다항식 커널 함수
struct PolynomialKernel {
    double alpha; // 커널 계수
    double constant; // 상수항
    int degree; // 다항식의 차수

    __host__ __device__
    PolynomialKernel(double alpha, double constant, int degree) : alpha(alpha), constant(constant), degree(degree) {}

    // 다항식 커널 계산
    __host__ __device__
    double operator()(const Eigen::Vector3f& x, const Eigen::Vector3f& y) const {
        return pow((alpha * x.dot(y) + constant), degree);
    }
};

// covariance_estimation_polynomial 함수 수정
void covariance_estimation_polynomial(const thrust::device_vector<Eigen::Vector3f>& points, double alpha, double constant, int degree, thrust::device_vector<Eigen::Matrix3f>& covariances) {
    covariances.resize(points.size());

    int num_blocks = (points.size() + (covariance_estimation_kernel::BLOCK_SIZE - 1)) / covariance_estimation_kernel::BLOCK_SIZE;
    // padding
    thrust::device_vector<Eigen::Vector3f> ext_points(num_blocks * covariance_estimation_kernel::BLOCK_SIZE);
    thrust::copy(points.begin(), points.end(), ext_points.begin());
    thrust::fill(ext_points.begin() + points.size(), ext_points.end(), Eigen::Vector3f(0.0f, 0.0f, 0.0f));

    thrust::device_vector<double> accumulated_dists(points.size() * num_blocks);

    thrust::system::cuda::detail::unique_stream stream;
    std::vector<thrust::system::cuda::unique_eager_event> events(num_blocks);

    // accumulate polynomial kerneled point distributions
    for (int i = 0; i < num_blocks; i++) {
        thrust::device_ptr<const Eigen::Vector3f> block_begin = ext_points.data() + covariance_estimation_kernel::BLOCK_SIZE * i;
        thrust::device_ptr<const Eigen::Vector3f> block_end = block_begin + thrust::min(covariance_estimation_kernel::BLOCK_SIZE, static_cast<int>(points.size()) - covariance_estimation_kernel::BLOCK_SIZE * i);
        
        // Apply the polynomial kernel to the data and accumulate the results
        thrust::transform(block_begin, block_end, points.begin(), accumulated_dists.begin() + points.size() * i, PolynomialKernel(alpha, constant, degree));
    }

    // finalize distributions
    thrust::transform(
        thrust::cuda::par.on(stream.native_handle()),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(points.size()),
        covariances.begin(),
        finalization_kernel_polynomial(points.size(), accumulated_dists));
}


}  // namespace cuda
}  // namespace fast_gicp