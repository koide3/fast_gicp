#include <vector>
#include <iostream>
#include <Eigen/Core>

#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/device_new.h>
#include <thrust/device_malloc.h>

int main(int argc, char ** argv) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  Eigen::Matrix<float, 20, 10> mat_A_ = Eigen::Matrix<float, 20, 10>::Random();

  thrust::device_ptr<Eigen::Matrix<float, 20, 10>> mat_A = thrust::device_new<Eigen::Matrix<float, 20, 10>>(1);
  thrust::device_ptr<Eigen::Matrix<float, 10, 10>> mat_C = thrust::device_new<Eigen::Matrix<float, 10, 10>>(1);

  float* mat_A_ptr = reinterpret_cast<float*>(thrust::raw_pointer_cast(mat_A));
  float* mat_C_ptr = reinterpret_cast<float*>(thrust::raw_pointer_cast(mat_C));
  cublasSetMatrix(20, 10, sizeof(float), mat_A_.data(), 20, mat_A_ptr, 20);

  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 10, 10, 20, &alpha, mat_A_ptr, 20, mat_A_ptr, 20, &beta, mat_C_ptr, 10);

  Eigen::Matrix<float, 10, 10> mat_C_;
  cublasGetMatrix(10, 10, sizeof(float), mat_C_ptr, 10, mat_C_.data(), 10);

  std::cout << "--- cpu ---" << std::endl << mat_A_.transpose() * mat_A_ << std::endl;
  std::cout << "--- cuda ---" << std::endl << mat_C_ << std::endl;

  return 0;
}