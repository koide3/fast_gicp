#include <iostream>
#include <algorithm>
#include <Eigen/Core>

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main(int argc, char** argv) {
  thrust::host_vector<Eigen::Vector4f> h_vec(10);

  return 0;
}
