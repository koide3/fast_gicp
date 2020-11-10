# fast_gicp

This package is a collection of GICP-based fast point cloud registration algorithms. It constains a multi-threaded GICP as well as multi-thread and GPU implementations of our voxelized GICP (VGICP) algorithm. All the implemented algorithms have the PCL registration interface so that they can be used as an inplace replacement for GICP in PCL.

- FastGICP: multi-threaded GICP algorithm (**\~40FPS**)
- FastGICPSingleThread: GICP algorithm optimized for single-threading (**\~15FPS**)
- FastVGICP: multi-threaded and voxelized GICP algorithm (**\~70FPS**)
- FastVGICPCuda: CUDA-optimized voxelized GICP algorithm (**\~120FPS**)
![proctime](data/proctime.png)

[![Build Status](https://travis-ci.org/SMRT-AIST/fast_gicp.svg?branch=master)](https://travis-ci.org/SMRT-AIST/fast_gicp) on melodic & noetic

## Installation

### Dependencies
- PCL
- Eigen
- [Sophus](https://github.com/strasdat/Sophus)
- [nvbio](https://github.com/NVlabs/nvbio)
- OpenMP (optional)
- CUDA (optional)

We have tested this package with Ubuntu 18.04, ROS melodic, and CUDA 10.2.

### CUDA

To enable CUDA-based features, uncomment ```find_package(CUDA)``` in ```CMakeLists.txt```.

### ROS
```bash
cd ~/catkin_ws/src
git clone https://github.com/SMRT-AIST/fast_gicp --recursive
cd .. && catkin_make -DCMAKE_BUILD_TYPE=Release
```

### Non-ROS
```bash
git clone https://github.com/SMRT-AIST/fast_gicp --recursive
mkdir fast_gicp/build && fast_gicp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## Benchmark
CPU:Core i9-9900K GPU:GeForce RTX2080Ti

```bash
roscd fast_gicp/data
rosrun fast_gicp gicp_align 251370668.pcd 251371071.pcd
```

```
target:17249[pts] source:17518[pts]
--- pcl_gicp ---
single:116.732[msec] 100times:10867.1[msec] fitness_score:0.204306
--- pcl_ndt ---
single:52.8007[msec] 100times:5220.49[msec] fitness_score:0.226416
--- fgicp_st ---
single:110.343[msec] 100times:10651.2[msec] 100times_reuse:6962.1[msec] fitness_score:0.0922969
--- fgicp_mt ---
single:24.3643[msec] 100times:2716.7[msec] 100times_reuse:1799.1[msec] fitness_score:0.0922969
--- vgicp_st ---
single:115.041[msec] 100times:8759.43[msec] 100times_reuse:4784.57[msec] fitness_score:0.0912174
--- vgicp_mt ---
single:19.705[msec] 100times:1963.74[msec] 100times_reuse:1044.29[msec] fitness_score:0.0912174
--- vgicp_cuda (parallel_kdtree) ---
single:16.1846[msec] 100times:1611.89[msec] 100times_reuse:779.65[msec] fitness_score:0.0709287
--- vgicp_cuda (gpu_bruteforce) ---
single:49.7294[msec] 100times:3145.78[msec] 100times_reuse:1541.36[msec] fitness_score:0.0710122

```

See [src/align.cpp](https://github.com/SMRT-AIST/fast_gicp/blob/master/src/align.cpp) for the detailed usage.

## Test on KITTI

```bash
# Perform frame-by-frame registration
rosrun fast_gicp gicp_kitti /your/kitti/path/sequences/00/velodyne
```

![kitti00](https://user-images.githubusercontent.com/31344317/86207074-b98ac280-bba8-11ea-9687-e65f03aaf25b.png)

## Related packages
- [ndt_omp](https://github.com/koide3/ndt_omp)
- [fast_gicp](https://github.com/SMRT-AIST/fast_gicp)


## Papers
- Kenji Koide, Masashi Yokozuka, Shuji Oishi, and Atsuhiko Banno, Voxelized GICP for fast and accurate 3D point cloud registration [[link]](https://easychair.org/publications/preprint/ftvV)

## Contact
Kenji Koide, k.koide@aist.go.jp

Robot Innovation Research Center, National Institute of Advanced Industrial Science and Technology, Japan  [\[URL\]](https://unit.aist.go.jp/rirc/en/team/smart_mobility.html)
