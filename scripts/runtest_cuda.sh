#!/bin/bash
docker build --tag focal_cuda -f docker/focal_cuda/Dockerfile .
docker run --gpus all -it --rm focal_cuda /root/fast_gicp/build/gicp_test /root/fast_gicp/data
