FROM nvidia/cuda:11.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    && apt-get install --no-install-recommends -y wget nano cmake build-essential git \
    ca-certificates libeigen3-dev libboost-all-dev libflann-dev libvtk7-dev libomp-dev libgtest-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN git clone https://github.com/PointCloudLibrary/pcl

WORKDIR /root/pcl/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_geometry=OFF -DBUILD_keypoints=OFF -DBUILD_ml=OFF -DBUILD_outofcore=OFF -DBUILD_people=OFF \
    -DBUILD_recognition=OFF -DBUILD_segmentation=OFF -DBUILD_stereo=OFF -DBUILD_surface=OFF -DBUILD_tools=OFF -DBUILD_tracking=OFF
RUN make -j$(nproc) && make install

COPY . /root/fast_gicp
WORKDIR /root/fast_gicp
RUN git submodule init && git submodule update

RUN rm -rf /root/fast_gicp/build
WORKDIR /root/fast_gicp/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_VGICP_CUDA=ON -DBUILD_apps=OFF -DBUILD_test=ON
RUN make -j$(nproc)

WORKDIR /root

CMD ["bash"]
