FROM ros:melodic

RUN apt-get update && apt-get install --no-install-recommends -y \
    && apt-get install --no-install-recommends -y wget nano build-essential \
                                                  libomp-dev clang lld ros-melodic-pcl-ros \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld 50

RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace'

COPY . /root/catkin_ws/src/fast_gicp/
WORKDIR /root/catkin_ws/src/fast_gicp
RUN git submodule init && git submodule update

WORKDIR /root/catkin_ws
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; CC=clang CXX=clang++ catkin_make'
RUN sed -i "6i source \"/root/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
