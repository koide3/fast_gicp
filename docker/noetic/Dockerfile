FROM ros:noetic

RUN apt-get update && apt-get install --no-install-recommends -y \
    && apt-get install --no-install-recommends -y wget nano build-essential \
                                                  git ros-noetic-pcl-ros \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_init_workspace'

COPY . /root/catkin_ws/src/fast_gicp/
WORKDIR /root/catkin_ws/src/fast_gicp
RUN git submodule init && git submodule update

WORKDIR /root/catkin_ws
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_make'
RUN sed -i "6i source \"/root/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
