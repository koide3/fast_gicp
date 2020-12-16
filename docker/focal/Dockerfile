FROM ros:focal

RUN apt-get update && apt-get install --no-install-recommends -y \
    && apt-get install --no-install-recommends -y wget nano build-essential \
    git
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN git clone https://github.com/PointCloudLibrary/pcl.git

WORKDIR /root

CMD ["bash"]
