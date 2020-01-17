#ifndef FAST_GICP_POINT_CLOUD_ORGANIZER_HPP
#define FAST_GICP_POINT_CLOUD_ORGANIZER_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace fast_gicp {

template<typename PointT>
class PointCloudOrganizer {
public:
  PointCloudOrganizer() {}

  boost::shared_ptr<pcl::PointCloud<PointT>> organize(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud, bool force_init_table = false) {
    if(tilt_table.empty() || force_init_table) {
      init_tilt_table(cloud);
    }

    boost::shared_ptr<pcl::PointCloud<PointT>> organized(new pcl::PointCloud<PointT>);
    organized->height = tilt_table.size();
    organized->width = cloud->size() / tilt_table.size();
    organized->is_dense = true;
    organized->resize(organized->width * organized->height);
    for(auto& pt : organized->points) {
      pt.getArray4fMap() = std::nanf("");
    }

    double pan_step = organized->width / (2.0 * M_PI);

    for(const auto& pt : cloud->points) {
      double dist = std::sqrt(pt.x * pt.x + pt.y * pt.y);
      double pan = std::atan2(pt.x, pt.y) + M_PI;
      double tilt = std::atan2(pt.z, dist);

      auto tilt_loc = std::upper_bound(tilt_table.begin(), tilt_table.end(), tilt);

      if(tilt_loc != tilt_table.begin()) {
        if(std::abs(*tilt_loc - tilt) > std::abs(*(tilt_loc - 1) - tilt)) {
          tilt_loc--;
        }
      }

      int tilt_idx = std::min<int>(std::distance(tilt_table.begin(), tilt_loc), tilt_table.size() - 1);
      int pan_idx = std::max<int>(0, std::min<int>(organized->width - 1, pan * pan_step));

      organized->at(tilt_idx * organized->width + pan_idx) = pt;
    }

    return organized;
  }

private:
  void init_tilt_table(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud) {
    double eps = 1.0 * M_PI / 180.0;

    std::vector<std::pair<double, int>> hist;
    for(const auto& pt : cloud->points) {
      double dist = std::sqrt(pt.x * pt.x + pt.y * pt.y);
      double tilt = std::atan2(pt.z, dist);

      if(hist.empty()) {
        hist.push_back(std::make_pair(tilt, 1));
        continue;
      }

      bool voted = false;
      for(auto& bin : hist) {
        if(std::abs(bin.first - tilt) < eps) {
          bin.second++;
          voted = true;
          break;
        }
      }

      if(!voted) {
        hist.push_back(std::make_pair(tilt, 1));
      }
    }

    std::sort(hist.begin(), hist.end(), [=](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) { return lhs.first < rhs.first; });

    int th = hist[hist.size() / 2].second / 2;

    tilt_table.clear();
    for(const auto& bin : hist) {
      if(bin.second > th) {
        tilt_table.push_back(bin.first);
      }
    }
  }

private:
  std::vector<double> tilt_table;
};

}

#endif