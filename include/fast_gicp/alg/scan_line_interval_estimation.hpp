#ifndef FAST_GICP_SCAN_LINE_INTERVAL_ESTIMATION_HPP
#define FAST_GICP_SCAN_LINE_INTERVAL_ESTIMATION_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace fast_gicp {

template<typename PointT>
class ScanLineIntervalEstimation {
public:
  ScanLineIntervalEstimation() {
    mean_tilt = -1;
  }

  void estimate(const boost::shared_ptr<const pcl::PointCloud<PointT>>& cloud) {
    init_tilt_table(cloud);

    std::vector<double> intervals(tilt_table.size() - 1);
    for(int i = 0; i < tilt_table.size() - 1; i++) {
      intervals[i] = tilt_table[i + 1] - tilt_table[i];
    }

    std::sort(tilt_table.begin(), tilt_table.end());
    mean_tilt = tilt_table[tilt_table.size() / 2];
  }

  size_t num_lines() const {
    return tilt_table.size();
  }

  double tilt(size_t i) const {
    return tilt_table[i];
  }

  double line_interval() const {
    return mean_tilt;
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

    std::sort(hist.begin(), hist.end(), [=](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) { return lhs.second < rhs.second; });

    for(const auto& bin : hist) {
      std::cout << bin.first << ":" << bin.second << std::endl;
    }

    int th = hist[hist.size() / 2].second / 4;

    tilt_table.clear();
    for(const auto& bin : hist) {
      if(bin.second > th) {
        tilt_table.push_back(bin.first);
      }
    }

    std::sort(hist.begin(), hist.end(), [=](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) { return lhs.first < rhs.first; });
  }

private:
  double mean_tilt;
  std::vector<double> tilt_table;
};

}  // namespace fast_gicp

#endif