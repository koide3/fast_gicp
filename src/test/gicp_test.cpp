#include <vector>
#include <sstream>
#include <iostream>
#include <gtest/gtest.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

struct GICPTestBase : public testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using PointCloudConstPtr = pcl::PointCloud<pcl::PointXYZ>::ConstPtr;

  GICPTestBase() {}

  virtual void SetUp() {
    if(!load(data_directory)) {
      exit(1);
    }
  }

  bool load(const std::string& data_directory) {
    relative_pose.setIdentity();

    std::ifstream ifs(data_directory + "/relative.txt");
    if(!ifs) {
      return false;
    }

    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 4; j++) {
        ifs >> relative_pose(i, j);
      }
    }

    auto target = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto source = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::io::loadPCDFile(data_directory + "/251370668.pcd", *target);
    pcl::io::loadPCDFile(data_directory + "/251371071.pcd", *source);
    if(target->empty() || source->empty()) {
      return true;
    }

    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.2, 0.2, 0.2);

    auto filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    voxelgrid.setInputCloud(target);
    voxelgrid.filter(*filtered);
    filtered.swap(target);

    voxelgrid.setInputCloud(source);
    voxelgrid.filter(*filtered);
    filtered.swap(source);

    this->target = target;
    this->source = source;

    return true;
  }

  Eigen::Vector2f pose_error(const Eigen::Matrix4f estimated) const {
    Eigen::Matrix4f delta = relative_pose.inverse() * estimated;
    double t_error = delta.block<3, 1>(0, 3).norm();
    double r_error = Eigen::AngleAxisf(delta.block<3, 3>(0, 0)).angle();
    return Eigen::Vector2f(t_error, r_error);
  }

  static std::string data_directory;

  PointCloudConstPtr target;
  PointCloudConstPtr source;
  Eigen::Matrix4f relative_pose;
};

std::string GICPTestBase::data_directory;

TEST_F(GICPTestBase, LoadCheck) {
  EXPECT_NE(target, nullptr);
  EXPECT_NE(source, nullptr);
  EXPECT_FALSE(target->empty());
  EXPECT_FALSE(source->empty());
}

using Parameters = std::tuple<const char*, bool>;
class AlignmentTest : public GICPTestBase, public testing::WithParamInterface<Parameters> {
public:
  pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr create_reg() {
    std::string method = std::get<0>(GetParam());
    int num_threads = std::get<1>(GetParam()) ? 4 : 1;

    if(method == "GICP") {
      auto gicp = pcl::make_shared<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>>();
      gicp->setNumThreads(num_threads);
      gicp->swapSourceAndTarget();
      return gicp;
    } else if(method == "VGICP") {
      auto vgicp = pcl::make_shared<fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>>();
      vgicp->setNumThreads(num_threads);
      return vgicp;
    } else if(method == "VGICP_CUDA") {
      return nullptr;
#ifdef USE_VGICP_CUDA
      auto vgicp = pcl::make_shared<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>>();
      return vgicp;
#endif
      return nullptr;
    }

    std::cerr << "unknown registration method:" << method << std::endl;
    return nullptr;
  }

  void swap_source_and_target(pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr reg) {
    fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>* lsq_reg = dynamic_cast<fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>*>(reg.get());
    if(lsq_reg != nullptr) {
      lsq_reg->swapSourceAndTarget();
      return;
    }

    std::cerr << "failed to swap source and target" << std::endl;
  }
};

INSTANTIATE_TEST_SUITE_P(AlignmentTest2, AlignmentTest, testing::Combine(testing::Values("GICP", "VGICP", "VGICP_CUDA"), testing::Bool()), [](const auto& info) {
  std::stringstream sst;
  sst << std::get<0>(info.param) << (std::get<1>(info.param) ? "_MT" : "_ST");
  return sst.str();
});

TEST_P(AlignmentTest, test) {
  const double t_tol = 0.05;
  const double r_tol = 1.0 * M_PI / 180.0;

  pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr reg = create_reg();
  if(reg == nullptr) {
    std::cout << "[          ] SKIP TEST" << std::endl;
    return;
  }

  // forward test
  auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  reg->setInputTarget(target);
  reg->setInputSource(source);
  reg->align(*aligned);

  Eigen::Vector2f errors = pose_error(reg->getFinalTransformation());
  EXPECT_LT(errors[0], t_tol) << "FORWARD TEST";
  EXPECT_LT(errors[1], r_tol) << "FORWARD TEST";
  EXPECT_TRUE(reg->hasConverged()) << "FORWARD TEST";

  // backward test
  reg->setInputTarget(source);
  reg->setInputSource(target);
  reg->align(*aligned);

  errors = pose_error(reg->getFinalTransformation().inverse());
  EXPECT_LT(errors[0], t_tol) << "BACKWARD TEST";
  EXPECT_LT(errors[1], r_tol) << "BACKWARD TEST";
  EXPECT_TRUE(reg->hasConverged()) << "BACKWARD TEST";

  // swap and set source
  reg = create_reg();
  reg->setInputSource(target);
  swap_source_and_target(reg);
  reg->setInputSource(source);
  reg->align(*aligned);

  errors = pose_error(reg->getFinalTransformation());
  EXPECT_LT(errors[0], t_tol) << "SWAP AND SET SOURCE TEST";
  EXPECT_LT(errors[1], r_tol) << "SWAP AND SET SOURCE TEST";
  EXPECT_TRUE(reg->hasConverged()) << "SWAP AND SET SOURCE TEST";

  // swap and set target
  reg = create_reg();
  reg->setInputTarget(source);
  swap_source_and_target(reg);  // source:target, target:source
  reg->setInputTarget(target);
  reg->align(*aligned);

  errors = pose_error(reg->getFinalTransformation());
  EXPECT_LT(errors[0], t_tol) << "SWAP AND SET TARGET TEST";
  EXPECT_LT(errors[1], r_tol) << "SWAP AND SET TARGET TEST";
  EXPECT_TRUE(reg->hasConverged()) << "SWAP AND SET TARGET TEST";
}

int main(int argc, char** argv) {
  GICPTestBase::data_directory = argv[1];
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
