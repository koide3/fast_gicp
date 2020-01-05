#ifndef GICP_DERIVATIVES_HPP
#define GICP_DERIVATIVES_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gicp {

/** 
 * f(x) = mean_B - (R * mean_A + t)
 * x = [r00, r10, ..., r22, t0, t1, t2]
 */
Eigen::Matrix<double, 3, 12> dtransform(const Eigen::Vector3d& mean_A, const Eigen::Vector3d& mean_B, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  Eigen::Matrix<double, 3, 12> J = Eigen::Matrix<double, 3, 12>::Zero();
  J.block<3, 3>(0, 0).diagonal().array() = -mean_A[0];
  J.block<3, 3>(0, 3).diagonal().array() = -mean_A[1];
  J.block<3, 3>(0, 6).diagonal().array() = -mean_A[2];
  J.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();
  return J;
}

/**
 * 
 */
void dRCR_impl(double* c, double* r, double* j) {
  j[0] = 2 * c[0] * r[0] + c[1] * r[1] + c[2] * r[2] + c[3] * r[1] + c[6] * r[2];
  j[1] = 0;
  j[2] = 0;
  j[3] = c[1] * r[0] + c[3] * r[0] + 2 * c[4] * r[1] + c[5] * r[2] + c[7] * r[2];
  j[4] = 0;
  j[5] = 0;
  j[6] = c[2] * r[0] + c[5] * r[1] + c[6] * r[0] + c[7] * r[1] + 2 * c[8] * r[2];
  j[7] = 0;
  j[8] = 0;
  j[9] = c[0] * r[3] + c[3] * r[4] + c[6] * r[5];
  j[10] = c[0] * r[0] + c[1] * r[1] + c[2] * r[2];
  j[11] = 0;
  j[12] = c[1] * r[3] + c[4] * r[4] + c[7] * r[5];
  j[13] = c[3] * r[0] + c[4] * r[1] + c[5] * r[2];
  j[14] = 0;
  j[15] = c[2] * r[3] + c[5] * r[4] + c[8] * r[5];
  j[16] = c[6] * r[0] + c[7] * r[1] + c[8] * r[2];
  j[17] = 0;
  j[18] = c[0] * r[6] + c[3] * r[7] + c[6] * r[8];
  j[19] = 0;
  j[20] = c[0] * r[0] + c[1] * r[1] + c[2] * r[2];
  j[21] = c[1] * r[6] + c[4] * r[7] + c[7] * r[8];
  j[22] = 0;
  j[23] = c[3] * r[0] + c[4] * r[1] + c[5] * r[2];
  j[24] = c[2] * r[6] + c[5] * r[7] + c[8] * r[8];
  j[25] = 0;
  j[26] = c[6] * r[0] + c[7] * r[1] + c[8] * r[2];
  j[27] = c[0] * r[3] + c[1] * r[4] + c[2] * r[5];
  j[28] = c[0] * r[0] + c[3] * r[1] + c[6] * r[2];
  j[29] = 0;
  j[30] = c[3] * r[3] + c[4] * r[4] + c[5] * r[5];
  j[31] = c[1] * r[0] + c[4] * r[1] + c[7] * r[2];
  j[32] = 0;
  j[33] = c[6] * r[3] + c[7] * r[4] + c[8] * r[5];
  j[34] = c[2] * r[0] + c[5] * r[1] + c[8] * r[2];
  j[35] = 0;
  j[36] = 0;
  j[37] = 2 * c[0] * r[3] + c[1] * r[4] + c[2] * r[5] + c[3] * r[4] + c[6] * r[5];
  j[38] = 0;
  j[39] = 0;
  j[40] = c[1] * r[3] + c[3] * r[3] + 2 * c[4] * r[4] + c[5] * r[5] + c[7] * r[5];
  j[41] = 0;
  j[42] = 0;
  j[43] = c[2] * r[3] + c[5] * r[4] + c[6] * r[3] + c[7] * r[4] + 2 * c[8] * r[5];
  j[44] = 0;
  j[45] = 0;
  j[46] = c[0] * r[6] + c[3] * r[7] + c[6] * r[8];
  j[47] = c[0] * r[3] + c[1] * r[4] + c[2] * r[5];
  j[48] = 0;
  j[49] = c[1] * r[6] + c[4] * r[7] + c[7] * r[8];
  j[50] = c[3] * r[3] + c[4] * r[4] + c[5] * r[5];
  j[51] = 0;
  j[52] = c[2] * r[6] + c[5] * r[7] + c[8] * r[8];
  j[53] = c[6] * r[3] + c[7] * r[4] + c[8] * r[5];
  j[54] = c[0] * r[6] + c[1] * r[7] + c[2] * r[8];
  j[55] = 0;
  j[56] = c[0] * r[0] + c[3] * r[1] + c[6] * r[2];
  j[57] = c[3] * r[6] + c[4] * r[7] + c[5] * r[8];
  j[58] = 0;
  j[59] = c[1] * r[0] + c[4] * r[1] + c[7] * r[2];
  j[60] = c[6] * r[6] + c[7] * r[7] + c[8] * r[8];
  j[61] = 0;
  j[62] = c[2] * r[0] + c[5] * r[1] + c[8] * r[2];
  j[63] = 0;
  j[64] = c[0] * r[6] + c[1] * r[7] + c[2] * r[8];
  j[65] = c[0] * r[3] + c[3] * r[4] + c[6] * r[5];
  j[66] = 0;
  j[67] = c[3] * r[6] + c[4] * r[7] + c[5] * r[8];
  j[68] = c[1] * r[3] + c[4] * r[4] + c[7] * r[5];
  j[69] = 0;
  j[70] = c[6] * r[6] + c[7] * r[7] + c[8] * r[8];
  j[71] = c[2] * r[3] + c[5] * r[4] + c[8] * r[5];
  j[72] = 0;
  j[73] = 0;
  j[74] = 2 * c[0] * r[6] + c[1] * r[7] + c[2] * r[8] + c[3] * r[7] + c[6] * r[8];
  j[75] = 0;
  j[76] = 0;
  j[77] = c[1] * r[6] + c[3] * r[6] + 2 * c[4] * r[7] + c[5] * r[8] + c[7] * r[8];
  j[78] = 0;
  j[79] = 0;
  j[80] = c[2] * r[6] + c[5] * r[7] + c[6] * r[6] + c[7] * r[7] + 2 * c[8] * r[8];
}

/**
 * f(x) = R * C * R.transpose();
 * x = [r00, r10, ..., r22]
 */
Eigen::Matrix<double, 9, 9> dRCR(const Eigen::Matrix3d& R, const Eigen::Matrix3d& C) {
  Eigen::Matrix<double, 9, 9> J = Eigen::Matrix<double, 9, 9>::Zero();

  Eigen::Matrix3d R_ = R.transpose();
  Eigen::Matrix3d C_ = C.transpose();

  dRCR_impl(C_.data(), R_.data(), J.data());

  return J.transpose();
}

/**
 * f(x) = |C| 
 * x = [c00, c10, ..., c22]
 */
Eigen::Matrix<double, 1, 9> dmat_det(const Eigen::Matrix3d& C) {
  Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
  J(0, 0) = C(1, 1) * C(2, 2) - C(1, 2) * C(2, 1);
  J(0, 1) = C(1, 2) * C(2, 0) - C(1, 0) * C(2, 2);
  J(0, 2) = C(1, 0) * C(2, 1) - C(1, 1) * C(2, 0);

  J(1, 0) = C(2, 1) * C(0, 2) - C(0, 1) * C(2, 2);
  J(1, 1) = C(0, 0) * C(2, 2) - C(0, 2) * C(2, 0);
  J(1, 2) = C(0, 1) * C(2, 0) - C(2, 1) * C(0, 0);

  J(2, 0) = C(0, 1) * C(1, 2) - C(0, 2) * C(1, 1);
  J(2, 1) = C(1, 0) * C(0, 2) - C(1, 2) * C(0, 0);
  J(2, 2) = C(0, 0) * C(1, 1) - C(0, 1) * C(1, 0);
  return Eigen::Map<Eigen::Matrix<double, 1, 9>>(J.data());
}

/**
 * f(x) = C.inverse();
 * x = [c00, c10, ..., c22]
 */
Eigen::Matrix<double, 9, 9> dmat_inv(const Eigen::Matrix3d& C) {
  double det = C.determinant();
  double inv_det = 1.0 / det;
  double inv_det_sq = 1.0 / (det * det);
  Eigen::Matrix<double, 1, 9> jdet = dmat_det(C);

  Eigen::Matrix3d jdet_mat = Eigen::Map<Eigen::Matrix3d>(jdet.data()).transpose();
  Eigen::Matrix<double, 1, 9> jdet_trans = Eigen::Map<Eigen::Matrix<double, 1, 9>>(jdet_mat.data());

  // matrix index
  // 0 3 6
  // 1 4 7
  // 2 5 8

  // (0, 0) -> 0
  // (0, 1) -> 3
  // (0, 2) -> 6

  // (1, 0) -> 1
  // (1, 1) -> 4
  // (1, 2) -> 7

  // (2, 0) -> 2
  // (2, 1) -> 5
  // (2, 2) -> 8

  Eigen::Matrix<double, 9, 9> J = -jdet_trans.transpose() * jdet * inv_det_sq;
  // (0, 0)
  // J.row(0) = -jdet * jdet(0) * inv_det_sq;
  Eigen::Vector4d sub00 = Eigen::Vector4d(C(2, 2), -C(1, 2), -C(2, 1), C(1, 1)) * det * inv_det_sq;
  J(0, 4) += sub00[0];
  J(0, 5) += sub00[1];
  J(0, 7) += sub00[2];
  J(0, 8) += sub00[3];

  // (0, 1)
  Eigen::Vector4d sub01 = Eigen::Vector4d(-C(1, 0), -C(2, 2), C(1, 2), C(2, 0)) * det * inv_det_sq;
  J(1, 8) += sub01[0];
  J(1, 1) += sub01[1];
  J(1, 2) += sub01[2];
  J(1, 7) += sub01[3];

  // (0, 2)
  Eigen::Vector4d sub02 = Eigen::Vector4d(C(1, 0), C(2, 1), -C(1, 1), -C(2, 0)) * det * inv_det_sq;
  J(2, 5) += sub02[0];
  J(2, 1) += sub02[1];
  J(2, 2) += sub02[2];
  J(2, 4) += sub02[3];

  // (1, 0)
  Eigen::Vector4d sub10 = Eigen::Vector4d(-C(0, 1), -C(2, 2), C(0, 2), C(2, 1)) * det * inv_det_sq;
  J(3, 8) += sub10[0];
  J(3, 3) += sub10[1];
  J(3, 5) += sub10[2];
  J(3, 6) += sub10[3];

  // (1, 1)
  Eigen::Vector4d sub11 = Eigen::Vector4d(C(0, 0), C(2, 2), -C(0, 2), -C(2, 0)) * det * inv_det_sq;
  J(4, 8) += sub11[0];
  J(4, 0) += sub11[1];
  J(4, 2) += sub11[2];
  J(4, 6) += sub11[3];

  // (1, 2)
  Eigen::Vector4d sub12 = Eigen::Vector4d(-C(0, 0), -C(2, 1), C(0, 1), C(2, 0)) * det * inv_det_sq;
  J(5, 5) += sub12[0];
  J(5, 0) += sub12[1];
  J(5, 2) += sub12[2];
  J(5, 3) += sub12[3];

  // (2, 0)
  Eigen::Vector4d sub20 = Eigen::Vector4d(C(0, 1), C(1, 2), -C(0, 2), -C(1, 1)) * det * inv_det_sq;
  J(6, 7) += sub20[0];
  J(6, 3) += sub20[1];
  J(6, 4) += sub20[2];
  J(6, 6) += sub20[3];

  // (2, 1)
  Eigen::Vector4d sub21 = Eigen::Vector4d(-C(0, 0), -C(1, 2), C(0, 2), C(1, 0)) * det * inv_det_sq;
  J(7, 7) += sub21[0];
  J(7, 0) += sub21[1];
  J(7, 1) += sub21[2];
  J(7, 6) += sub21[3];

  // (2, 2)
  Eigen::Vector4d sub22 = Eigen::Vector4d(C(0, 0), C(1, 1), -C(0, 1), -C(1, 0)) * det * inv_det_sq;
  J(8, 4) += sub22[0];
  J(8, 0) += sub22[1];
  J(8, 1) += sub22[2];
  J(8, 3) += sub22[3];

  return J;
}

}

#endif