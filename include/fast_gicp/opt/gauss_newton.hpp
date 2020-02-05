#ifndef FAST_GICP_GAUSS_NEWTON_HPP
#define FAST_GICP_GAUSS_NEWTON_HPP

namespace fast_gicp {

template<typename Scalar, int N>
class GaussNewton {
public:
  GaussNewton() {}
  virtual ~GaussNewton() {}

  virtual Eigen::Matrix<Scalar, N, 1> delta(const Eigen::Matrix<Scalar, -1, 1>& e, const Eigen::Matrix<Scalar, -1, -1>& J) const {
    Eigen::Matrix<Scalar, N, N> JJ = J.transpose() * J;
    Eigen::Matrix<Scalar, N, 1> delta = JJ.llt().solve(J.transpose() * e);

    if(!delta.array().isFinite().all()) {
      // std::cerr << "!!!! delta corrupted !!!!" << std::endl;
      return Eigen::Matrix<Scalar, N, 1>::Random() * 1e-3;
    }

    return delta;
  }
};


// WIP
template<typename Scalar, int N>
class LevenbergMarquardt : public GaussNewton<Scalar, N> {
public:
  LevenbergMarquardt(double init_lambda=10.0)
  : lambda(init_lambda)
  {}
  virtual ~LevenbergMarquardt() override {}

  virtual Eigen::Matrix<Scalar, N, 1> delta(const Eigen::Matrix<Scalar, -1, 1>& e, const Eigen::Matrix<Scalar, -1, -1>& J) const override {
    Eigen::Matrix<Scalar, N, N> JJ = J.transpose() * J;

    for(int i = 0; i<N; i++) {
      JJ(i, i) = JJ(i, i) + lambda * J(i, i);
    }

    Eigen::Matrix<Scalar, N, 1> delta = JJ.llt().solve(J.transpose() * e);
    return delta;
  }

private:
  double lambda;
};
}

#endif