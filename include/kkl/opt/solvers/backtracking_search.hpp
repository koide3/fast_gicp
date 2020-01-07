#ifndef KKL_OPT_BACKTRACKING_SEARCH_HPP
#define KKL_OPT_BACKTRACKING_SEARCH_HPP

#include <kkl/opt/line_search.hpp>

namespace kkl {
namespace opt {

class BackTrackingSearch : public LineSearch {
public:
  BackTrackingSearch(const std::function<double(double)>& f, double f0, double gnorm) : LineSearch(f), f0(f0), gnorm(gnorm) {}
  virtual ~BackTrackingSearch() override {}

  virtual double minimize(double a, double b, const TerminationCriteria& criteria = TerminationCriteria()) override {
    double alpha = b;
    double rho = 0.5;
    double c = criteria.eps;
    double cgnorm = c * gnorm;

    while(this->function(alpha) > f0 + c * alpha * gnorm) {
      alpha *= rho;
    }

    return alpha;
  }

private:
  double f0;
  double gnorm;
};

}  // namespace opt
}  // namespace kkl

#endif