#ifndef KKL_OPT_BACKTRACKING_SEARCH_HPP
#define KKL_OPT_BACKTRACKING_SEARCH_HPP

#include <kkl/opt/line_search.hpp>

namespace kkl {
namespace opt {

template<typename Scalar>
class BackTrackingSearch : public LineSearch<Scalar> {
public:
  BackTrackingSearch(const std::function<Scalar(Scalar)>& f, Scalar f0, Scalar gnorm) : LineSearch<Scalar>(f), f0(f0), gnorm(gnorm) {}
  virtual ~BackTrackingSearch() override {}

  virtual Scalar minimize(Scalar a, Scalar b, const TerminationCriteria& criteria = TerminationCriteria()) override {
    Scalar x = b;
    Scalar rho = 0.5;
    Scalar c = 1e-3;
    Scalar cgnorm = c * gnorm;

    while(this->function(x) > f0 + x * c * gnorm) {
      x *= rho;
    }

    return x;
  }

private:
  Scalar f0;
  Scalar gnorm;
};

}  // namespace opt
}  // namespace kkl

#endif