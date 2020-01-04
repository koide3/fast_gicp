#ifndef KKL_OPT_LINE_SEARCH_HPP
#define KKL_OPT_LINE_SEARCH_HPP

#include <functional>
#include <Eigen/Core>

#include <kkl/opt/termination_criteria.hpp>

namespace kkl {
namespace opt {

template <typename Scalar>
class LineSearch {
public:
  LineSearch(const std::function<Scalar(Scalar)>& f) : function(f) {}
  virtual ~LineSearch() {}

  virtual Scalar minimize(Scalar x0, Scalar x1, const TerminationCriteria& criteria) = 0;

protected:
  TerminationCriteria criteria;
  std::function<Scalar(Scalar)> function;
};

}  // namespace opt
}  // namespace kkl

#endif