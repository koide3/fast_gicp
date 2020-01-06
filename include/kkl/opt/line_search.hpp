#ifndef KKL_OPT_LINE_SEARCH_HPP
#define KKL_OPT_LINE_SEARCH_HPP

#include <functional>
#include <Eigen/Core>

#include <kkl/opt/termination_criteria.hpp>

namespace kkl {
namespace opt {

class LineSearch {
public:
  LineSearch(const std::function<double(double)>& f) : function(f) {}
  virtual ~LineSearch() {}

  virtual double minimize(double x0, double x1, const TerminationCriteria& criteria) = 0;

protected:
  TerminationCriteria criteria;
  std::function<double(double)> function;
};

}  // namespace opt
}  // namespace kkl

#endif