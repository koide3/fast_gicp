#ifndef KKL_OPT_TERMINATION_CRITERIA_HPP
#define KKL_OPT_TERMINATION_CRITERIA_HPP

namespace kkl {
namespace opt {

struct TerminationCriteria {
public:
  TerminationCriteria(int max_iterations = 1024, double eps = 1e-6) : max_iterations(max_iterations), eps(eps) {}

  int max_iterations;
  double eps;
};

}  // namespace opt
}  // namespace kkl

#endif