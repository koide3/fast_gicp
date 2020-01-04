#ifndef KKL_OPT_GOLDEN_SECTION_SEARCH_HPP
#define KKL_OPT_GOLDEN_SECTION_SEARCH_HPP

#include <kkl/opt/line_search.hpp>

namespace kkl {
namespace opt {

template <typename Scalar>
class GoldenSectionSearch : public LineSearch<Scalar> {
public:
  GoldenSectionSearch(const std::function<Scalar(Scalar)>& f) : LineSearch<Scalar>(f) {}
  virtual ~GoldenSectionSearch() override {}

  virtual Scalar minimize(Scalar a, Scalar b, const TerminationCriteria& criteria = TerminationCriteria()) override {
    const double gr = (1 + std::sqrt(5.0)) / 2.0;
    Scalar c = b - (b - a) / gr;
    Scalar d = a + (b - a) / gr;

    for (int i = 0; i < this->criteria.max_iterations; i++) {
      if (std::abs(c - d) < this->criteria.eps) {
        break;
      }

      if (this->function(c) < this->function(d)) {
        b = d;
      } else {
        a = c;
      }

      c = b - (b - a) / gr;
      d = a + (b - a) / gr;
    }

    return (b + a) / 2;
  }
};

}  // namespace opt
}  // namespace kkl

#endif