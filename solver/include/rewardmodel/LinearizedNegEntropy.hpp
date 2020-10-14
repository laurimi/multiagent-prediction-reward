#ifndef LINEARIZEDNEGENTROPY_HPP
#define LINEARIZEDNEGENTROPY_HPP
#include "utilities/Common.hpp"
namespace npgi {
namespace negative_entropy {
template <typename Scalar>
std::vector<Scalar> gradient(std::vector<Scalar> x) {
  constexpr static Scalar emin =
      std::log(std::numeric_limits<Scalar>::epsilon());
  std::transform(x.begin(), x.end(), x.begin(), [](Scalar bi) {
    return (1.0 + (is_almost_zero(bi) ? emin : std::log(bi)));
  });
  return x;
}

template <typename Scalar>
Scalar fenchel_conjugate(const std::vector<Scalar>& grad) {
  (void)grad; // on the unit simplex, the fenchel conjugate (log sum exp) always equals 1.0, so grad is not needed
  return Scalar(1.0);
}

// Given a convex function f, its linearizing hyperplane at x is computed as
// \grad f(x) - f^*(\grad x),
// where f^* is the convex (Fenchel) conjugate function of f.
//
// The Fenchel conjugate of negative entropy is the log-sum-exp function.
// The log-sum-exp for vectors on the probability simplex always equals 1.
// Gradient of negative entropy is log(x) + 1.
//
// By simple algebra we derive that the +1 from the gradient and -1 from f^*
// cancel out. Here we still compute explicitly both gradient and f^*.
template <typename Scalar>
std::vector<Scalar> linearizing_hyperplane(const std::vector<Scalar>& b0) {
  std::vector<Scalar> g = gradient(b0);
  const Scalar fc = fenchel_conjugate(g);
  std::transform(g.begin(), g.end(), g.begin(),
                 [&fc](Scalar& ai) { return ai - fc; });

  return g;
}
}  // namespace negative_entropy
}  // namespace npgi
#endif  // LINEARIZEDNEGENTROPY_HPP