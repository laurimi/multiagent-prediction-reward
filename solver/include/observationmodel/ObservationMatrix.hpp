#ifndef OBSERVATIONMATRIX_HPP
#define OBSERVATIONMATRIX_HPP
#include <eigen3/Eigen/Dense>
#include <map>
#include "utilities/EigenUtils.hpp"

namespace npgi {
template <typename Action, typename Observation, typename StateIndex,
          typename Scalar>
class ObservationMatrix {
 public:
  using pmf_type = std::vector<Scalar>;
  using matrix_type = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  ObservationMatrix(const std::map<Action, matrix_type>& O) : O_(O) {}

  Scalar observation_probability(const Observation& o, const StateIndex& s,
                                 const Action& a) const {
    return O_.at(a.index())(o.index(), s);
  }

  Observation sample_observation(const StateIndex& s, const Action& a,
                                 const Scalar random01) const {
    return detail::sample_from_pmf(O_.at(a.index()).col(s), random01);
  }

  Scalar update(pmf_type& b, const Action& a, const Observation& o) const {
    using vector_type = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    Eigen::Map<vector_type> bv(b.data(), b.size());
    return detail::bayes_update(bv,
                                O_.at(a.index()).row(o.index()).transpose());
  }

 private:
  std::map<Action, matrix_type> O_;
};
}  // namespace npgi
#endif  // OBSERVATIONMATRIX_HPP