#ifndef STATETRANSITIONMATRIX_HPP
#define STATETRANSITIONMATRIX_HPP
#include <eigen3/Eigen/Dense>
#include <map>
#include "utilities/EigenUtils.hpp"

namespace npgi {

template <typename Action, typename StateIndex, typename Scalar>
class StateTransitionMatrix {
 public:
  using pmf_type = std::vector<Scalar>;
  using matrix_type = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  StateTransitionMatrix(const std::map<Action, matrix_type>& T) : T_(T) {}

  Scalar transition_probability(const StateIndex& next, const StateIndex& old,
                                const Action& a) const {
    return T_.at(a.index())(next, old);
  }

  StateIndex sample_next_state(const StateIndex& old, const Action& a,
                               const Scalar random01) const {
    return detail::sample_from_pmf(T_.at(a.index()).col(old), random01);
  }

  pmf_type predict(const pmf_type& b, const Action& a) const {
    using vector_type = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    pmf_type b_next(b);
    Eigen::Map<vector_type> bv(b_next.data(), b_next.size());
    bv = T_.at(a.index()) * bv;
    return b_next;
  }

  std::size_t rows() const { return T_.begin()->second.rows(); }

 private:
  std::map<Action, matrix_type> T_;
};
}  // namespace npgi

#endif  // STATETRANSITIONMATRIX_HPP