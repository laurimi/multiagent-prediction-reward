#ifndef JOINTBELIEFFLAT_HPP
#define JOINTBELIEFFLAT_HPP
#include <vector>
#include "decpomdp/base/JointBelief.hpp"
#include "utilities/EigenUtils.hpp"

namespace npgi {
template <typename State, typename Scalar>
class JointBeliefFlat;

template <typename State, typename Scalar>
struct joint_belief_traits<JointBeliefFlat<State, Scalar>> {
  using state_type = State;
  using scalar_type = Scalar;
  using pmf_type = std::vector<Scalar>;
};

template <typename State = std::size_t, typename Scalar = double>
struct JointBeliefFlat : public JointBelief<JointBeliefFlat<State, Scalar>> {
  using Derived = JointBeliefFlat<State, Scalar>;
  using state_type = typename joint_belief_traits<Derived>::state_type;
  using scalar_type = typename joint_belief_traits<Derived>::scalar_type;
  using pmf_type = typename joint_belief_traits<Derived>::pmf_type;

  JointBeliefFlat(const std::vector<Scalar>& pmf) : pmf_(pmf) {}
  JointBeliefFlat(const std::vector<Scalar>&& pmf) : pmf_(std::move(pmf)) {}

  state_type sample_state(const scalar_type random01) const {
    using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;
    Eigen::Map<const vector_type> b(pmf_.data(), pmf_.size());
    return detail::sample_from_pmf(b, random01);
  }
  scalar_type probability(const state_type& s) const { return pmf_.at(s); }
  scalar_type entropy() const {
    using vector_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;
    Eigen::Map<const vector_type> b(pmf_.data(), pmf_.size());
    return detail::entropy(b);
  }
  const pmf_type& pmf() const { return pmf_; }
  pmf_type& pmf() { return pmf_; }

 private:
  std::vector<Scalar> pmf_;
};

}  // namespace npgi

#endif  // JOINTBELIEFFLAT_HPP