#ifndef JOINTBELIEF_HPP
#define JOINTBELIEF_HPP
#include <utility>
#include "core/CRTPHelper.hpp"
#include "observationmodel/ObservationModel.hpp"
#include "rewardmodel/RewardModel.hpp"
#include "statetransitionmodel/StateTransitionModel.hpp"
namespace npgi {

template <typename Derived>
struct joint_belief_traits;

template <typename Derived>
struct JointBelief : crtp_helper<Derived> {
  using state_type = typename joint_belief_traits<Derived>::state_type;
  using scalar_type = typename joint_belief_traits<Derived>::scalar_type;
  using pmf_type = typename joint_belief_traits<Derived>::pmf_type;

  state_type sample_state(const scalar_type random01) const {
    return this->underlying().sample_state(random01);
  }
  scalar_type probability(const state_type& s) const {
    return this->underlying().probability(s);
  }
  scalar_type entropy() const { return this->underlying().entropy(); }

  const pmf_type& pmf() const { return this->underlying().pmf(); }
  pmf_type& pmf() { return this->underlying().pmf(); }
};
}  // namespace npgi

#endif  // JOINTBELIEF_HPP