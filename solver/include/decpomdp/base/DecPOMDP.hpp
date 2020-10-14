#ifndef DECPOMDP_HPP
#define DECPOMDP_HPP
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "core/CRTPHelper.hpp"

namespace npgi {
template <typename Derived>
struct decpomdp_traits;

template <typename Derived>
struct DecPOMDP : crtp_helper<Derived> {
  using agent_type = typename decpomdp_traits<Derived>::agent_type;
  using timestep_type = typename decpomdp_traits<Derived>::timestep_type;
  using scalar_type = typename decpomdp_traits<Derived>::scalar_type;
  using state_type = typename decpomdp_traits<Derived>::state_type;
  using joint_belief_type = typename decpomdp_traits<Derived>::joint_belief_type;

  // model components
  using joint_action_space_type =
      typename decpomdp_traits<Derived>::joint_action_space_type;
  using joint_observation_space_type =
      typename decpomdp_traits<Derived>::joint_observation_space_type;
  using transition_model_type =
      typename decpomdp_traits<Derived>::transition_model_type;
  using observation_model_type =
      typename decpomdp_traits<Derived>::observation_model_type;
  using reward_model_type =
      typename decpomdp_traits<Derived>::reward_model_type;

  // actions and observations
  using joint_action_type = typename joint_action_space_type::joint_action_type;
  using joint_observation_type =
      typename joint_observation_space_type::joint_observation_type;

  DecPOMDP(joint_action_space_type A, joint_observation_space_type Z,
           transition_model_type Tr, observation_model_type Ob,
           reward_model_type R)
      : A_(A), Z_(Z), Tr_(Tr), Ob_(Ob), R_(R) {}

  const joint_action_space_type& A() const { return A_; }
  const joint_observation_space_type& Z() const { return Z_; }
  const transition_model_type& Tr() const { return Tr_; }
  const observation_model_type& Ob() const { return Ob_; }
  const reward_model_type& R() const { return R_; }

 private:
  joint_action_space_type A_;
  joint_observation_space_type Z_;
  transition_model_type Tr_;
  observation_model_type Ob_;
  reward_model_type R_;
};
}  // namespace npgi
#endif  // DECPOMDP_HPP