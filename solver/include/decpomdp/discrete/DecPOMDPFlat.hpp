#ifndef DECPOMDPFLAT_HPP
#define DECPOMDPFLAT_HPP
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "JointActionSpaceFlat.hpp"
#include "JointBeliefFlat.hpp"
#include "JointObservationSpaceFlat.hpp"
#include "decpomdp/base/DecPOMDP.hpp"
#include "observationmodel/ObservationModelFlat.hpp"
#include "rewardmodel/RewardModelFlat.hpp"
#include "statetransitionmodel/StateTransitionModelFlat.hpp"

namespace npgi {
template <typename State, typename ActionIndex, typename ObservationIndex,
          typename Scalar, typename TimeStep>
class DecPOMDPFlat;

template <typename State, typename ActionIndex, typename ObservationIndex,
          typename Scalar, typename TimeStep>
struct decpomdp_traits<
    DecPOMDPFlat<State, ActionIndex, ObservationIndex, Scalar, TimeStep>> {
  using agent_type = Agent<ActionIndex>;
  using timestep_type = TimeStep;
  using scalar_type = Scalar;
  using state_type = State;
  using joint_belief_type = JointBeliefFlat<State, Scalar>;
  using joint_action_space_type =
      JointActionSpaceFlat<ActionIndex, timestep_type>;
  using joint_observation_space_type =
      JointObservationSpaceFlat<ObservationIndex, timestep_type>;
  using transition_model_type =
      StateTransitionModelFlat<State, ActionIndex, Scalar, timestep_type>;
  using observation_model_type =
      ObservationModelFlat<State, ActionIndex, ObservationIndex, Scalar,
                           timestep_type>;
  using reward_model_type =
      RewardModelFlat<State, ActionIndex, Scalar, timestep_type>;
};

template <typename State = std::size_t, typename ActionIndex = std::size_t,
          typename ObservationIndex = std::size_t, typename Scalar = double,
          typename TimeStep = int>
struct DecPOMDPFlat
    : public DecPOMDP<DecPOMDPFlat<State, ActionIndex, ObservationIndex, Scalar,
                                   TimeStep>> {
  using Derived = DecPOMDPFlat<State, ActionIndex, ObservationIndex, Scalar>;
  using state_type = typename decpomdp_traits<Derived>::state_type;
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

  DecPOMDPFlat(joint_action_space_type A, joint_observation_space_type Z,
               transition_model_type Tr, observation_model_type Ob,
               reward_model_type R)
      : DecPOMDP<Derived>(A, Z, Tr, Ob, R) {}
};
}  // namespace npgi
#endif  // DECPOMDPFLAT_HPP