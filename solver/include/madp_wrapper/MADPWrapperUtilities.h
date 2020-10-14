#ifndef MADPWRAPPERUTILITIES_H
#define MADPWRAPPERUTILITIES_H
#include "MADPWrapper.h"
#include "decpomdp/discrete/DecPOMDPFlat.hpp"
#include "decpomdp/discrete/DiscreteElements.hpp"
#include "decpomdp/discrete/DiscreteSpaces.hpp"
#include "statetransitionmodel/StateTransitionMatrix.hpp"
#include "observationmodel/ObservationMatrix.hpp"
#include "rewardmodel/RewardMatrix.hpp"
#include <vector>

// Tools for converting from wrapper to NPGI classes via intermediate representation
namespace npgi {
namespace madpwrapper {
StateTransitionMatrix<DiscreteJointAction<>, std::size_t, double>
state_transition_matrix(const MADPDecPOMDPDiscrete& d);

ObservationMatrix<DiscreteJointAction<>, DiscreteJointObservation<>, std::size_t, double>
observation_matrix(const MADPDecPOMDPDiscrete& d);

RewardMatrix<DiscreteJointAction<>, std::size_t, double>
reward_matrix(const MADPDecPOMDPDiscrete& d);

std::vector<double> initial_state_distribution(const MADPDecPOMDPDiscrete& d);

Agent<> get_agent(const MADPDecPOMDPDiscrete& d, std::size_t agent_index);

DiscreteLocalAction<> local_action(const MADPDecPOMDPDiscrete& d,
                                                 std::size_t agent_index, std::size_t action_index);
DiscreteLocalObservation<> local_observation(
    const MADPDecPOMDPDiscrete& d, std::size_t agent_index, std::size_t observation_index);

DiscreteLocalActionSpace<> local_action_space(const MADPDecPOMDPDiscrete& d,
                                              std::size_t agent_index);
DiscreteLocalObservationSpace<> local_observation_space(
    const MADPDecPOMDPDiscrete& d, std::size_t agent_index);

DiscreteJointActionSpace<> joint_action_space(const MADPDecPOMDPDiscrete& d);
DiscreteJointObservationSpace<> joint_observation_space(const MADPDecPOMDPDiscrete& d);

DecPOMDPFlat<std::size_t, std::size_t, std::size_t, double, int> to_flat_decpomdp(const MADPDecPOMDPDiscrete& d);

}  // namespace madpwrapper
}  // namespace npgi

#endif  // MADPWRAPPERUTILITIES_H