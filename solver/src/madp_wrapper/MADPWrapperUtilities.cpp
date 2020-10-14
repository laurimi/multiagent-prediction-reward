#include "madp_wrapper/MADPWrapperUtilities.h"
#include <numeric>
#include <map>

namespace npgi {
namespace madpwrapper {

StateTransitionMatrix<DiscreteJointAction<>, std::size_t, double>
state_transition_matrix(const MADPDecPOMDPDiscrete& d) {
  using matrix_type = StateTransitionMatrix<DiscreteJointAction<>, std::size_t, double>::matrix_type;
  std::map<DiscreteJointAction<>, matrix_type> T;
  for (std::size_t a = 0; a < d.num_joint_actions(); ++a)
  {
    matrix_type M(d.num_states(), d.num_states());
    for (std::size_t snew = 0; snew < d.num_states(); ++snew)
      for (std::size_t sold = 0; sold < d.num_states(); ++sold)
        M(snew, sold) = d.transition_probability(snew, sold, a);
    T.emplace(a, M);
  }

  return StateTransitionMatrix<DiscreteJointAction<>, std::size_t, double>(T);
}

ObservationMatrix<DiscreteJointAction<>, DiscreteJointObservation<>, std::size_t, double>
observation_matrix(const MADPDecPOMDPDiscrete& d)
{
  using matrix_type =
      ObservationMatrix<DiscreteJointAction<>, DiscreteJointObservation<>,
                        std::size_t, double>::matrix_type;
  std::map<DiscreteJointAction<>, matrix_type> O;
  for (std::size_t a = 0; a < d.num_joint_actions(); ++a) {
    matrix_type M(d.num_joint_observations(), d.num_states());
    for (std::size_t o = 0; o < d.num_joint_observations(); ++o)
      for (std::size_t s = 0; s < d.num_states(); ++s)
        M(o, s) = d.observation_probability(o, s, a);
    O.emplace(a, M);
  }
  return ObservationMatrix<DiscreteJointAction<>, DiscreteJointObservation<>, std::size_t, double>(O);
}

RewardMatrix<DiscreteJointAction<>, std::size_t, double>
reward_matrix(const MADPDecPOMDPDiscrete& d)
{
  using vector_type = RewardMatrix<DiscreteJointAction<>, std::size_t, double>::vector_type;

  std::map<DiscreteJointAction<>, vector_type> R;
  for (std::size_t a = 0; a < d.num_joint_actions(); ++a) {
    vector_type v(d.num_states());
    for (std::size_t s = 0; s < d.num_states(); ++s) v(s) = d.reward(s, a);
    R.emplace(a, v);
  }
  return RewardMatrix<DiscreteJointAction<>, std::size_t, double>(R);
}

std::vector<double> initial_state_distribution(const MADPDecPOMDPDiscrete& d) {
  std::vector<double> isd(d.num_states(), 0.0);
  for (std::size_t s = 0; s < d.num_states(); ++s)
    isd[s] = d.initial_belief_at(s);
  return isd;
}

Agent<> get_agent(const MADPDecPOMDPDiscrete& d, std::size_t agent_index)
{
  return Agent<>({agent_index, d.agent_name(agent_index)});
}

DiscreteLocalAction<> local_action(const MADPDecPOMDPDiscrete& d,
                                    std::size_t agent_index,
                                    std::size_t action_index) {
  return DiscreteLocalAction<>(
      {action_index, d.action_name(agent_index, action_index)});
}

DiscreteLocalObservation<> local_observation(const MADPDecPOMDPDiscrete& d,
                                              std::size_t agent_index,
                                              std::size_t observation_index) {
  return DiscreteLocalObservation<>(
      {observation_index, d.observation_name(agent_index, observation_index)});
}

DiscreteLocalActionSpace<> local_action_space(const MADPDecPOMDPDiscrete& d,
                                              std::size_t agent_index) {
  DiscreteLocalActionSpace<> da;
  for (std::size_t i = 0; i < d.num_actions(agent_index); ++i)
    da.insert(i, local_action(d, agent_index, i));
  return da;
}

DiscreteLocalObservationSpace<> local_observation_space(const MADPDecPOMDPDiscrete& d,
                                              std::size_t agent_index) {
  DiscreteLocalObservationSpace<> da;
  for (std::size_t i = 0; i < d.num_observations(agent_index); ++i)
    da.insert(i, local_observation(d, agent_index, i));
  return da;
}

DiscreteJointActionSpace<> joint_action_space(const MADPDecPOMDPDiscrete& d)
{
  std::map<Agent<>, DiscreteLocalActionSpace<>> local_spaces;
  for (std::size_t agent_index = 0; agent_index < d.num_agents(); ++agent_index)
    local_spaces.emplace(get_agent(d, agent_index), local_action_space(d, agent_index) );

  return DiscreteJointActionSpace<>(local_spaces);
}

DiscreteJointObservationSpace<> joint_observation_space(const MADPDecPOMDPDiscrete& d)
{
    std::map<Agent<>, DiscreteLocalObservationSpace<>> local_spaces;
  for (std::size_t agent_index = 0; agent_index < d.num_agents(); ++agent_index)
    local_spaces.emplace(get_agent(d, agent_index), local_observation_space(d, agent_index) );

  return DiscreteJointObservationSpace<>(local_spaces);
}

DecPOMDPFlat<std::size_t, std::size_t, std::size_t, double, int> to_flat_decpomdp(const MADPDecPOMDPDiscrete& d) {
  auto A = joint_action_space(d);
  auto Z = joint_observation_space(d);
  auto t = state_transition_matrix(d);
  auto o = npgi::madpwrapper::observation_matrix(d);
  RewardModelFlat r(npgi::madpwrapper::reward_matrix(d), d.discount());
  return DecPOMDPFlat<std::size_t, std::size_t, std::size_t, double, int>(A, Z, t, o, r);
}

}  // namespace madpwrapper
}  // namespace npgi