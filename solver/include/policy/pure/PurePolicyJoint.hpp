#ifndef PUREPOLICYJOINT_HPP
#define PUREPOLICYJOINT_HPP
#include "PurePolicy.hpp"
#include "PurePolicyJointExecutionState.hpp"

namespace npgi {
namespace policy {

template <typename Index, typename TimeStep>
class PurePolicyJoint;

template <typename Index, typename TimeStep>
struct policy_traits<PurePolicyJoint<Index, TimeStep>> {
  using timestep_type = TimeStep;
  using joint_action_type = DiscreteJointAction<Index>;
  using local_action_type = DiscreteLocalAction<Index>;
  using joint_observation_type = DiscreteJointObservation<Index>;
  using local_observation_type = DiscreteLocalObservation<Index>;
  using execution_state_type =
      policy::PurePolicyJointExecutionState<Index, TimeStep>;
};

template <typename Index = std::size_t, typename TimeStep = int>
class PurePolicyJoint : public Policy<PurePolicyJoint<Index, TimeStep>> {
 public:
  using base_type = Policy<PurePolicyJoint<Index, TimeStep>>;
  using derived_type = PurePolicyJoint<Index, TimeStep>;
  using trait_type = policy_traits<derived_type>;

  using timestep_type = typename trait_type::timestep_type;
  using execution_state_type = typename trait_type::execution_state_type;
  using joint_action_type = typename trait_type::joint_action_type;
  using joint_observation_type = typename trait_type::joint_observation_type;

  using agent_type = Agent<Index>;
  using joint_action_space_type = JointActionSpaceFlat<Index, TimeStep>;
  using joint_observation_space_type =
      JointObservationSpaceFlat<Index, TimeStep>;

  using local_action_type = typename joint_action_space_type::local_action_type;
  using local_observation_type =
      typename joint_observation_space_type::local_observation_type;

  using local_policy_type =
      PurePolicy<local_action_type, local_observation_type>;

  PurePolicyJoint(timestep_type tstart, timestep_type tlast,
                  const std::map<agent_type, local_policy_type>& local,
                  const joint_action_space_type& A,
                  const joint_observation_space_type& Z)
      : base_type(tstart, tlast), local_(local), A_(A), Z_(Z) {}

  execution_state_type initial_execution_state() const {
    return execution_state_type(this->start_time(), local_);
  }

  execution_state_type next_execution_state(
      const execution_state_type& e, const joint_action_type& a,
      const joint_observation_type& o) const {
    execution_state_type e_next(e);
    update_execution_state(e_next, a, o);
    return e_next;
  }

  void update_execution_state(execution_state_type& e,
                              const joint_action_type& a,
                              const joint_observation_type& o) const {
    for (const auto& [agent, localpolicy] : local_) {
      localpolicy.update_execution_state(
          e.local_exec_states_.at(agent),
          A_.get_local_action(a, e.time_, agent.index()),
          Z_.get_local_observation(o, e.time_ + 1, agent.index()));
    }
    ++e.time_;
  }

  joint_action_type get_joint_action(const execution_state_type& e) const {
    std::map<agent_type, local_action_type> local_actions;
    for (const auto& [agent, fsc] : local_) {
      local_actions[agent] =
          local_.at(agent).get_action(e.local_exec_states_.at(agent));
    }
    return A_.get_joint_action(local_actions, e.time_step());
  }

 private:
  std::map<agent_type, local_policy_type> local_;
  const joint_action_space_type& A_;
  const joint_observation_space_type& Z_;
};
}  // namespace policy
}  // namespace npgi

#endif  // PUREPOLICYJOINT_HPP