#ifndef PUREPOLICYJOINTEXECUTIONSTATE_HPP
#define PUREPOLICYJOINTEXECUTIONSTATE_HPP
#include "PurePolicy.hpp"
#include "policy/base/ExecutionState.hpp"
namespace npgi {
namespace policy {

template <typename Index, typename TimeStep>
struct PurePolicyJointExecutionState;

template <typename Index, typename TimeStep>
std::ostream& operator<<(std::ostream&,
                         const PurePolicyJointExecutionState<Index, TimeStep>&);

template <typename Index, typename TimeStep>
struct execution_state_traits<PurePolicyJointExecutionState<Index, TimeStep>> {
  using timestep_type = TimeStep;
  using joint_observation_type = DiscreteJointObservation<Index>;
  using local_observation_type = DiscreteLocalObservation<Index>;
  using joint_action_type = DiscreteJointAction<Index>;
  using local_action_type = DiscreteLocalAction<Index>;
  using local_policy_type =
      PurePolicy<local_action_type, local_observation_type>;
  using local_policy_execution_state_type =
      PurePolicyExecutionState<local_observation_type>;
};

template <typename Index, typename TimeStep>
struct PurePolicyJointExecutionState
    : public ExecutionState<PurePolicyJointExecutionState<Index, TimeStep>> {
  using derived_type = PurePolicyJointExecutionState<Index, TimeStep>;
  using base_type = ExecutionState<derived_type>;
  using trait_type = execution_state_traits<derived_type>;
  using timestep_type = typename trait_type::timestep_type;
  using local_policy_type = typename trait_type::local_policy_type;
  using local_policy_execution_state_type =
      typename trait_type::local_policy_execution_state_type;

  using agent_type = Agent<Index>;

  PurePolicyJointExecutionState(
      timestep_type time,
      const std::map<agent_type, local_policy_type>& local_policies)
      : time_(time), local_exec_states_([&local_policies]() {
          std::map<agent_type, local_policy_execution_state_type> m;
          for (const auto& [agent, local] : local_policies) {
            m.emplace(agent, local.initial_execution_state());
          }
          return m;
        }()) {}

  bool operator<(const PurePolicyJointExecutionState& other) const {
    if (time_ != other.time_)
      return (time_ < other.time_);
    else
      return (local_exec_states_ < other.local_exec_states_);
  }

  friend std::ostream& operator<<<Index, TimeStep>(
      std::ostream&, const PurePolicyJointExecutionState<Index, TimeStep>&);

  timestep_type time_;
  std::map<agent_type, local_policy_execution_state_type> local_exec_states_;
};

template <typename Index, typename TimeStep>
std::ostream& operator<<(
    std::ostream& out,
    const PurePolicyJointExecutionState<Index, TimeStep>& x) {
  out << "time = " << x.time_ << ", [";
  for (auto i = x.local_exec_states_.begin(); i != x.local_exec_states_.end();
       i++) {
    if (i != x.local_exec_states_.begin()) {
      out << "; ";
    }
    out << i->first << ": " << i->second;
  }
  out << "]";
  return out;
}

}  // namespace policy
}  // namespace npgi
#endif  // PUREPOLICYJOINTEXECUTIONSTATE_HPP