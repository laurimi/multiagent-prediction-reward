#ifndef TIMEDPOLICY_HPP
#define TIMEDPOLICY_HPP
#include "ExecutionState.hpp"
#include "Policy.hpp"
namespace npgi {
namespace policy {

template <typename JointAction, typename JointObservation, typename TimeStep>
class TimedPolicy;

template <typename JointAction, typename JointObservation, typename TimeStep>
struct policy_traits<TimedPolicy<JointAction, JointObservation, TimeStep>> {
  using joint_action_type = JointAction;
  using joint_observation_type = JointObservation;
  using timestep_type = TimeStep;
  using execution_state_type = TimedExecutionState<TimeStep>;
};

template <typename JointAction, typename JointObservation, typename TimeStep>
struct TimedPolicy
    : public Policy<TimedPolicy<JointAction, JointObservation, TimeStep>> {
  using derived_type = TimedPolicy<JointAction, JointObservation, TimeStep>;
  using base_type = Policy<derived_type>;

  using timestep_type = typename policy_traits<derived_type>::timestep_type;
  using joint_action_type =
      typename policy_traits<derived_type>::joint_action_type;
  using joint_observation_type =
      typename policy_traits<derived_type>::joint_observation_type;
  using execution_state_type =
      typename policy_traits<derived_type>::execution_state_type;

  TimedPolicy(timestep_type start, timestep_type end) : base_type(start, end) {}

  execution_state_type initial_execution_state() const {
    return execution_state_type(this->start_time());
  }

  execution_state_type next_execution_state(
      const execution_state_type& e, const joint_action_type& a,
      const joint_observation_type& o) const {
    (void)a;
    (void)o; // timed policy ignores action and observation, just increments time.
    return execution_state_type(e.time_ + 1);
  }

  void update_execution_state(execution_state_type& e,
                              const joint_action_type& a,
                              const joint_observation_type& o) const {
    (void)a;
    (void)o; // timed policy ignores action and observation, just increments time.
    ++e.time_;
  }

  joint_action_type get_joint_action(const execution_state_type& e) const {
    return this->underlying().get_joint_action(e);
  }
};
}  // namespace policy
}  // namespace npgi

#endif  // TIMEDPOLICY_HPP