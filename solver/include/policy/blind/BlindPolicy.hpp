#ifndef BLINDPOLICY_HPP
#define BLINDPOLICY_HPP
#include "policy/base/TimedPolicy.hpp"
namespace npgi {
namespace policy {

template <typename JointAction, typename JointObservation, typename TimeStep>
struct BlindPolicy
    : public TimedPolicy<JointAction, JointObservation, TimeStep> {
  using base_type = TimedPolicy<JointAction, JointObservation, TimeStep>;
  using timestep_type = typename base_type::timestep_type;
  using joint_action_type = typename base_type::joint_action_type;
  using execution_state_type = typename base_type::execution_state_type;

  BlindPolicy(timestep_type start, timestep_type end,
              const joint_action_type& blind_joint_action)
      : base_type(start, end), blind_joint_action_(blind_joint_action) {}

  joint_action_type get_joint_action(const execution_state_type& e) const {
    (void)e; // blind policy ignores execution state and always returns the fixed action
    return blind_joint_action_;
  }

 private:
  joint_action_type blind_joint_action_;
};

} // namespace policy
}  // namespace npgi

#endif  // BLINDPOLICY_HPP