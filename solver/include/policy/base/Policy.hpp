#ifndef POLICY_HPP
#define POLICY_HPP
#include <vector>
#include "core/CRTPHelper.hpp"
namespace npgi {
namespace policy {

template <typename Derived>
struct policy_traits;

template <typename Derived>
struct Policy : crtp_helper<Derived> {
  using joint_action_type = typename policy_traits<Derived>::joint_action_type;
  using joint_observation_type = typename policy_traits<Derived>::joint_observation_type;
  using timestep_type = typename policy_traits<Derived>::timestep_type;
  using execution_state_type = typename policy_traits<Derived>::execution_state_type;

  Policy(timestep_type start, timestep_type end)
      : start_(start), end_(end) {}

  execution_state_type initial_execution_state() const {
    return this->underlying().initial_execution_state();
  }

  execution_state_type next_execution_state(const execution_state_type& e, const joint_action_type& a, const joint_observation_type& o) const
  {
    return this->underlying().next_execution_state(e, a, o);
  }

  void update_execution_state(execution_state_type& e, const joint_action_type& a, const joint_observation_type& o) const
  {
    this->underlying().update_execution_state(e, a, o);
  }

  joint_action_type get_joint_action(const execution_state_type& e) const {
    return this->underlying().get_joint_action(e);
  }

  timestep_type start_time() const { return start_; }
  timestep_type end_time() const { return end_; }

 private:
  timestep_type start_, end_;
};
} // namespace policy
}  // namespace npgi

#endif  // POLICY_HPP