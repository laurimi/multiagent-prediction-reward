#ifndef POLICYSTATE_HPP
#define POLICYSTATE_HPP
#include <ostream>
#include "core/CRTPHelper.hpp"
namespace npgi {
namespace policy {

template <typename Derived>
struct execution_state_traits;

template <typename Derived>
struct ExecutionState : crtp_helper<Derived> {
  using timestep_type = typename execution_state_traits<Derived>::timestep_type;

  timestep_type time_step() const { return this->underlying().time_; }
};

template <typename TimeStep>
struct TimedExecutionState;

template <typename TimeStep>
struct execution_state_traits<TimedExecutionState<TimeStep>> {
  using timestep_type = TimeStep;
};

template <typename TimeStep>
struct TimedExecutionState : public ExecutionState<TimedExecutionState<TimeStep>> {
  using base_type = ExecutionState<TimedExecutionState<TimeStep>>;
  using timestep_type = typename base_type::timestep_type;

  TimedExecutionState(timestep_type t) : time_(t) {}
  bool operator<(const TimedExecutionState& other) const {
    return time_ < other.time_;
  }

  timestep_type time_;
};

}  // namespace policy
}  // namespace npgi

#endif  // POLICYSTATE_HPP