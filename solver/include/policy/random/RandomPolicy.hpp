#ifndef RANDOMPOLICY_HPP
#define RANDOMPOLICY_HPP
#include <random>
#include <vector>
#include "policy/base/TimedPolicy.hpp"
#include "utilities/SamplingUtilities.hpp"
namespace npgi {
namespace policy {

template <typename ActionSet, typename JointObservation, typename TimeStep,
          typename RandomNumberGenerator>
struct RandomPolicy
    : public TimedPolicy<typename ActionSet::value_type, JointObservation, TimeStep> {
  using base_type = TimedPolicy<typename ActionSet::value_type, JointObservation, TimeStep>;
  using timestep_type = typename base_type::timestep_type;
  using joint_action_type = typename base_type::joint_action_type;
  using execution_state_type = typename base_type::execution_state_type;
  using rng_type = RandomNumberGenerator;
  
  RandomPolicy(timestep_type start, timestep_type end,
               const ActionSet& actionset, rng_type& g)
      : base_type(start, end), actionset_(actionset), g_(g) {}

  execution_state_type initial_execution_state() const {
    return execution_state_type(this->first_time_step());
  }

  joint_action_type get_joint_action(const execution_state_type& e) const {
    return *reservoir_sample(actionset_.begin(), actionset_.end(), g_);
  }

 private:
  ActionSet actionset_;
  rng_type& g_;
};
} // namespace policy
}  // namespace npgi

#endif  // BLINDPOLICY_HPP