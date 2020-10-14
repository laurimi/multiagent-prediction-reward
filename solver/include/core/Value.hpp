#ifndef VALUE_HPP
#define VALUE_HPP
#include <limits>
#include <random>
#include "decpomdp/base/DecPOMDP.hpp"
#include "policy/base/Policy.hpp"

namespace npgi {
template <typename Derived, typename Policy, class RandomNumberGenerator>
typename DecPOMDP<Derived>::scalar_type sample_value(
    const DecPOMDP<Derived>& d, const typename DecPOMDP<Derived>::state_type& s,
    const Policy& p, typename Policy::execution_state_type h,
    RandomNumberGenerator& g) {
  using state_type = typename DecPOMDP<Derived>::state_type;
  using joint_action_type =
      typename DecPOMDP<Derived>::joint_action_space_type::joint_action_type;
  using joint_observation_type = typename DecPOMDP<
      Derived>::joint_observation_space_type::joint_observation_type;
  using scalar_type = typename DecPOMDP<Derived>::scalar_type;

  if (h.time_step() == p.end_time()) return 0;

  if (h.time_step() > p.end_time())
    return std::numeric_limits<scalar_type>::quiet_NaN();

  const joint_action_type a = p.get_joint_action(h);
  const scalar_type immediate_reward = d.R().reward(s, a, h.time_step());
  std::uniform_real_distribution<scalar_type> dis;
  const state_type s_next =
      d.Tr().sample_next_state(s, a, h.time_step(), dis(g));
  const joint_observation_type o =
      d.Ob().sample_observation(s_next, a, h.time_step() + 1, dis(g));
  p.update_execution_state(h, a, o);
  return (immediate_reward +
          d.R().discount() * sample_value(d, s_next, p, h, g));
}

template <typename Derived, typename Policy, class RandomNumberGenerator>
typename DecPOMDP<Derived>::scalar_type sample_value(
    const DecPOMDP<Derived>& d,
    const typename DecPOMDP<Derived>::state_type& initial_state,
    const Policy& policy, RandomNumberGenerator& g) {
  auto e = policy.initial_execution_state();
  return sample_value(d, initial_state, policy, e, g);
}

template <typename Derived, typename Policy, class RandomNumberGenerator>
typename DecPOMDP<Derived>::scalar_type estimate_value(
    const DecPOMDP<Derived>& d,
    const typename DecPOMDP<Derived>::joint_belief_type& initial_belief,
    const Policy& policy, const std::size_t num_rollouts, RandomNumberGenerator& g) {
  using scalar_type = typename DecPOMDP<Derived>::scalar_type;
  scalar_type v = 0.0;
  std::uniform_real_distribution<scalar_type> dis;
  for (std::size_t r = 0; r < num_rollouts; ++r)
  {
    v = (sample_value(d, initial_belief.sample_state(dis(g)), policy, g) +
         static_cast<scalar_type>(r) * v) /
        static_cast<scalar_type>(r + 1);
  }
  return v;
}

template <typename Derived, typename Policy, typename FinalRewardFunctor>
typename DecPOMDP<Derived>::scalar_type value(
    const DecPOMDP<Derived>& d,
    const typename DecPOMDP<Derived>::joint_belief_type& belief,
    const Policy& policy, const typename Policy::execution_state_type& h,
    FinalRewardFunctor& f) {
  using joint_action_type =
      typename DecPOMDP<Derived>::joint_action_space_type::joint_action_type;
  using scalar_type = typename DecPOMDP<Derived>::scalar_type;
  using joint_belief_type = typename DecPOMDP<Derived>::joint_belief_type;

  if (h.time_step() == policy.end_time()) return f(belief);

  if (h.time_step() > policy.end_time())
    return std::numeric_limits<scalar_type>::quiet_NaN();

  const joint_action_type a = policy.get_joint_action(h);
  const scalar_type immediate_reward =
      d.R().reward(belief.pmf(), a, h.time_step());

  scalar_type future_value = 0.0;
  for (auto obs_it = d.Z().begin(h.time_step()+1),
            obs_it_end = d.Z().end(h.time_step()+1);
       obs_it != obs_it_end; ++obs_it) {
    joint_belief_type b_next =
        joint_belief_type(d.Tr().predict(belief.pmf(), a, h.time_step()));
    const scalar_type p_observation =
        d.Ob().update(b_next.pmf(), a, *obs_it, h.time_step() + 1);
    if (p_observation > 0.0) {
      future_value += d.R().discount() * p_observation *
                      value(d, b_next, policy,
                            policy.next_execution_state(h, a, *obs_it), f);
    }
  }

  return (immediate_reward + future_value);
}

template <typename Derived, typename Policy, typename FinalRewardFunctor>
typename DecPOMDP<Derived>::scalar_type value(
    const DecPOMDP<Derived>& d,
    const typename DecPOMDP<Derived>::joint_belief_type& initial_belief,
    const Policy& policy, FinalRewardFunctor& f) {
  auto e = policy.initial_execution_state();
  return value(d, initial_belief, policy, e, f);
}

}  // namespace npgi

#endif  // VALUE_HPP