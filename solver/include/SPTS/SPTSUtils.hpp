#ifndef SPTSUTILS_HPP
#define SPTSUTILS_HPP
#include <iterator>
#include "SPTS.hpp"
#include "decpomdp/base/DecPOMDP.hpp"
#include "decpomdp/base/Simulator.hpp"
#include "policy/base/Policy.hpp"
namespace npgi {

template <typename DecPOMDP, typename Policy>
struct SPTSTraits {
  using state_t = typename DecPOMDP::state_type;
  using scalar_t = typename DecPOMDP::scalar_type;
  using execution_state_t = typename Policy::execution_state_type;
  using spts_t = SPTS<execution_state_t, sample_t<DecPOMDP>>;
};

template <typename DecPOMDP, typename Policy, typename RandomNumberGenerator>
typename SPTSTraits<DecPOMDP, Policy>::scalar_t add_rollout(
    typename SPTSTraits<DecPOMDP, Policy>::spts_t& spts,
    const Policy& p, const typename Policy::execution_state_type& e,
    const Simulator<DecPOMDP>& sim, const sample_t<DecPOMDP>& sample,
    RandomNumberGenerator& g) {
  spts.insert(e, sample);
  if (e.time_step() == p.end_time()) {
    return 0.0;
  }

  auto joint_action = p.get_joint_action(e);
  auto next_sample = sim.step(sample, joint_action, g);
  double future_disc_reward = add_rollout(
      spts, p,
      p.next_execution_state(e, joint_action, next_sample.last_observation()),
      sim, next_sample, g);

  return next_sample.last_reward() + future_disc_reward;
}

template <typename DecPOMDP, typename Policy>
struct ForwardPassOutput {
  typename SPTSTraits<DecPOMDP, Policy>::spts_t spts;
};

template <typename DecPOMDP, typename Policy, typename RandomNumberGenerator>
ForwardPassOutput<DecPOMDP, Policy> forwardpass(
    const DecPOMDP& d, const Policy& p,
    const typename DecPOMDP::joint_belief_type& initial_state_distribution,
    std::size_t num_rollouts, RandomNumberGenerator& g) {
  ForwardPassOutput<DecPOMDP, Policy> out;
  Simulator sim(&d, p.end_time());
  for (std::size_t i = 0; i < num_rollouts; ++i) {
    auto sim_sample = Simulator<DecPOMDP>::sample_initial_simulation_state(
        initial_state_distribution, p.start_time(), g);
    add_rollout(out.spts, p, p.initial_execution_state(), sim,
                sim_sample, g);
  }
  return out;
}

// Useful predicate functions as structs
template <typename SPTS>
struct spts_timestep_is {
  using timestep_type = typename SPTS::execution_state_t::timestep_type;
  using value_type = typename std::iterator_traits<
      typename SPTS::spts_const_iterator>::value_type;

  spts_timestep_is(timestep_type time) : time_(time) {}

  bool operator()(const value_type& x) {
    return (time_ == x.first.time_step());
  }

 private:
  timestep_type time_;
};

}  // namespace npgi

#endif  // SPTSUTILS_HPP