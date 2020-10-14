#ifndef MADPWRAPPER_H
#define MADPWRAPPER_H
#include <cstddef>
#include <memory>
#include <string>

namespace npgi {
namespace madpwrapper {
class MADPDecPOMDPDiscrete {
 public:
  MADPDecPOMDPDiscrete(const std::string& dpomdp_filename, bool verbose = false);
  ~MADPDecPOMDPDiscrete();
  std::size_t num_agents() const;
  std::size_t num_states() const;
  std::size_t num_joint_actions() const;
  std::size_t num_joint_observations() const;
  std::size_t num_actions(std::size_t agent) const;
  std::size_t num_observations(std::size_t agent) const;
  std::string agent_name(std::size_t agent) const;
  std::string state_name(std::size_t state_index) const;
  std::string action_name(std::size_t agent, std::size_t action_index) const;
  std::string observation_name(std::size_t agent,
                               std::size_t observation_index) const;

  double initial_belief_at(std::size_t state) const;
  double reward(std::size_t state, std::size_t j_act) const;
  double discount() const;
  double observation_probability(std::size_t j_obs, std::size_t state,
                                 std::size_t j_act) const;
  double transition_probability(std::size_t new_state, std::size_t state,
                                std::size_t j_act) const;

 private:
  class MADPDecPOMDPDiscrete_impl;
  std::unique_ptr<MADPDecPOMDPDiscrete_impl> pimpl_;
};
}  // namespace madpwrapper
}  // namespace npgi

#endif  // MADPWRAPPER_H