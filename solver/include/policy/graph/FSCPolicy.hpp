#ifndef FSCPOLICY_HPP
#define FSCPOLICY_HPP
#include <map>
#include <ostream>
#include "FSC.hpp"
#include "FSCExecutionState.hpp"
#include "policy/base/Policy.hpp"
#include "decpomdp/discrete/DiscreteElements.hpp"
#include "decpomdp/discrete/JointActionSpaceFlat.hpp"
#include "decpomdp/discrete/JointObservationSpaceFlat.hpp"

namespace npgi {
namespace policy {

template <typename Index, typename TimeStep>
class FSCPolicy;

template <typename Index, typename TimeStep>
struct policy_traits<FSCPolicy<Index, TimeStep>>
{
  using timestep_type = TimeStep;
  using joint_action_type = DiscreteJointAction<Index>;
  using joint_observation_type = DiscreteJointObservation<Index>;
  using execution_state_type = policy::FSCExecutionState<timestep_type, fsc_node_t>;
};

template <typename Index = std::size_t, typename TimeStep = int>
class FSCPolicy : public Policy<FSCPolicy<Index, TimeStep>>
{
public:
  using base_type = Policy<FSCPolicy<Index, TimeStep>>;
  using derived_type = FSCPolicy<Index, TimeStep>;
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
  using local_fsc_type = fsc_graph_t<local_action_type, local_observation_type>;

  FSCPolicy(timestep_type tstart, timestep_type tlast,
            const std::map<agent_type, local_fsc_type>& local,
            const joint_action_space_type& A,
            const joint_observation_space_type& Z)
      : base_type(tstart, tlast), local_(local), A_(A), Z_(Z) {}

  execution_state_type initial_execution_state() const {
    return execution_state_type(this->start_time(),
                                std::vector<fsc_node_t>(local_.size(), 0));
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
    (void)a; // a is not needed for the state update
    update_nodes(e, o);
    ++e.time_;
  }

  joint_action_type get_joint_action(const execution_state_type& e) const {
    std::map<agent_type, local_action_type> local_actions;
    for (const auto &[agent, fsc] : local_) {
      local_actions[agent] = fsc[e.nodes_.at(agent.index())].action_;
    }
    return A_.get_joint_action(local_actions, e.time_step());
  }

 private:
  void update_nodes(execution_state_type& e,
                    const joint_observation_type& o) const {
    for (const auto &[agent, fsc] : local_) {
      auto ek = out_edge_exists(
          e.nodes_.at(agent.index()),
          Z_.get_local_observation(o, e.time_ + 1, agent.index()), fsc);
      if (!ek.second) {
        std::ostringstream os;
        os << "edge " << o << " not found!";
        throw std::runtime_error(os.str());
      }
      e.nodes_.at(agent.index()) = boost::target(*ek.first, fsc);
    }
  }

  const std::map<agent_type, local_fsc_type>& local_;
  const joint_action_space_type& A_;
  const joint_observation_space_type& Z_;
};
} // namespace policy
}  // namespace npgi

#endif  // FSCPOLICY_HPP