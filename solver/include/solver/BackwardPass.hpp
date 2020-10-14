#ifndef BACKWARDPASS_HPP
#define BACKWARDPASS_HPP
#include <boost/iterator/filter_iterator.hpp>
#include <limits>
#include <map>
#include <vector>
#include "SPTS/SPTS.hpp"
#include "SPTS/SPTSUtils.hpp"
#include "UCB1.hpp"
#include "core/Value.hpp"
#include "decpomdp/base/DecPOMDP.hpp"
#include "policy/graph/FSC.hpp"
#include "policy/graph/FSCPolicy.hpp"
#include "policy/graph/NodeConfigurations.hpp"
#include "policy/random/RandomPolicy.hpp"
#include "utilities/SamplingUtilities.hpp"
namespace npgi {
namespace backwardpass {

template <typename SPTS>
struct agent_node_is {
  using value_type = typename std::iterator_traits<
      typename SPTS::spts_const_iterator>::value_type;

  agent_node_is(std::size_t agent_idx, fsc_node_t q)
      : agent_idx_(agent_idx), q_(q) {}

  bool operator()(const value_type& x) {
    return (q_ == x.first.nodes_.at(agent_idx_));
  }

 private:
  std::size_t agent_idx_;
  fsc_node_t q_;
};

template <typename SPTS>
struct joint_node_is {
  using value_type = typename std::iterator_traits<
      typename SPTS::spts_const_iterator>::value_type;

  joint_node_is(const std::vector<fsc_node_t>& q) : q_(q) {}

  bool operator()(const value_type& x) { return (q_ == x.first.nodes_); }

 private:
  std::vector<fsc_node_t> q_;
};

template <typename DecPOMDP>
struct BackPassTraits {
  using agent_t = typename DecPOMDP::agent_type;
  using state_t = typename DecPOMDP::state_type;
  using scalar_t = typename DecPOMDP::scalar_type;
  using timestep_t = typename DecPOMDP::timestep_type;
  using joint_belief_t = typename DecPOMDP::joint_belief_type;
  using index_t = typename DecPOMDP::joint_action_space_type::index_type;

  using local_action_type =
      typename DecPOMDP::joint_action_space_type::local_action_type;
  using local_observation_type =
      typename DecPOMDP::joint_observation_space_type::local_observation_type;

  using policy_t = policy::FSCPolicy<index_t, timestep_t>;
  using execution_state_t = typename policy_t::execution_state_type;
  using local_fsc_t = typename policy_t::local_fsc_type;

  using spts_t = typename SPTSTraits<DecPOMDP, policy_t>::spts_t;

  using fwd_pass_t = ForwardPassOutput<DecPOMDP, policy_t>;
};

template <typename DecPOMDP>
struct BackPassInputs {
  using agent_t = typename BackPassTraits<DecPOMDP>::agent_t;
  using local_fsc_t = typename BackPassTraits<DecPOMDP>::local_fsc_t;
  using timestep_t = typename BackPassTraits<DecPOMDP>::timestep_t;
  using joint_belief_t = typename BackPassTraits<DecPOMDP>::joint_belief_t;

  BackPassInputs(joint_belief_t initial_state_distribution,
                 std::map<agent_t, local_fsc_t> fscs, DecPOMDP* problem,
                 double prob_random_policy, std::size_t num_rollouts,
                 std::size_t num_heur_samples, timestep_t start, timestep_t end)
      : initial_state_distribution(initial_state_distribution),
        local_policy_graphs(fscs),
        problem(problem),
        prob_random_policy(prob_random_policy),
        num_rollouts(num_rollouts),
        num_heur_samples(num_heur_samples),
        t_start(start),
        t_end(end) {}

  joint_belief_t initial_state_distribution;
  std::map<agent_t, local_fsc_t> local_policy_graphs;

  DecPOMDP* problem;

  double prob_random_policy;
  std::size_t num_rollouts;
  std::size_t num_heur_samples;
  timestep_t t_start;
  timestep_t t_end;
};

template <typename DecPOMDP>
struct BackPassOutputs {
  using agent_t = typename BackPassTraits<DecPOMDP>::agent_t;
  using local_fsc_t = typename BackPassTraits<DecPOMDP>::local_fsc_t;
  std::map<agent_t, local_fsc_t> local_policy_graphs;
};

template <typename DecPOMDP, typename RandomNumberGenerator>
BackPassOutputs<DecPOMDP> backwardpass(const BackPassInputs<DecPOMDP>& in,
                                       RandomNumberGenerator& g) {
  // initial forward pass
  typename BackPassTraits<DecPOMDP>::fwd_pass_t fwd = npgi::forwardpass(
      *in.problem,
      npgi::policy::FSCPolicy(in.t_start, in.t_end, in.local_policy_graphs,
                              in.problem->A(), in.problem->Z()),
      in.initial_state_distribution, in.num_rollouts, g);
  BackPassOutputs<DecPOMDP> out{in.local_policy_graphs};
  for (auto t = in.t_end - 1; t >= in.t_start; --t) {
    improve_time_step(out.local_policy_graphs, fwd, t, in, g);
  }
  return out;
}

template <typename DecPOMDP, typename RandomNumberGenerator, typename SPTSIter>
typename BackPassTraits<DecPOMDP>::spts_t heuristic_spts_at(
    fsc_node_t q, const typename BackPassTraits<DecPOMDP>::agent_t& agent,
    SPTSIter begin, SPTSIter end, std::size_t n, RandomNumberGenerator& g) {
  typename BackPassTraits<DecPOMDP>::spts_t::spts_container_t m;
  std::sample(begin, end, std::inserter(m, m.end()), n, g);

  // modify the execution states so that agent is at q
  for (const auto & [ execstate, sample ] : m) {
    (void)sample;  // the value can be ignored, we just modify the key
    auto nh = m.extract(execstate);
    nh.key().nodes_.at(agent.index()) = q;
    m.insert(std::move(nh));
  }
  return typename BackPassTraits<DecPOMDP>::spts_t(std::move(m));
}

template <typename DecPOMDP, typename RandomNumberGenerator>
void improve_time_step(std::map<typename BackPassTraits<DecPOMDP>::agent_t,
                                typename BackPassTraits<DecPOMDP>::local_fsc_t>&
                           local_policy_graphs,
                       typename BackPassTraits<DecPOMDP>::fwd_pass_t& fwd,
                       typename BackPassTraits<DecPOMDP>::timestep_t t,
                       const BackPassInputs<DecPOMDP>& in,
                       RandomNumberGenerator& g) {
  bool forward_pass_update_required(false);
  for (auto & [ agent, fsc ] : local_policy_graphs) {
    std::vector<fsc_node_t> already_improved;
    for (const auto& q : get_layer(t, fsc)) {
      if (forward_pass_update_required) {
        fwd = npgi::forwardpass(
            *in.problem,
            npgi::policy::FSCPolicy(in.t_start, t + 1, local_policy_graphs,
                                    in.problem->A(), in.problem->Z()),
            in.initial_state_distribution, in.num_rollouts, g);
        forward_pass_update_required = false;
      }
      agent_node_is<typename BackPassTraits<DecPOMDP>::spts_t> pred(
          agent.index(), q);
      auto spts_begin = fwd.spts.begin(pred);
      auto spts_end = fwd.spts.end(pred);

      // std::cout << "updating agent " << agent << " node " << q << "\n";
      if (std::distance(spts_begin, spts_end) == 0) {
        // std::cout << "no data in SPTS - HEURISTIC IMPROVEMENT!\n";

        // create a SPTS for heuristic update
        npgi::spts_timestep_is<typename BackPassTraits<DecPOMDP>::spts_t> ct(t);
        auto in_begin = fwd.spts.begin(ct), in_end = fwd.spts.end(ct);
        typename BackPassTraits<DecPOMDP>::spts_t spts_heur =
            heuristic_spts_at<DecPOMDP, RandomNumberGenerator>(
                q, agent, in_begin, in_end, in.num_heur_samples, g);
        update_node(q, agent, local_policy_graphs, spts_heur.begin(),
                    spts_heur.end(), t, in, g);
      } else {
        // std::cout << "** updating agent " << agent << " node " << q << " at
        // time step " << t << "\n";
        update_node(q, agent, local_policy_graphs, spts_begin, spts_end, t, in,
                    g);
      }

      // check for same policies in other nodes
      auto identical_to_q =
          get_identical_policy_nodes(q, already_improved, fsc);
      if (!identical_to_q.empty()) {
        // std::cout << "found duplicate policy, redirecting and randomizing "
        // <<
        // q
        // << "\n";
        redirect(q, identical_to_q.front(), fsc);
        auto local_action_space = in.problem->A().local_action_space(agent, t);
        auto local_observation_space =
            in.problem->Z().local_observation_space(agent, t + 1);
        auto configs = get_configurations(q, local_action_space,
                                          local_observation_space, fsc);
        set_unique_random_configuration(configs, q, fsc, g);
        forward_pass_update_required = true;
      }
      already_improved.push_back(q);
    }
  }
}

template <typename SPTSIterator, typename DecPOMDP,
          typename RandomNumberGenerator>
void update_node(fsc_node_t q, typename BackPassTraits<DecPOMDP>::agent_t agent,
                 std::map<typename BackPassTraits<DecPOMDP>::agent_t,
                          typename BackPassTraits<DecPOMDP>::local_fsc_t>&
                     local_policy_graphs,
                 const SPTSIterator begin, const SPTSIterator end,
                 typename BackPassTraits<DecPOMDP>::timestep_t t,
                 const BackPassInputs<DecPOMDP>& in, RandomNumberGenerator& g) {
  auto& fsc = local_policy_graphs[agent];
  auto local_action_space = in.problem->A().local_action_space(agent, t);
  auto local_observation_space =
      in.problem->Z().local_observation_space(agent, t + 1);

  // auto configs =
  //     get_configurations(q, local_action_space, local_observation_space,
  //     fsc);

  // consider only configurations that have non-zero estimated probability of
  // appearing
  using local_action_type =
      typename BackPassTraits<DecPOMDP>::local_action_type;
  using local_observation_type =
      typename BackPassTraits<DecPOMDP>::local_observation_type;
  NodeConfigurationSet<local_action_type, local_observation_type, fsc_node_t>
      cfgs;

  for (const auto& a : in.problem->A().local_action_space(agent, t)) {
    Simulator sim(in.problem, in.t_end);
    typename BackPassTraits<DecPOMDP>::policy_t current_policy(
        t, in.t_end, local_policy_graphs, in.problem->A(), in.problem->Z());

    std::set<local_observation_type> local_observations_nonzero_probability;
    for (SPTSIterator it(begin); it != end; ++it) {
      auto j_act = current_policy.get_joint_action(it->first);
      // assign local action
      std::map<decltype(agent),
               typename DecPOMDP::joint_action_space_type::local_action_type>
          local_actions;
      for (const auto & [ ia, fsc ] : local_policy_graphs) {
        (void)fsc;  // may be ignored.
        local_actions[ia] =
            ((ia == agent) ? a : in.problem->A()
                                     .joint_action_space_at_time(t)
                                     .get_local_element(j_act, ia));
      }
      auto next_action =
          in.problem->A().joint_action_space_at_time(t).get_joint_element(
              local_actions);
      sample_t next_state =
          sim.step(sample_t<DecPOMDP>(t, it->second.state()), next_action, g);
      typename DecPOMDP::joint_observation_space_type::local_observation_type
          local_obs =
              in.problem->Z()
                  .joint_observation_space_at_time(t + 1)
                  .get_local_element(next_state.last_observation(), agent);
      local_observations_nonzero_probability.insert(local_obs);
    }

    EdgeConfigurationSet<local_observation_type, fsc_node_t> edge_cfgs;
    for (const auto& obs : local_observation_space) {
      auto it = local_observations_nonzero_probability.find(obs);
      if (it == local_observations_nonzero_probability.end()) {
        // keep current configuration
        auto ek = out_edge_exists(q, obs, fsc);
        edge_cfgs.set_valid_next_nodes(
            obs, std::vector<fsc_node_t>{boost::target(*ek.first, fsc)});
      } else {
        // this can be changed
        edge_cfgs.set_valid_next_nodes(obs, get_layer(t + 1, fsc));
      }
    }
    cfgs.set_valid_edge_configurations(a, edge_cfgs);
  }
  // std::cout << "there are a total of " << cfgs.size() << " configs (new)\n";

  std::uniform_real_distribution<> u;
  if (u(g) < in.prob_random_policy) {
    // std::cout << "RANDOMIZE!\n";
    set_configuration(*reservoir_sample(cfgs.begin(), cfgs.end(), g), q, fsc);
    return;
  }

  // determine scale of cumulative rewards
  using scalar_t = typename BackPassTraits<DecPOMDP>::scalar_t;
  scalar_t min_cumulative_reward(0.0), max_cumulative_reward(0.0);
  for (typename BackPassTraits<DecPOMDP>::timestep_t t0 = t; t0 < in.t_end;
       ++t0) {
    const scalar_t effective_disc =
        std::pow(in.problem->R().discount(), t0 - t);
    min_cumulative_reward += effective_disc * in.problem->R().min_reward(t0);
    max_cumulative_reward += effective_disc * in.problem->R().max_reward(t0);
  }
  RewardScaler scaler(min_cumulative_reward, max_cumulative_reward);
  // std::cout << "min / max cumulative discounted rewards: "
  //           << min_cumulative_reward << ", " << max_cumulative_reward <<
  //           "\n";

  // std::cout << "SPTS elements: " << std::distance(begin, end)  <<  "\n";
  int num_configs = cfgs.size();
  // std::cout << "agent " << agent << " node " << q << ": " << num_configs << "
  // configurations\n";
  npgi::ucb1::UCB1<typename decltype(cfgs)::value_type,
                   typename BackPassTraits<DecPOMDP>::scalar_t>
      ucb(cfgs.begin(), cfgs.end());

  // optimization
  SPTSIterator it(begin);
  for (int i = 0; i < 10 * num_configs; ++i) {
    const auto& config = ucb.ucb_policy();
    set_configuration(config, q, fsc);
    typename BackPassTraits<DecPOMDP>::policy_t p_eval(
        t, in.t_end, local_policy_graphs, in.problem->A(), in.problem->Z());
    // std::cout << "policy graph node " << it->first << "\n";
    auto reward =
        sample_value(*in.problem, it->second.state(), p_eval, it->first, g);

    // normalize reward to scale [0,1]
    auto scaled_reward = scaler(reward);

    // if ((scaled_reward < 0.0) || (scaled_reward > 1.0)) {
    //   std::cout << "scale exceeded with raw " << reward
    //             << " (min/max): " << min_cumulative_reward << " / "
    //             << max_cumulative_reward << "\n";
    // }

    // std::cout << "UCB rollout " << i << " tried config " << config.index_
    //           << " sampled reward " << reward << " (scaled " << scaled_reward
    //           << ")\n";

    ucb.update(config, scaled_reward);

    ++it;
    if (it == end) it = begin;
  }
  set_configuration(ucb.greedy_policy(), q, fsc);
}

}  // namespace backwardpass
}  // namespace npgi

#endif  // BACKWARDPASS_HPP