#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "core/Value.hpp"
#include "decpomdp/conversions/DecPOMDPConversions.hpp"
#include "decpomdp/discrete/DecPOMDPFlat.hpp"
#include "madp_wrapper/MADPWrapper.h"
#include "madp_wrapper/MADPWrapperUtilities.h"
#include "policy/graph/FSC.hpp"
#include "policy/graph/FSCPolicy.hpp"
#include "policy/graph/FSCUtils.hpp"
#include "rewardmodel/LinearizedNegEntropy.hpp"
#include "solver/BackwardPass.hpp"
#include "utilities/Combinations.hpp"
#include "utilities/SamplingUtilities.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  if (!v.empty()) {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

int main(int argc, char** argv) {
  int tstart;
  int tlast;
  std::size_t num_prediction_actions;
  std::size_t num_rollouts;
  double prob_random_policy;
  std::size_t num_heur_samples;
  // std::size_t num_value_rollouts;
  std::size_t num_improvements;
  unsigned int seed;
  std::vector<std::size_t> non_start_layer_widths;

  namespace po = boost::program_options;
  po::options_description config(
      "Evaluate value of blind policy in a finite horizon discrete "
      "(Dec)-rhoPOMDP "
      "\nUsage: " +
      std::string(argv[0]) + " [OPTION]... [DPOMDP-FILE]\nOptions");
  config.add_options()("help", "produce help message")(
      "start,s", po::value<int>(&tstart)->default_value(0),
      "starting time step")("last,l", po::value<int>(&tlast)->default_value(1),
                            "last time step")(
      "width", po::value<std::vector<std::size_t>>(&non_start_layer_widths)
                   ->multitoken(),
      "[WIDTH_1] ... [WIDTH_M] policy graph layer widths")(
      "num-prediction-actions,e",
      po::value<std::size_t>(&num_prediction_actions)->default_value(0),
      "toggle to use expected prediction rewards")
      (
            "use-entropy-reward,m",
            po::bool_switch()->default_value(false),
            "toggle to use entropy rewards as final reward (if num_prediction_actions == 0)")(
      "update-prediction-rewards,u", po::bool_switch()->default_value(false),
      "toggle to update prediction rewards")(
      "rollouts,r", po::value<std::size_t>(&num_rollouts)->default_value(1),
      "number of rollouts in forward pass")(
      "randomprob,p",
      po::value<double>(&prob_random_policy)->default_value(0.1),
      "probability to randomize policy")(
      "nheursamples,n",
      po::value<std::size_t>(&num_heur_samples)->default_value(1),
      "number of SPTS samples to use in heuristic update")/*(
      "valuerollouts,v",
      po::value<std::size_t>(&num_value_rollouts)->default_value(1000),
      "rollouts for policy value estimation between iterations")*/(
      "improvements,i",
      po::value<std::size_t>(&num_improvements)->default_value(1),
      "number of improvements")(
      "seed,g", po::value<unsigned int>(&seed)->default_value(1234567890),
      "RNG seed");

  po::positional_options_description pp;
  pp.add("dpomdp-file", -1);

  po::options_description hidden("Hidden options");
  hidden.add_options()("dpomdp-file", po::value<std::string>(), "input file");

  po::options_description cmdline_options;
  cmdline_options.add(config).add(hidden);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(cmdline_options)
                .positional(pp)
                .run(),
            vm);
  po::notify(vm);
  if (vm.count("help") || argc == 1) {
    std::cout << config << std::endl;
    return 1;
  }
  if (!vm.count("dpomdp-file")) {
    std::cout << "error: You must specify an input dpomdp file" << std::endl;
    return 1;
  }

  if (tlast < tstart) {
    std::cout << "error: last time step must be less than or equal to starting "
                 "time step\n";
    return 1;
  }

  if (num_heur_samples > num_rollouts) {
    std::cout << "error: number of rollouts must be equal to or greater than "
                 "number of samples for heuristic update\n";
    return 1;
  }

  std::mt19937 g(seed);

  npgi::madpwrapper::MADPDecPOMDPDiscrete madp(
      vm["dpomdp-file"].as<std::string>());
  std::cout << "Loaded problem file " << vm["dpomdp-file"].as<std::string>()
            << std::endl;
  npgi::DecPOMDPFlat d = npgi::madpwrapper::to_flat_decpomdp(madp);
  npgi::JointBeliefFlat initial_state_distribution(
      npgi::madpwrapper::initial_state_distribution(madp));

  int t_prediction = tlast;
  if (num_prediction_actions > 0) {
    std::cout << "Adding " << num_prediction_actions << " prediction rewards\n";
    npgi::conversion_settings s;
    s.time_of_prediction_actions = t_prediction;
    for (std::size_t i = 0; i < num_prediction_actions; ++i) {
      s.linearization_points.emplace_back(
          npgi::sample_unit_simplex<double, decltype(g)>(madp.num_states(), g));
    }

    // write out prediction actions
    std::ostringstream os;
    os << "linearization_beliefs_step_" << std::setfill('0') << std::setw(4) << 0
       << ".csv";
    std::ofstream out(os.str());
    for (const auto& v : s.linearization_points) {
      std::ostringstream ss;
      if(!v.empty()) {
         std::copy(v.begin(), std::prev(v.end()), std::ostream_iterator<double>(ss, ", "));
         ss << v.back();
      }
      out << ss.str() << "\n";
    }

    d = add_prediction_actions(d, s);
    ++tlast;
  }

  std::vector<std::size_t> layer_widths{1};
  layer_widths.insert(layer_widths.end(), non_start_layer_widths.begin(),
                      non_start_layer_widths.end());

  using pg_type = npgi::fsc_graph_t<npgi::DiscreteLocalAction<>,
                                    npgi::DiscreteLocalObservation<>>;
  std::map<npgi::Agent<>, pg_type> fscs;
  for (std::size_t i = 0; i < madp.num_agents(); ++i) {
    npgi::Agent<> agent(i);
    fscs[agent] = get_fsc(layer_widths, agent, d.A(), d.Z());
    npgi::randomize(fscs[agent], agent, d.A(), d.Z(), g);


    std::ostringstream os;
    os << "agent_" << std::setfill('0') << std::setw(3)
       << std::to_string(agent.index()) + "_step_" << std::setfill('0')
       << std::setw(4) << 0 << ".dot";
    std::ofstream out(os.str());
    out << fscs.at(agent);
  }

  std::cout << "tstart = " << tstart << ", tlast = " << tlast << std::endl;
  using decpomdp_type =
      npgi::DecPOMDPFlat<std::size_t, std::size_t, std::size_t, double, int>;
  auto final_reward =
      (((num_prediction_actions == 0) && vm["use-entropy-reward"].as<bool>())
           ? [](const typename decpomdp_type::joint_belief_type&
                    b) { return -b.entropy(); }
           : [](const typename decpomdp_type::joint_belief_type&) {
               return 0.0;
             });

  // auto estimated_best_value = estimate_value(d, initial_state_distribution,
  // npgi::policy::FSCPolicy(
  //                     tstart, tlast, fscs, d.A(), d.Z()), num_value_rollouts,
  //                     g);

  auto best_policy_value = npgi::value(
      d, initial_state_distribution,
      npgi::policy::FSCPolicy(tstart, t_prediction, fscs, d.A(), d.Z()),
      final_reward);

  auto best_fscs = fscs;

  auto start = std::chrono::steady_clock::now();
  std::cout << std::setw(4) << "iter" << ", " << std::setw(8) << "seconds"
            << ", " << std::setw(8) << "value" << ", " << std::setw(8) << "best value" << std::endl;

  // print null iteration result
  std::cout << std::setw(4) << 0 << ", " << std::setw(8)
            << 0 << ", " << std::setw(8) << best_policy_value << ", "
            << std::setw(8) << best_policy_value << std::endl;

  for (std::size_t j = 0; j < num_improvements; ++j) {
    // auto p = npgi::policy::FSCPolicy(tstart, tlast, best_fscs, d.A(), d.Z());
    npgi::backwardpass::BackPassInputs in(initial_state_distribution, best_fscs,
                                          &d, prob_random_policy, num_rollouts,
                                          num_heur_samples, tstart, tlast);

    auto out = npgi::backwardpass::backwardpass(in, g);

    // auto improved_policy = npgi::policy::FSCPolicy(tstart, tlast,
    // out.local_policy_graphs, d.A(), d.Z());
    // auto value = estimate_value(
    //     d, initial_state_distribution, improved_policy, num_value_rollouts,
    //     g);

    auto improved_policy = npgi::policy::FSCPolicy(
        tstart, t_prediction, out.local_policy_graphs, d.A(), d.Z());
    auto value = npgi::value(d, initial_state_distribution, improved_policy,
                             final_reward);
    if (value > best_policy_value) {
      best_fscs = out.local_policy_graphs;
      best_policy_value = value;
    }


    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << std::setw(4) << (j + 1) << ", " << std::setw(8)
              << elapsed_seconds.count() << ", " << std::setw(8) << value
              << ", " << std::setw(8) << best_policy_value << std::endl;

    // store improved policy to file
    for (std::size_t i = 0; i < madp.num_agents(); ++i) {
      npgi::Agent<> agent(i);
      std::ostringstream os;
      os << "agent_" << std::setfill('0') << std::setw(3)
         << std::to_string(agent.index()) + "_step_" << std::setfill('0')
         << std::setw(4) << (j + 1) << ".dot";
      std::ofstream ofs(os.str());
      ofs << out.local_policy_graphs.at(agent);
    }

    if (vm["update-prediction-rewards"].as<bool>() &&
        (num_prediction_actions > 0)) {
      // std::cout << "updating prediction rewards\n";
      auto p = npgi::policy::FSCPolicy(tstart, tlast, best_fscs, d.A(), d.Z());
      auto fwd =
          npgi::forwardpass(d, p, initial_state_distribution, num_rollouts, g);
      auto spts = fwd.spts;

      // get a map from reachability probability to the expected belief state at
      // the node
      std::map<double,
               std::pair<std::vector<npgi::fsc_node_t>, std::vector<double>>>
          m;

      std::vector<std::vector<npgi::fsc_node_t>> final_nodes;
      for (const auto & [ agent, fsc ] : best_fscs) {
        final_nodes.emplace_back(npgi::get_layer(tlast - 1, fsc));
      }

      for (const auto& xc : npgi::make_combinations(final_nodes)) {
        auto x = npgi::get_combination(xc);
        npgi::backwardpass::joint_node_is<decltype(spts)> pred(x);
        auto first = spts.begin(pred), second = spts.end(pred);
        const int n = std::distance(first, second);
        const double prob_x =
            static_cast<double>(n) /
            static_cast<double>(
                num_rollouts);  // reachability probability of this node
        std::pair<std::vector<npgi::fsc_node_t>, std::vector<double>> q_b_pair{
            x, std::vector<double>()};
        if (n == 0) {
          q_b_pair.second = npgi::sample_unit_simplex<double, decltype(g)>(
              madp.num_states(), g);
        } else {
          const double sample_contribution = 1.0 / static_cast<double>(n);
          q_b_pair.second.resize(madp.num_states());
          for (; first != second; ++first) {
            q_b_pair.second[first->second.state()] += sample_contribution;
          }
        }

        m.emplace(prob_x, q_b_pair);
      }

      // std::cout << "Got probabilities and nodes:\n";
      // for (const auto& [prob, qb] : m)
      // {
      //   std::cout << "node " << qb.first << ", probability = " << prob
      //             << ", belief: " << qb.second << "\n";
      // }

      npgi::conversion_settings s;
      s.time_of_prediction_actions = t_prediction;
      for (auto iter = m.rbegin();
           (iter != m.rend() &&
            s.linearization_points.size() < num_prediction_actions);
           ++iter) {
        // std::cout << "adding for node " << iter->second.first
        //           << ", probability = " << iter->first
        //           << ", belief: " << iter->second.second << "\n";
        s.linearization_points.push_back(iter->second.second);
      }
      // if needed, add more random entries.
      for (std::size_t i = s.linearization_points.size();
           i < num_prediction_actions; ++i) {
        std::vector<double> blin(npgi::sample_unit_simplex<double, decltype(g)>(
            madp.num_states(), g));
        // std::cout << "adding random belief " << blin << "\n";
        s.linearization_points.emplace_back(blin);
      }

      // write out prediction actions
      std::ostringstream os;
      os << "linearization_beliefs_step_" << std::setfill('0') << std::setw(4)
         << (j + 1) << ".csv";
      std::ofstream out(os.str());
      for (const auto& v : s.linearization_points) {
        std::ostringstream ss;
        if(!v.empty()) {
           std::copy(v.begin(), std::prev(v.end()), std::ostream_iterator<double>(ss, ", "));
           ss << v.back();
        }
        out << ss.str() << std::endl;
      }

      // Sanity check: does the value improve??
      // NO, because we don't check for the ordering of the updated prediction
      // actions. Should be fine though...
      // auto p_best = npgi::policy::FSCPolicy(tstart, tlast, best_fscs, d.A(),
      // d.Z());
      // auto value = npgi::value(d, initial_state_distribution, p_best,
      // final_reward);
      d = npgi::update_prediction_actions(d, s);
      // auto updated_value = npgi::value(d, initial_state_distribution, p_best,
      // final_reward);
      // std::cout << "value before update = " << value << ", value after update
      // = " << updated_value << "\n";
    }
  }

  // store improved policy to file
  for (std::size_t i = 0; i < best_fscs.size(); ++i) {
    npgi::Agent<> agent(i);
    std::ostringstream os;
    os << "agent_" << std::setfill('0') << std::setw(3)
       << std::to_string(agent.index()) + "_best_policy.dot";
    std::ofstream ofs(os.str());
    ofs << best_fscs.at(agent);
  }

  return 0;
}