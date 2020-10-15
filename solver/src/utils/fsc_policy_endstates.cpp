#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include "SPTS/SPTS.hpp"
#include "SPTS/SPTSUtils.hpp"
#include "core/Value.hpp"
#include "decpomdp/discrete/DecPOMDPFlat.hpp"
#include "decpomdp/discrete/JointActionSpaceFlat.hpp"
#include "decpomdp/discrete/JointBeliefFlat.hpp"
#include "decpomdp/discrete/JointObservationSpaceFlat.hpp"
#include "madp_wrapper/MADPWrapper.h"
#include "madp_wrapper/MADPWrapperUtilities.h"
#include "observationmodel/ObservationModelFlat.hpp"
#include "policy/pure/PurePolicyJoint.hpp"
#include "policy/pure/PurePolicyTools.hpp"
#include "rewardmodel/RewardModelFlat.hpp"
#include "solver/BackwardPass.hpp"
#include "statetransitionmodel/StateTransitionModelFlat.hpp"

int main(int argc, char** argv) {
  int tstart;
  int tlast;
  int num_rollouts;
  unsigned int seed;

  namespace po = boost::program_options;
  po::options_description config(
      "Sample end states under a given joint policy in a Dec-rhoPOMDP"
      "\nUsage: " +
      std::string(argv[0]) + " [OPTION]... [DPOMDP-FILE]\nOptions");
  config.add_options()("help", "produce help message")(
      "start,s", po::value<int>(&tstart)->default_value(0),
      "starting time step")("last,l", po::value<int>(&tlast)->default_value(1),
                            "last time step")(
      "files", po::value<std::vector<std::string>>()->multitoken(),
      "[AGENT0_POLICY] ... [AGENTN_POLICY] policy files to read")(
      "num-states,n", po::value<int>(&num_rollouts)->default_value(1),
      "number of end states to sample")(
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

  npgi::madpwrapper::MADPDecPOMDPDiscrete madp(
      vm["dpomdp-file"].as<std::string>());
  std::cout << "Loaded problem file " << vm["dpomdp-file"].as<std::string>()
            << std::endl;
  npgi::DecPOMDPFlat d = npgi::madpwrapper::to_flat_decpomdp(madp);
  npgi::JointBeliefFlat initial_state_distribution(
      npgi::madpwrapper::initial_state_distribution(madp));
  const std::size_t num_files =
      vm["files"].as<std::vector<std::string>>().size();
  if (num_files != madp.num_agents()) {
    std::cout << "error: number of policy files provided (" << num_files
              << ") does not match number of agents (" << madp.num_agents()
              << ")\n";
    return 1;
  }

  using pg_type = npgi::fsc_graph_t<npgi::DiscreteLocalAction<>,
                                    npgi::DiscreteLocalObservation<>>;
  std::map<npgi::Agent<>, pg_type> fscs;
  for (std::size_t i = 0; i < num_files; ++i) {
    const std::string filename =
        vm["files"].as<std::vector<std::string>>().at(i);
    std::ifstream is(filename);
    if (!is.is_open()) {
      std::cout << "error: failed to open file: " << filename << '\n';
      return 1;
    } else {
      is >> fscs[i];
    }
    std::cout << "agent " << i << ", read policy\n" << fscs[i] << std::endl;
  }
  auto policy = npgi::policy::FSCPolicy(tstart, tlast, fscs, d.A(), d.Z());

  std::mt19937 g(seed);
  auto fwd =
      npgi::forwardpass(d, policy, initial_state_distribution, num_rollouts, g);

  using sample_type = npgi::sample_t<npgi::DecPOMDPFlat<>>;
  using execution_state_t =
      npgi::policy::FSCPolicy<>::execution_state_type;
  using spts_t = npgi::SPTS<execution_state_t, sample_type>;
  npgi::spts_timestep_is<spts_t> ct(tlast - 1);
  auto in_begin = fwd.spts.begin(ct), in_end = fwd.spts.end(ct);
  std::cout << "policy execution state -- state\n";
  for (; in_begin != in_end; ++in_begin)
    std::cout << in_begin->first << " -- " << in_begin->second.state() << "\n";

  return 0;
}