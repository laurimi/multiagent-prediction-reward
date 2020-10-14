#include "madp_wrapper/MADPWrapper.h"
#include "madp/DecPOMDPDiscrete.h"
#include "madp/MADPParser.h"
namespace npgi {
namespace madpwrapper {

struct MADPDecPOMDPDiscrete::MADPDecPOMDPDiscrete_impl {
  MADPDecPOMDPDiscrete_impl(const std::string& dpomdp_filename)
      : d_(std::make_unique<::DecPOMDPDiscrete>("", "", dpomdp_filename)) {
    MADPParser parser(d_.get());
  }
  std::unique_ptr<::DecPOMDPDiscrete> d_;
};

MADPDecPOMDPDiscrete::MADPDecPOMDPDiscrete(const std::string& dpomdp_filename, bool verbose)
    : pimpl_(std::make_unique<MADPDecPOMDPDiscrete::MADPDecPOMDPDiscrete_impl>(
          dpomdp_filename)) {
  if (!verbose) return;

  std::cout << "read " << dpomdp_filename << " with " << num_agents()
            << " agents, " << num_joint_actions() << " joint actions and "
            << num_joint_observations() << " joint observations\n";

  for (std::size_t a = 0; a < num_joint_actions(); ++a) {
    for (std::size_t sn = 0; sn < num_joint_observations(); ++sn) {
      for (std::size_t so = 0; so < num_states(); ++so)
        std::cout << "T[sn=" << sn << ", so=" << so << ", a=" << a
                  << "] = " << transition_probability(sn, so, a) << "\n";
    }
  }

  for (std::size_t a = 0; a < num_joint_actions(); ++a) {
    for (std::size_t o = 0; o < num_joint_observations(); ++o) {
      for (std::size_t s = 0; s < num_states(); ++s)
        std::cout << "O[o=" << o << ", s=" << s << ", a=" << a
                  << "] = " << observation_probability(o, s, a) << "\n";
    }
  }
}

MADPDecPOMDPDiscrete::~MADPDecPOMDPDiscrete() = default;

std::size_t MADPDecPOMDPDiscrete::num_agents() const {
  return pimpl_->d_->GetNrAgents();
}
std::size_t MADPDecPOMDPDiscrete::num_states() const {
  return pimpl_->d_->GetNrStates();
}
std::size_t MADPDecPOMDPDiscrete::num_joint_actions() const {
  return pimpl_->d_->GetNrJointActions();
}
std::size_t MADPDecPOMDPDiscrete::num_joint_observations() const {
  return pimpl_->d_->GetNrJointObservations();
}
std::size_t MADPDecPOMDPDiscrete::num_actions(std::size_t agent) const {
  return pimpl_->d_->GetNrActions(agent);
}
std::size_t MADPDecPOMDPDiscrete::num_observations(std::size_t agent) const {
  return pimpl_->d_->GetNrObservations(agent);
}
std::string MADPDecPOMDPDiscrete::agent_name(std::size_t agent) const {
  return pimpl_->d_->GetAgentNameByIndex(agent);
}
std::string MADPDecPOMDPDiscrete::state_name(std::size_t state_index) const {
  return pimpl_->d_->GetState(state_index)->GetName();
}
std::string MADPDecPOMDPDiscrete::action_name(std::size_t agent,
                                              std::size_t action_index) const {
  return pimpl_->d_->GetAction(agent, action_index)->GetName();
}
std::string MADPDecPOMDPDiscrete::observation_name(
    std::size_t agent, std::size_t observation_index) const {
  return pimpl_->d_->GetObservation(agent, observation_index)->GetName();
}
double MADPDecPOMDPDiscrete::initial_belief_at(std::size_t state) const {
  return pimpl_->d_->GetInitialStateProbability(state);
}
double MADPDecPOMDPDiscrete::reward(std::size_t state,
                                    std::size_t j_act) const {
  return pimpl_->d_->GetReward(state, j_act);
}

double MADPDecPOMDPDiscrete::discount() const {
  return pimpl_->d_->GetDiscount();
}

double MADPDecPOMDPDiscrete::observation_probability(std::size_t j_obs,
                                                     std::size_t state,
                                                     std::size_t j_act) const {
  return pimpl_->d_->GetObservationModelDiscretePtr()->Get(j_act, state, j_obs);
}
double MADPDecPOMDPDiscrete::transition_probability(std::size_t new_state,
                                                    std::size_t state,
                                                    std::size_t j_act) const {
  return pimpl_->d_->GetTransitionModelDiscretePtr()->Get(state, j_act,
                                                          new_state);
}
}  // namespace madpwrapper
}  // namespace npgi