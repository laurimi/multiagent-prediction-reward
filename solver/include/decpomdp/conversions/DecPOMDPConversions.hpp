#ifndef DECPOMDPCONVERSIONS_HPP
#define DECPOMDPCONVERSIONS_HPP
#include "decpomdp/discrete/DecPOMDPFlat.hpp"
#include "decpomdp/discrete/DiscreteSpaces.hpp"
#include "rewardmodel/LinearizedNegEntropy.hpp"
namespace npgi {

struct conversion_settings {
  std::vector<std::vector<double>> linearization_points;
  int time_of_prediction_actions;
};

template <typename DecPOMDP>
DecPOMDP add_prediction_actions(const DecPOMDP& rhopomdp,
                                const conversion_settings& s) {
  auto joint_action_space = rhopomdp.A();
  auto joint_observation_space = rhopomdp.Z();
  auto transition_model = rhopomdp.Tr();
  auto observation_model = rhopomdp.Ob();
  auto reward_model = rhopomdp.R();

  // create the new set of local prediction actions
  DiscreteLocalActionSpace<> pred;
  for (std::size_t pred_idx = 0; pred_idx < s.linearization_points.size();
       ++pred_idx)
    pred.insert(pred_idx, DiscreteLocalAction<>(
                              pred_idx, "predict" + std::to_string(pred_idx)));

  // create new set of local observations
  DiscreteLocalObservationSpace<> null_sp;
  null_sp.insert(0, DiscreteLocalObservation<>(0, "NULL"));

  // create joint action and observation spaces for prediction time and the step
  // after, respectively
  using agent_type = typename DecPOMDP::joint_action_space_type::agent_type;
  std::map<agent_type, DiscreteLocalActionSpace<>> local_pred_spaces;
  std::map<agent_type, DiscreteLocalObservationSpace<>> local_null_obs_spaces;
  for (std::size_t agent_idx = 0; agent_idx < joint_action_space.num_agents();
       ++agent_idx) {
    local_pred_spaces.emplace(Agent<>(agent_idx), pred);
    local_null_obs_spaces.emplace(Agent<>(agent_idx), null_sp);
  }

  DiscreteJointActionSpace<> joint_pred_space(local_pred_spaces);
  DiscreteJointObservationSpace<> joint_null_space(local_null_obs_spaces);

  std::cout << joint_pred_space.size() << " joint prediction actions, "
            << joint_null_space.size() << " joint null observations\n";

  // create transition, reward, and observation matrices
  using local_action_type =
      typename DecPOMDP::joint_action_space_type::local_action_type;
  using joint_action_type =
      typename DecPOMDP::joint_action_space_type::joint_action_type;
  using matrix_type = typename DecPOMDP::transition_model_type::
      state_transition_matrix_type::matrix_type;
  using vector_type =
      typename DecPOMDP::reward_model_type::reward_matrix_type::vector_type;
  std::map<joint_action_type, matrix_type> t;
  std::map<joint_action_type, matrix_type> o;
  std::map<joint_action_type, vector_type> r;
  for (auto it = joint_pred_space.begin(), iend = joint_pred_space.end();
       it != iend; ++it) {
    // transition matrix: diagonal
    t[*it] = matrix_type::Identity(transition_model.num_states(),
                                   transition_model.num_states());
    // observation matrix: prob 1.0 for NULL
    o[*it] = matrix_type::Constant(1, transition_model.num_states(), 1.0);

    vector_type pred_reward = vector_type::Zero(transition_model.num_states());
    const double per_agent_contrib =
        1.0 / static_cast<double>(joint_action_space.num_agents());

    std::map<agent_type, local_action_type> local_actions;
    for (std::size_t agent_idx = 0; agent_idx < joint_action_space.num_agents();
         ++agent_idx) {
      local_action_type local_pred_act =
          joint_pred_space.get_local_element(*it, agent_idx);
      // std::cout << local_pred_act << " ";

      // get the linearized vector for this local prediction, add contribution
      // to pred_reward
      const std::vector<double> L = negative_entropy::linearizing_hyperplane(
          s.linearization_points.at(local_pred_act.index()));
      Eigen::Map<const vector_type> lv(L.data(), L.size());
      pred_reward += per_agent_contrib * lv;
    }
    r[*it] = pred_reward;

    // std::cout << "-- pred_reward: " << pred_reward << "\n";
  }

  // modify the original spaces appropriately
  joint_action_space.set_custom_joint_action_space(
      s.time_of_prediction_actions,
      DiscreteJointActionSpace<>(joint_pred_space));
  joint_observation_space.set_custom_joint_observation_space(
      s.time_of_prediction_actions + 1, joint_null_space);
  transition_model.set_custom_state_transition_matrix(
      s.time_of_prediction_actions, t);
  observation_model.set_custom_observation_matrix(
      s.time_of_prediction_actions + 1, o);
  reward_model.set_custom_reward_matrix(s.time_of_prediction_actions, r);

  return DecPOMDP(joint_action_space, joint_observation_space, transition_model,
                  observation_model, reward_model);
}

template <typename DecPOMDP>
DecPOMDP update_prediction_actions(const DecPOMDP& rhopomdp,
                                   const conversion_settings& s) {
  auto joint_action_space = rhopomdp.A();
  auto joint_observation_space = rhopomdp.Z();
  auto transition_model = rhopomdp.Tr();
  auto observation_model = rhopomdp.Ob();
  auto reward_model = rhopomdp.R();

  using local_action_type =
      typename DecPOMDP::joint_action_space_type::local_action_type;
  using joint_action_type =
      typename DecPOMDP::joint_action_space_type::joint_action_type;
  using vector_type =
      typename DecPOMDP::reward_model_type::reward_matrix_type::vector_type;
  std::map<joint_action_type, vector_type> r;

  const auto joint_pred_space = joint_action_space.joint_action_space_at_time(
      s.time_of_prediction_actions);
  for (auto it = joint_pred_space.begin(), iend = joint_pred_space.end();
       it != iend; ++it) {
    vector_type pred_reward = vector_type::Zero(transition_model.num_states());
    const double per_agent_contrib =
        1.0 / static_cast<double>(joint_action_space.num_agents());

    using agent_type = typename DecPOMDP::joint_action_space_type::agent_type;
    std::map<agent_type, local_action_type> local_actions;
    for (std::size_t agent_idx = 0; agent_idx < joint_action_space.num_agents();
         ++agent_idx) {
      local_action_type local_pred_act =
          joint_pred_space.get_local_element(*it, agent_idx);
      // std::cout << local_pred_act << " ";

      // get the linearized vector for this local prediction, add contribution
      // to pred_reward
      const std::vector<double> L = negative_entropy::linearizing_hyperplane(
          s.linearization_points.at(local_pred_act.index()));
      Eigen::Map<const vector_type> lv(L.data(), L.size());
      pred_reward += per_agent_contrib * lv;
    }
    r[*it] = pred_reward;

    // std::cout << "-- pred_reward: " << pred_reward << "\n";
  }

  reward_model.set_custom_reward_matrix(s.time_of_prediction_actions, r);
  return DecPOMDP(joint_action_space, joint_observation_space, transition_model,
                  observation_model, reward_model);
}

}  // namespace npgi

#endif  // DECPOMDPCONVERSIONS_HPP