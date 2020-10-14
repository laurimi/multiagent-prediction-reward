#ifndef REWARDMODELINDEXED_HPP
#define REWARDMODELINDEXED_HPP
#include <map>
#include "decpomdp/discrete/DiscreteElements.hpp"
#include "RewardModel.hpp"
#include "RewardMatrix.hpp"

namespace npgi {

template <typename State, typename ActionIndex, typename Scalar, typename TimeStep>
class RewardModelFlat;

template <typename State, typename ActionIndex, typename Scalar, typename TimeStep>
struct reward_model_traits<RewardModelFlat<State, ActionIndex, Scalar, TimeStep>> {
  static_assert(std::is_integral<ActionIndex>::value,
                "ActionIndex must be integral type");
  static_assert(std::is_floating_point<Scalar>::value,
                "Scalar must be floating point type");
  using state_type = State;
  using action_type = DiscreteJointAction<ActionIndex>;
  using scalar_type = Scalar;
  using pmf_type = std::vector<scalar_type>;
  using timestep_type = TimeStep;
};

template <typename State = std::size_t, typename ActionIndex = std::size_t,
          typename Scalar = double, typename TimeStep = int>
struct RewardModelFlat
    : public RewardModel<RewardModelFlat<State, ActionIndex, Scalar, TimeStep>> {
  using Derived = RewardModelFlat<State, ActionIndex, Scalar, TimeStep>;
  using state_type = typename reward_model_traits<Derived>::state_type;
  using pmf_type = typename reward_model_traits<Derived>::pmf_type;
  using action_type = typename reward_model_traits<Derived>::action_type;
  using scalar_type = typename reward_model_traits<Derived>::scalar_type;
  using timestep_type = typename reward_model_traits<Derived>::timestep_type;

  using reward_matrix_type = RewardMatrix<action_type, state_type, scalar_type>;

  RewardModelFlat(const reward_matrix_type& R, scalar_type discount)
      : R_(R), custom_reward_m_(), discount_(discount) {}

  void set_custom_reward_matrix(const timestep_type& t, const reward_matrix_type& r)
  {
    custom_reward_m_.insert_or_assign(t,r);
  }

  const reward_matrix_type& reward_matrix(
      const timestep_type& t) const {
    auto im = custom_reward_m_.find(t);
    if (im == custom_reward_m_.end())
      return R_;
    else
      return im->second;
  }

  scalar_type reward(const pmf_type& b, const action_type& a,
                     const timestep_type& t) const {
    return reward_matrix(t).reward(b,a);
  }

  scalar_type reward(const state_type& s, const action_type& a,
                     const timestep_type& t) const {
    return reward_matrix(t).reward(s,a);
  }

  scalar_type max_reward(const timestep_type& t) const {
    return reward_matrix(t).max_reward();
  }

  scalar_type min_reward(const timestep_type& t) const {
    return reward_matrix(t).min_reward();
  }

  scalar_type discount() const { return discount_; }

 private:
  reward_matrix_type R_;
  std::map<timestep_type, reward_matrix_type> custom_reward_m_;
  scalar_type discount_;
};
}  // namespace npgi

#endif  // REWARDMODELINDEXED_HPP