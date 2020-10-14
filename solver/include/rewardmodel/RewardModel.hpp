#ifndef REWARDMODEL_HPP
#define REWARDMODEL_HPP
#include "core/CRTPHelper.hpp"

namespace npgi {

template <typename Derived>
struct reward_model_traits;

template <typename Derived>
struct RewardModel : crtp_helper<Derived> {
  using state_type = typename reward_model_traits<Derived>::state_type;
  using pmf_type = typename reward_model_traits<Derived>::pmf_type;
  using action_type = typename reward_model_traits<Derived>::action_type;
  using scalar_type = typename reward_model_traits<Derived>::scalar_type;
  using timestep_type = typename reward_model_traits<Derived>::timestep_type;

  scalar_type reward(const pmf_type& b, const action_type& a,
                     const timestep_type& t) const {
    return this->underlying().reward(b, a, t);
  }

  scalar_type reward(const state_type& s, const action_type& a,
                     const timestep_type& t) const {
    return this->underlying().reward(s, a, t);
  }

  scalar_type discount() const { return this->underlying().discount(); }
};
}  // namespace npgi

#endif  // REWARDMODEL_HPP