#ifndef STATETRANSITIONMODEL_HPP
#define STATETRANSITIONMODEL_HPP
#include "core/CRTPHelper.hpp"
namespace npgi {

template <typename Derived>
struct state_transition_model_traits;

template <typename Derived>
struct StateTransitionModel : crtp_helper<Derived> {
  using state_type =
      typename state_transition_model_traits<Derived>::state_type;
  using pmf_type = typename state_transition_model_traits<Derived>::pmf_type;
  using action_type =
      typename state_transition_model_traits<Derived>::action_type;
  using scalar_type =
      typename state_transition_model_traits<Derived>::scalar_type;
  using timestep_type =
      typename state_transition_model_traits<Derived>::timestep_type;

  scalar_type transition_probability(const state_type& next,
                                     const state_type& old,
                                     const action_type& a,
                                     const timestep_type t) const {
    return this->underlying().transition_probability(next, old, a, t);
  }

  state_type sample_next_state(const state_type& old, const action_type& a,
                               const timestep_type t,
                               const scalar_type random01) const {
    return this->underlying().sample_next_state(old, a, t, random01);
  }

  pmf_type predict(const pmf_type& b, const action_type& a,
                   const timestep_type t) const {
    return this->underlying().predict(b, a, t);
  }
};

}  // namespace npgi

#endif  // STATETRANSITIONMODEL_HPP