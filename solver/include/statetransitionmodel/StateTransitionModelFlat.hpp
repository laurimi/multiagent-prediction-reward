#ifndef STATETRANSITIONMODELINDEXED_HPP
#define STATETRANSITIONMODELINDEXED_HPP
#include <map>
#include "StateTransitionModel.hpp"
#include "StateTransitionMatrix.hpp"


namespace npgi {

template <typename StateIndex, typename ActionIndex, typename Scalar, typename TimeStep>
class StateTransitionModelFlat;

template <typename StateIndex, typename ActionIndex, typename Scalar, typename TimeStep>
struct state_transition_model_traits<
    StateTransitionModelFlat<StateIndex, ActionIndex, Scalar, TimeStep>> {
  static_assert(std::is_integral<StateIndex>::value, "StateIndex must be integral type");
  static_assert(std::is_integral<ActionIndex>::value,
                "ActionIndex must be integral type");
  static_assert(std::is_floating_point<Scalar>::value,
                "Scalar must be floating point type");

  using state_type = StateIndex;
  using action_type = DiscreteJointAction<ActionIndex>;
  using scalar_type = Scalar;
  using pmf_type = std::vector<scalar_type>;
  using timestep_type = TimeStep;
};

template <typename StateIndex = std::size_t, typename ActionIndex = std::size_t,
          typename Scalar = double, typename TimeStep = int>
struct StateTransitionModelFlat
    : public StateTransitionModel<
          StateTransitionModelFlat<StateIndex, ActionIndex, Scalar, TimeStep>> {
  using Derived = StateTransitionModelFlat<StateIndex, ActionIndex, Scalar, TimeStep>;
  using state_type =
      typename state_transition_model_traits<Derived>::state_type;
  using pmf_type = typename state_transition_model_traits<Derived>::pmf_type;
  using action_type =
      typename state_transition_model_traits<Derived>::action_type;
  using scalar_type =
      typename state_transition_model_traits<Derived>::scalar_type;
  using timestep_type =
      typename state_transition_model_traits<Derived>::timestep_type;

  using state_transition_matrix_type =
      StateTransitionMatrix<action_type, StateIndex, Scalar>;

  StateTransitionModelFlat(const state_transition_matrix_type& T)
      : T_(T), custom_state_transition_m_() {}

  void set_custom_state_transition_matrix(const timestep_type& t, const state_transition_matrix_type& a)
  {
    custom_state_transition_m_.insert_or_assign(t,a);
  }

  const state_transition_matrix_type& state_transition_matrix(const timestep_type& t) const
  {
    auto im = custom_state_transition_m_.find(t);
    if (im == custom_state_transition_m_.end())
      return T_;
    else
      return im->second;
  }

  scalar_type transition_probability(const state_type& next,
                                     const state_type& old,
                                     const action_type& a,
                                     const timestep_type t) const {
    return state_transition_matrix(t).transition_probability(next, old, a);
  }

  state_type sample_next_state(const state_type& old, const action_type& a,
                               const timestep_type t,
                               const scalar_type random01) const {
    return state_transition_matrix(t).sample_next_state(old, a, random01);
  }

  pmf_type predict(const pmf_type& b, const action_type& a,
                   const timestep_type t) const {
    return state_transition_matrix(t).predict(b, a);
  }

  std::size_t num_states() const { return T_.rows(); }

 private:
  state_transition_matrix_type T_;
  std::map<timestep_type, state_transition_matrix_type> custom_state_transition_m_;
};
}  // namespace npgi
#endif  // STATETRANSITIONMODELINDEXED_HPP