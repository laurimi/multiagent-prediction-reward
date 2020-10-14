#ifndef OBSERVATIONMODELINDEXED_HPP
#define OBSERVATIONMODELINDEXED_HPP
#include <map>
#include "ObservationMatrix.hpp"
#include "ObservationModel.hpp"

namespace npgi {

template <typename State, typename ActionIndex, typename ObservationIndex,
          typename Scalar, typename TimeStep>
class ObservationModelFlat;

template <typename State, typename ActionIndex, typename ObservationIndex,
          typename Scalar, typename TimeStep>
struct observation_model_traits<ObservationModelFlat<
    State, ActionIndex, ObservationIndex, Scalar, TimeStep>> {
  static_assert(std::is_integral<State>::value, "State must be integral type");
  static_assert(std::is_integral<ActionIndex>::value,
                "ActionIndex must be integral type");
  static_assert(std::is_integral<ObservationIndex>::value,
                "ObservationIndex must be integral type");
  static_assert(std::is_floating_point<Scalar>::value,
                "Scalar must be floating point type");

  using state_type = State;
  using action_type = DiscreteJointAction<ActionIndex>;
  using observation_type = DiscreteJointObservation<ObservationIndex>;
  using scalar_type = Scalar;
  using pmf_type = std::vector<scalar_type>;
  using timestep_type = TimeStep;
};

template <typename State = std::size_t, typename ActionIndex = std::size_t,
          typename ObservationIndex = std::size_t, typename Scalar = double,
          typename TimeStep = int>
struct ObservationModelFlat
    : public ObservationModel<ObservationModelFlat<
          State, ActionIndex, ObservationIndex, Scalar, TimeStep>> {
  using Derived = ObservationModelFlat<State, ActionIndex, ObservationIndex,
                                       Scalar, TimeStep>;
  using state_type = typename observation_model_traits<Derived>::state_type;
  using pmf_type = typename observation_model_traits<Derived>::pmf_type;
  using action_type = typename observation_model_traits<Derived>::action_type;
  using observation_type =
      typename observation_model_traits<Derived>::observation_type;
  using scalar_type = typename observation_model_traits<Derived>::scalar_type;
  using timestep_type =
      typename observation_model_traits<Derived>::timestep_type;

  using observation_matrix_type =
      ObservationMatrix<action_type, observation_type, State, Scalar>;

  ObservationModelFlat(const observation_matrix_type& O) : O_(O) {}

  void set_custom_observation_matrix(const timestep_type& t,
                                     const observation_matrix_type& o) {
    custom_observation_m_.insert_or_assign(t,o);
  }

  const observation_matrix_type& observation_matrix(
      const timestep_type& t) const {
    auto im = custom_observation_m_.find(t);
    if (im == custom_observation_m_.end())
      return O_;
    else
      return im->second;
  }

  scalar_type observation_probability(const observation_type& o,
                                      const state_type& s, const action_type& a,
                                      const timestep_type t) const {
    return observation_matrix(t).observation_probability(o, s, a);
  }

  observation_type sample_observation(const state_type& s, const action_type& a,
                                      const timestep_type t,
                                      const scalar_type random01) const {
    return observation_matrix(t).sample_observation(s, a, random01);
  }

  scalar_type update(pmf_type& b, const action_type& a,
                     const observation_type& o, const timestep_type t) const {
    return observation_matrix(t).update(b, a, o);
  }

 private:
  observation_matrix_type O_;
  std::map<timestep_type, observation_matrix_type> custom_observation_m_;
};
}  // namespace npgi
#endif  // OBSERVATIONMODELINDEXED_HPP