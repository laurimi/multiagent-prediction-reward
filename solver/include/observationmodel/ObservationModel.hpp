#ifndef OBSERVATIONMODEL_HPP
#define OBSERVATIONMODEL_HPP
#include "core/CRTPHelper.hpp"
namespace npgi {

template <typename Derived>
struct observation_model_traits;

template <typename Derived>
struct ObservationModel : crtp_helper<Derived> {
  using state_type = typename observation_model_traits<Derived>::state_type;
  using pmf_type = typename observation_model_traits<Derived>::pmf_type;
  using action_type = typename observation_model_traits<Derived>::action_type;
  using observation_type =
      typename observation_model_traits<Derived>::observation_type;
  using scalar_type = typename observation_model_traits<Derived>::scalar_type;
  using timestep_type =
      typename observation_model_traits<Derived>::timestep_type;

  scalar_type observation_probability(const observation_type& o,
                                      const state_type& s, const action_type& a,
                                      const timestep_type t) const {
    return this->underlying().observation_probability(o, s, a, t);
  }

  observation_type sample_observation(const state_type& s, const action_type& a,
                                      const timestep_type t,
                                      const scalar_type random01) const {
    return this->underlying().sample_observation(s, a, t, random01);
  }

  scalar_type update(pmf_type& b, const action_type& a,
                     const observation_type& o, const timestep_type t) const {
    return this->underlying().update(b, a, o, t);
  }
};
}  // namespace npgi

#endif  // OBSERVATIONMODEL_HPP