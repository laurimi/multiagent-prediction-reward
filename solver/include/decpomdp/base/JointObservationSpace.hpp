#ifndef JOINTOBSERVATIONSPACE_HPP
#define JOINTOBSERVATIONSPACE_HPP
#include <map>
#include "core/CRTPHelper.hpp"
namespace npgi {

template <typename Derived>
struct joint_observation_space_traits;

template <typename Derived>
struct JointObservationSpace : crtp_helper<Derived> {
  using agent_type =
      typename joint_observation_space_traits<Derived>::agent_type;
  using joint_observation_type =
      typename joint_observation_space_traits<Derived>::joint_observation_type;
  using local_observation_type =
      typename joint_observation_space_traits<Derived>::local_observation_type;
  using timestep_type =
      typename joint_observation_space_traits<Derived>::timestep_type;

  joint_observation_type get_joint_observation(
      const std::map<agent_type, local_observation_type>& locals,
      timestep_type t) const {
    return this->underlying().get_joint_observation(locals, t);
  }

  local_observation_type get_local_observation(
      const joint_observation_type& joint, timestep_type t,
      const agent_type& agent) const {
    this->underlying().get_local_observation(joint, agent, t);
  }
};
}  // namespace npgi

#endif  // JOINTOBSERVATIONSPACE_HPP