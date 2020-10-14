#ifndef JOINTACTIONSPACE_HPP
#define JOINTACTIONSPACE_HPP
#include <map>
#include "core/CRTPHelper.hpp"
namespace npgi {

template <typename Derived>
struct joint_action_space_traits;

template <typename Derived>
struct JointActionSpace : crtp_helper<Derived> {
  using agent_type =
      typename joint_action_space_traits<Derived>::agent_type;
  using joint_action_type =
      typename joint_action_space_traits<Derived>::joint_action_type;
  using local_action_type =
      typename joint_action_space_traits<Derived>::local_action_type;
  using timestep_type =
      typename joint_action_space_traits<Derived>::timestep_type;

  joint_action_type get_joint_action(
      const std::map<agent_type, local_action_type>& locals,
      timestep_type t) const {
    return this->underlying().get_joint_action(locals, t);
  }

  local_action_type get_local_action(const joint_action_type& joint,
                                     timestep_type t,
                                     const agent_type& agent) const {
    this->underlying().get_local_action(joint, t, agent);
  }
};
}  // namespace npgi

#endif  // JOINTACTIONSPACE_HPP