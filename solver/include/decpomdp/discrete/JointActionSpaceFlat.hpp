#ifndef JOINTACTIONSPACEFLAT_HPP
#define JOINTACTIONSPACEFLAT_HPP
#include <vector>

#include "DiscreteElements.hpp"
#include "DiscreteSpaces.hpp"
#include "core/CRTPHelper.hpp"
#include "decpomdp/base/JointActionSpace.hpp"
namespace npgi {

template <typename Index, typename TimeStep>
class JointActionSpaceFlat;

template <typename Index, typename TimeStep>
struct joint_action_space_traits<JointActionSpaceFlat<Index, TimeStep>> {
  using agent_type = Agent<Index>;
  using joint_action_type = DiscreteJointAction<Index>;
  using local_action_type = DiscreteLocalAction<Index>;
  using timestep_type = TimeStep;
};

template <typename Index = std::size_t, typename TimeStep = int>
struct JointActionSpaceFlat
    : public JointActionSpace<JointActionSpaceFlat<Index, TimeStep>> {
  using derived_t = JointActionSpaceFlat<Index, TimeStep>;
  using agent_type = typename joint_action_space_traits<derived_t>::agent_type;
  using joint_action_type =
      typename joint_action_space_traits<derived_t>::joint_action_type;
  using local_action_type =
      typename joint_action_space_traits<derived_t>::local_action_type;
  using timestep_type =
      typename joint_action_space_traits<derived_t>::timestep_type;
  using index_type = Index;

  using joint_action_space_type =
      DiscreteJointActionSpace<index_type, local_action_type,
                               joint_action_type>;

  using joint_action_space_type_const_iter =
      typename joint_action_space_type::const_iterator;
  using local_action_space_type =
      typename joint_action_space_type::local_space_type;

  JointActionSpaceFlat(const joint_action_space_type& d)
      : d_(d), custom_action_space_m_() {}

  void set_custom_joint_action_space(const timestep_type& t,
                                     const joint_action_space_type& a) {
    custom_action_space_m_.insert_or_assign(t, a);
  }

  const joint_action_space_type& joint_action_space_at_time(
      const timestep_type& t) const {
    auto im = custom_action_space_m_.find(t);
    if (im == custom_action_space_m_.end())
      return d_;
    else
      return im->second;
  }

  const local_action_space_type& local_action_space(
      const agent_type& i, const timestep_type& t) const {
    return joint_action_space_at_time(t).get_local_space(i);
  }

  joint_action_type get_joint_action(
      const std::map<agent_type, local_action_type>& locals,
      timestep_type t) const {
    return joint_action_space_at_time(t).get_joint_element(locals);
  }

  local_action_type get_local_action(const joint_action_type& joint,
                                     timestep_type t,
                                     const agent_type& agent) const {
    return joint_action_space_at_time(t).get_local_element(joint, agent);
  }

  std::size_t num_agents() const { return d_.num_agents(); }

 private:
  joint_action_space_type d_;
  std::map<timestep_type, joint_action_space_type> custom_action_space_m_;
};

}  // namespace npgi

#endif  // JOINTACTIONSPACEFLAT_HPP