#ifndef JOINTOBSERVATIONSPACEFLAT_HPP
#define JOINTOBSERVATIONSPACEFLAT_HPP
#include "core/CRTPHelper.hpp"
#include "decpomdp/base/JointObservationSpace.hpp"
#include "DiscreteElements.hpp"
#include "DiscreteSpaces.hpp"
namespace npgi {

template <typename Index, typename TimeStep>
class JointObservationSpaceFlat;

template <typename Index, typename TimeStep>
struct joint_observation_space_traits<
    JointObservationSpaceFlat<Index, TimeStep>> {
  using agent_type = Agent<Index>;
  using joint_observation_type = DiscreteJointObservation<Index>;
  using local_observation_type = DiscreteLocalObservation<Index>;
  using timestep_type = TimeStep;
};

template <typename Index = std::size_t, typename TimeStep = int>
struct JointObservationSpaceFlat
    : public JointObservationSpace<JointObservationSpaceFlat<Index, TimeStep>> {
       using derived_t = JointObservationSpaceFlat<Index, TimeStep>;
       using agent_type = typename joint_observation_space_traits<derived_t>::agent_type;
       using joint_observation_type = typename joint_observation_space_traits<derived_t>::joint_observation_type;
       using local_observation_type = typename joint_observation_space_traits<derived_t>::local_observation_type;
       using timestep_type = typename joint_observation_space_traits<derived_t>::timestep_type;
       using index_type = Index;

       using joint_observation_space_type = DiscreteJointObservationSpace<index_type, local_observation_type, joint_observation_type>;
       using joint_observation_space_type_const_iter = typename joint_observation_space_type::const_iterator;
       

       using local_observation_space_type = typename joint_observation_space_type::local_space_type;

       JointObservationSpaceFlat(const joint_observation_space_type& d) : d_(d), custom_observation_space_m_() {}

       void set_custom_joint_observation_space(const timestep_type& t, const joint_observation_space_type& z)
       {
         custom_observation_space_m_.insert_or_assign(t,z);
       }

       joint_observation_space_type_const_iter begin(
           const timestep_type& t) const {
         return joint_observation_space_at_time(t).begin();
       }

       joint_observation_space_type_const_iter end(
           const timestep_type& t) const {
         return joint_observation_space_at_time(t).end();
       }

       const joint_observation_space_type& joint_observation_space_at_time(
           const timestep_type& t) const {
         auto im = custom_observation_space_m_.find(t);
         if (im == custom_observation_space_m_.end())
           return d_;
         else
           return im->second;
       }

       const local_observation_space_type& local_observation_space(
           const agent_type& i, const timestep_type& t) const {
         return joint_observation_space_at_time(t).get_local_space(i);
       }

       joint_observation_type get_joint_observation(
           const std::map<agent_type, local_observation_type>& locals,
           timestep_type t) const {
         return joint_observation_space_at_time(t).get_joint_element(locals);
       }

       local_observation_type get_local_observation(
           const joint_observation_type& joint, timestep_type t,
           const agent_type& agent) const {
         return joint_observation_space_at_time(t).get_local_element(joint, agent);
       }

      private:
       joint_observation_space_type d_;
       std::map<timestep_type, joint_observation_space_type> custom_observation_space_m_;
};
}  // namespace npgi

#endif  // JOINTOBSERVATIONSPACEFLAT_HPP