#ifndef PUREPOLICY_HPP
#define PUREPOLICY_HPP
#include "PurePolicyExecutionState.hpp"

namespace npgi {
namespace policy {

template <typename LocalAction, typename LocalObservation>
struct PurePolicy {
 public:
  PurePolicy() : m_() {}

  void set_action(const PurePolicyExecutionState<LocalObservation>& es,
                  const LocalAction& a) {
    m_[es] = a;
  }

  const LocalAction& get_action(
      const PurePolicyExecutionState<LocalObservation>& es) const {
    return m_.at(es);
  }

  PurePolicyExecutionState<LocalObservation> initial_execution_state() const {
    return PurePolicyExecutionState<LocalObservation>();
  }

  void update_execution_state(PurePolicyExecutionState<LocalObservation>& e,
                              const LocalAction& a,
                              const LocalObservation& o) const {
    (void)a;  // a is not needed for the state update
    e.observation_history_.emplace_back(o);
  }

 private:
  std::map<PurePolicyExecutionState<LocalObservation>, LocalAction> m_;
};

}  // namespace policy
}  // namespace npgi

#endif  // PUREPOLICY_HPP