#ifndef PUREPOLICYEXECUTIONSTATE_HPP
#define PUREPOLICYEXECUTIONSTATE_HPP
namespace npgi {
namespace policy {

template <typename LocalObservation>
struct PurePolicyExecutionState;

template <typename LocalObservation>
std::ostream& operator<<(std::ostream&,
                         const PurePolicyExecutionState<LocalObservation>&);

template <typename LocalObservation>
struct PurePolicyExecutionState {
  PurePolicyExecutionState() : observation_history_() {}
  PurePolicyExecutionState(const std::vector<LocalObservation>& observation_history)
      : observation_history_(observation_history) {
      }

  bool operator<(const PurePolicyExecutionState& other) const {
    return (observation_history_ < other.observation_history_);
  }

  bool operator==(const PurePolicyExecutionState& other) const {
    return (observation_history_ == other.observation_history_);
  }

  friend std::ostream& operator<<<LocalObservation>(
      std::ostream&, const PurePolicyExecutionState<LocalObservation>&);

  std::vector<LocalObservation> observation_history_;
};

template <typename LocalObservation>
std::ostream& operator<<(std::ostream& out,
                         const PurePolicyExecutionState<LocalObservation>& x) {
  out << "(";
  for (auto i = x.observation_history_.begin();
       i != x.observation_history_.end(); i++) {
    if (i != x.observation_history_.begin()) {
      out << " ";
    }
    out << *i;
  }
  out << ")";
  return out;
}

}  // namespace policy
}  // namespace npgi
#endif  // PUREPOLICYEXECUTIONSTATE_HPP