#ifndef FSCEXECUTIONSTATE_HPP
#define FSCEXECUTIONSTATE_HPP
#include <vector>
#include <ostream>
#include "policy/base/ExecutionState.hpp"
namespace npgi {
namespace policy {

template <typename TimeStep, typename Node>
struct FSCExecutionState;

template <typename TimeStep, typename Node>
std::ostream& operator<<(std::ostream&,
                         const FSCExecutionState<TimeStep, Node>&);

template <typename TimeStep, typename Node>
struct execution_state_traits<FSCExecutionState<TimeStep, Node>> {
  using timestep_type = TimeStep;
};

template <typename TimeStep, typename Node>
struct FSCExecutionState : public ExecutionState<FSCExecutionState<TimeStep, Node>>
{
  using base_type = ExecutionState<FSCExecutionState<TimeStep, Node>>;
  using timestep_type = typename base_type::timestep_type;

  FSCExecutionState(timestep_type time, const std::vector<Node>& start_nodes)
  : time_(time), nodes_(start_nodes)
  {}

  bool operator<(const FSCExecutionState& other) const {
    if (time_ != other.time_)
      return (time_ < other.time_);
    else
      return (nodes_ < other.nodes_);
  }

  friend std::ostream& operator<< <TimeStep, Node>(std::ostream&, const FSCExecutionState<TimeStep, Node>&);

  timestep_type time_;
  std::vector<Node> nodes_;
};

template <typename TimeStep, typename Node>
std::ostream& operator<<(std::ostream& out, const FSCExecutionState<TimeStep, Node>& x)
{
  out << "time = " << x.time_ << ", nodes = [";
  for (auto i = x.nodes_.begin(); i != x.nodes_.end(); i++)
  {
    if (i != x.nodes_.begin()) {
      out << " ";
    }
    out << *i;
  }
  out << "]";
  return out;
}

}  // namespace policy
}  // namespace npgi

#endif  // FSCEXECUTIONSTATE_HPP