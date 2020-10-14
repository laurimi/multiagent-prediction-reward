#ifndef PUREPOLICY_TOOLS_HPP
#define PUREPOLICY_TOOLS_HPP
#include <string>

#include "PurePolicyExecutionState.hpp"
#include "decpomdp/discrete/DiscreteElements.hpp"
#include "decpomdp/discrete/DiscreteSpaces.hpp"
namespace npgi {
namespace policy {

template <typename Index = std::size_t>
class PurePolicyExecutionStateParser {
 public:
  using observation_type = DiscreteLocalObservation<Index>;
  using action_type = DiscreteLocalAction<Index>;

  PurePolicyExecutionStateParser(const DiscreteLocalActionSpace<Index>& A,
                                 const DiscreteLocalObservationSpace<Index>& Z)
      : A_(A), Z_(Z) {}

  std::pair<PurePolicyExecutionState<observation_type>, action_type> parse(
      const std::string& s) const {
    std::size_t d_pos = s.find(history_action_delimiter);

    const std::string history_str = s.substr(0, d_pos);
    const std::string action_str =
        s.substr(d_pos + history_action_delimiter.length(), s.length());

    return std::make_pair(parse_observation_history(history_str),
                          parse_action(action_str));
  }

  PurePolicyExecutionState<observation_type> parse_observation_history(
      std::string s) const {
    PurePolicyExecutionState<observation_type> e;

    // skip empty observation
    std::size_t pos =
        empty_observation_str.length() + observation_delimiter.length();
    s.erase(0, pos);

    // parse all others
    std::string token;
    while ((pos = s.find(observation_delimiter)) != std::string::npos) {
      token = s.substr(0, pos);
      s.erase(0, pos + observation_delimiter.length());

      auto it = std::find_if(
          Z_.begin(), Z_.end(),
          [&token](const observation_type& z) { return (z.name() == token); });
      if (it != Z_.end()) {
        e.observation_history_.emplace_back(*it);
      } else {
        std::string err = "could not find observation token: " + token;
        throw std::runtime_error(err);
      }
    }

    return e;
  }

  action_type parse_action(const std::string& s) const {
    auto it = std::find_if(A_.begin(), A_.end(), [&s](const action_type& a) {
      return (a.name() == s);
    });
    if (it != A_.end()) {
      return (*it);
    } else {
      std::string err = "could not find action token: " + s;
      throw std::runtime_error(err);
    }
  }

 private:
  const std::string empty_observation_str{"Oempty"};
  const std::string observation_delimiter{", "};
  const std::string history_action_delimiter{" --> "};
  const DiscreteLocalActionSpace<Index>& A_;
  const DiscreteLocalObservationSpace<Index>& Z_;
};

}  // namespace policy
}  // namespace npgi
#endif  // PUREPOLICY_TOOLS_HPP