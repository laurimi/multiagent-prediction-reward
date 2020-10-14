#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP
#include <optional>
#include <ostream>
#include <random>
namespace npgi {
template <typename DecPOMDP>
class Simulator;
template <typename DecPOMDP>
struct sample_t;
template <typename DecPOMDP>
std::ostream& operator<<(std::ostream&, const sample_t<DecPOMDP>&);

template <typename DecPOMDP>
struct sample_t {
  friend class Simulator<DecPOMDP>;
  using timestep_t = typename DecPOMDP::timestep_type;
  using state_t = typename DecPOMDP::state_type;
  using joint_action_t = typename DecPOMDP::joint_action_type;
  using joint_observation_t = typename DecPOMDP::joint_observation_type;
  using scalar_t = typename DecPOMDP::scalar_type;

  sample_t(timestep_t t, state_t s, scalar_t cumulative_reward = 0.0)
      : time_(t),
        state_(s),
        cumulative_reward_(cumulative_reward),
        last_action_(),
        last_observation_(),
        last_reward_() {}

  timestep_t time() const { return time_; }
  state_t state() const { return state_; }
  scalar_t cumulative_reward() const { return cumulative_reward_; }
  joint_observation_t last_observation() const {
    if (!last_observation_) {
      throw std::runtime_error("requesting non-existent last observation");
    }
    return last_observation_.value();
  }

  scalar_t last_reward() const {
    if (!last_reward_) {
      throw std::runtime_error("requesting non-existent last reward");
    }
    return last_reward_.value();
  }

 private:
  timestep_t time_;
  state_t state_;
  scalar_t cumulative_reward_;
  std::optional<joint_action_t> last_action_;
  std::optional<joint_observation_t> last_observation_;
  std::optional<scalar_t> last_reward_;

  friend std::ostream& operator<<<DecPOMDP>(std::ostream&,
                                            const sample_t<DecPOMDP>&);
};

template <typename DecPOMDP>
std::ostream& operator<<(std::ostream& os, const sample_t<DecPOMDP>& x) {
  os << "{t: " << x.time_ << ", s: " << x.state_
     << ", cr: " << x.cumulative_reward_ << ", la: ";
  if (x.last_action_)
    os << x.last_action_.value();
  else
    os << "[n/a]";
  os << ", lo: ";
  if (x.last_observation_)
    os << x.last_observation_.value();
  else
    os << "[n/a]";
  os << ", lr: ";
  if (x.last_reward_)
    os << x.last_reward_.value();
  else
    os << "[n/a]";

  os << "}";
  return os;
}

template <typename DecPOMDP>
class Simulator {
 public:
  using timestep_t = typename DecPOMDP::timestep_type;
  using state_t = typename DecPOMDP::state_type;
  using joint_action_t = typename DecPOMDP::joint_action_type;
  using joint_observation_t = typename DecPOMDP::joint_observation_type;
  using scalar_t = typename DecPOMDP::scalar_type;

  using joint_belief_t = typename DecPOMDP::joint_belief_type;

  Simulator(const DecPOMDP* d, timestep_t end_time)
      : d_(d), end_time_(end_time) {}

  template <typename RandomNumberGenerator>
  sample_t<DecPOMDP> step(sample_t<DecPOMDP> current,
                          joint_action_t current_action,
                          RandomNumberGenerator& g) const {
    if (current.time_ >= end_time_)
      throw std::runtime_error(
          "DecPOMDPSimulator::step called with t >= end_time_");

    std::uniform_real_distribution<scalar_t> dist01;

    current.last_reward_ =
        d_->R().reward(current.state_, current_action, current.time_);
    current.state_ = d_->Tr().sample_next_state(current.state_, current_action,
                                                current.time_, dist01(g));
    ++current.time_;
    current.last_action_ = current_action;
    current.last_observation_ = d_->Ob().sample_observation(
        current.state_, current_action, current.time_, dist01(g));

    current.cumulative_reward_ += current.last_reward_.value();

    return current;
  }

  template <typename RandomNumberGenerator>
  static sample_t<DecPOMDP> sample_initial_simulation_state(
      const joint_belief_t& initial_state_distribution, timestep_t t0,
      RandomNumberGenerator& g) {
    std::uniform_real_distribution<scalar_t> dist01;
    return sample_t<DecPOMDP>(
        t0, initial_state_distribution.sample_state(dist01(g)));
  }

 private:
  const DecPOMDP* d_;
  timestep_t end_time_;
};
}  // namespace npgi
#endif  // SIMULATOR_HPP