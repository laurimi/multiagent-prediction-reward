#ifndef UCB1_HPP
#define UCB1_HPP
#include <cmath>
#include <limits>
#include <ostream>
#include "utilities/Common.hpp"
namespace npgi {
namespace ucb1 {

template <typename Scalar>
Scalar score(const npgi::CumulativeMovingAverage<Scalar>& s, Scalar log_n_total) {
  if ((s.count() == 0))
    return std::numeric_limits<Scalar>::infinity();

  return s.average() +
         std::sqrt(2.0 * log_n_total /
                   static_cast<Scalar>(s.count()));
}

template <typename Action, typename Scalar>
class UCB1 {
 public:
  using statistics_type = npgi::CumulativeMovingAverage<Scalar>;
  using action_statistic_pair = std::pair<Action, statistics_type>;
  using statistics_map_type = std::vector<action_statistic_pair>;
  using statistics_map_value_type = typename statistics_map_type::value_type;
  // using statistics_map_type = std::map<Action, statistics_type>;
  // using statistics_map_value_type = typename statistics_map_type::value_type;

  template <typename ActionIterator>
  UCB1(ActionIterator first, ActionIterator second) : stats(), n_total(0) {
    for (; first != second; ++first) stats.emplace_back(std::make_pair(*first, statistics_type()));
  }

  const Action& ucb_policy() const {
    if (n_total == 0)
      return stats.begin()->first;

  	const Scalar log_n_total = std::log(static_cast<Scalar>(n_total));
    auto it = std::max_element(
        stats.begin(), stats.end(), [&log_n_total](const statistics_map_value_type& x,
                                           const statistics_map_value_type& y) {
          return (score(x.second, log_n_total) < score(y.second, log_n_total));
        });
    return it->first;
  }

  const Action& greedy_policy() const {
    auto it = std::max_element(
        stats.begin(), stats.end(), [](const statistics_map_value_type& x,
                                       const statistics_map_value_type& y) {
          return (x.second.average() < y.second.average());
        });
    return it->first;
  }

  void update(const Action& a, Scalar reward) {
    // if ((reward < 0.0) || (reward > 1.0)) {
    //   std::ostringstream os;
    //   os << "reward " << reward << " not in range [0, 1]";
    //   throw std::runtime_error(os.str());
    // }

    auto it = std::find_if(
        stats.begin(), stats.end(),
        [&a](const statistics_map_value_type& x) { return (a == x.first); });
    it->second.update(reward);
    ++n_total;
  }

  std::size_t total_count() const { return n_total; }

 private:
  statistics_map_type stats;
  std::size_t n_total;
};
}  // namespace ucb1
}  // namespace npgi
#endif  // UCB1_HPP