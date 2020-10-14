#ifndef REWARDMATRIX_HPP
#define REWARDMATRIX_HPP
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <map>
namespace npgi {

template <typename Action, typename StateIndex, typename Scalar>
class RewardMatrix {
 public:
  using pmf_type = std::vector<Scalar>;
  using vector_type = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  RewardMatrix(const std::map<Action, vector_type>& R)
      : R_(R),
        max_reward_(
            std::max_element(
                R.begin(), R.end(),
                [](const typename std::map<Action, vector_type>::value_type& x,
                   const typename std::map<Action, vector_type>::value_type&
                       y) {
                  return (x.second.maxCoeff() < y.second.maxCoeff());
                })
                ->second.maxCoeff()),
        min_reward_(
            std::min_element(
                R.begin(), R.end(),
                [](const typename std::map<Action, vector_type>::value_type& x,
                   const typename std::map<Action, vector_type>::value_type&
                       y) {
                  return (x.second.minCoeff() < y.second.minCoeff());
                })
                ->second.minCoeff()) {
        }

  Scalar reward(const pmf_type& b, const Action& a) const {
    Eigen::Map<const vector_type> bv(b.data(), b.size());
    return R_.at(a).dot(bv);
  }

  Scalar reward(const StateIndex& s, const Action& a) const {
    return R_.at(a)(s);
  }

  Scalar max_reward() const { return max_reward_; }
  Scalar min_reward() const { return min_reward_; }

 private:
  std::map<Action, vector_type> R_;
  Scalar max_reward_;
  Scalar min_reward_;
};
}  // namespace npgi

#endif  // REWARDMATRIX_HPP