#ifndef SPTS_HPP
#define SPTS_HPP
#include <boost/iterator/filter_iterator.hpp>
#include <map>
#include <vector>

namespace npgi {

template <typename ExecutionState, typename Sample>
class SPTS {
 public:
  using spts_container_t = typename std::multimap<ExecutionState, Sample>;
  using execution_state_t = ExecutionState;
  using sample_t = Sample;
  using spts_const_iterator = typename spts_container_t::const_iterator;


  SPTS() = default;
  SPTS(spts_container_t&& m) : m_(m) {}

  template <class Predicate>
  using filtered_const_iterator =
      boost::filter_iterator<Predicate, spts_const_iterator>;

  template <typename Predicate>
  filtered_const_iterator<Predicate> begin(const Predicate& pred) const {
    return boost::make_filter_iterator(pred, m_.begin(), m_.end());
  }

  template <typename Predicate>
  filtered_const_iterator<Predicate> end(const Predicate& pred) const {
    return boost::make_filter_iterator(pred, m_.end(), m_.end());
  }

  spts_const_iterator begin() const { return m_.begin(); }
  spts_const_iterator end() const { return m_.end(); }

  void insert(const ExecutionState& e, const Sample& x) { m_.emplace(e, x); }

 private:
  spts_container_t m_;
};
}  // namespace npgi

#endif  // SPTS_HPP