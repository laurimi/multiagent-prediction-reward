#ifndef DISCRETESPACES_HPP
#define DISCRETESPACES_HPP
#include <algorithm>
#include <iterator>
#include <map>
#include <vector>
#include "DiscreteElements.hpp"
#include "utilities/IndexSpace.hpp"

namespace npgi {
template <typename Index, typename Element>
class DiscreteSpace {
  friend class const_iterator;

 public:
  using local_element_type = Element;
  using index_type = Index;
  using size_type = std::size_t;
  using action_map_type = std::map<index_type, local_element_type>;
  using action_map_type_const_iterator =
      typename action_map_type::const_iterator;

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = Element;
    using difference_type = long long;
    using pointer = Element*;
    using reference = Element&;
    using const_reference = const Element&;

    const_iterator(const const_iterator& other)
        : m_(other.m_), it_(other.it_) {}

    const_iterator& operator++() {
      if (it_ != m_.end())
        ++it_;
      else
        it_ = m_.end();
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator ret(*this);
      ++(*this);
      return ret;
    }

    const_iterator& operator--() {
      if (it_ != m_.begin())
        --it_;
      else
        it_ = m_.begin();
      return *this;
    }

    const_iterator operator--(int) {
      const_iterator ret(*this);
      --(*this);
      return ret;
    }

    static const_iterator make_begin(const action_map_type& m) {
      return const_iterator(m, m.begin());
    }

    static const_iterator make_end(const action_map_type& m) {
      return const_iterator(m, m.end());
    }

    const_reference operator*() const { return it_->second; }

    bool operator==(const const_iterator& other) const {
      return ((&m_ == &other.m_) && (it_ == other.it_));
    }
    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

   private:
    const_iterator(const action_map_type& m, action_map_type_const_iterator it)
        : m_(m), it_(it) {}
    const action_map_type& m_;
    action_map_type_const_iterator it_;
  };

  const_iterator begin() const { return const_iterator::make_begin(this->m_); }
  const_iterator end() const { return const_iterator::make_end(this->m_); }

  void insert(const index_type& index, const local_element_type& x) {
    m_.emplace(index, x);
  }

  size_type size() const { return m_.size(); }
  const local_element_type& at(const index_type& i) const { return m_.at(i); }

 private:
  action_map_type m_;
};

template <typename K, typename V>
std::vector<std::size_t> num_local_elements(const std::map<K, V>& m) {
  std::vector<std::size_t> num_local_elements(m.size());
  std::transform(m.begin(), m.end(), num_local_elements.begin(),
                 [](const typename std::map<K, V>::value_type& x) {
                   return x.second.size();
                 });
  return num_local_elements;
}

template <typename Index, typename LocalElement, typename JointElement>
class JointDiscreteSpace {
  friend class const_iterator;

 public:
  using index_type = Index;
  using local_element_type = LocalElement;
  using joint_element_type = JointElement;

  using agent_type = Agent<index_type>;
  using local_space_type = DiscreteSpace<index_type, local_element_type>;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using value_type = joint_element_type;
  using pointer = joint_element_type*;
  using reference = joint_element_type&;

  class const_iterator {
   public:
    using joint_discrete_set_type =
        JointDiscreteSpace<Index, LocalElement, JointElement>;
    using joint_element_type =
        typename joint_discrete_set_type::joint_element_type;

    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = joint_element_type;
    using difference_type = long long;
    using pointer = joint_element_type*;
    using const_reference = const joint_element_type&;

    const_iterator(const const_iterator& other) : s_(other.s_), e_(other.e_) {}

    const_iterator& operator++() {
      if (e_.index() < s_.size())
      	e_ = joint_element_type(Index(e_.index() + 1));
      else
      	set_to_end_();
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator ret(*this);
      ++(*this);
      return ret;
    }

    const_iterator& operator--() {
      if (e_.index() > 0)
        e_ = joint_element_type(Index(e_.index() - 1));
      else
        set_to_begin_();
      return *this;
    }

    const_iterator operator--(int) {
      const_iterator ret(*this);
      --(*this);
      return ret;
    }

    static const_iterator make_begin(const joint_discrete_set_type& s) {
      return const_iterator(s, joint_element_type(Index(0)));
    }

    static const_iterator make_end(const joint_discrete_set_type& s) {
      return const_iterator(s, joint_element_type(s.size()));
    }

    const_reference operator*() const { return e_; }

    bool operator==(const const_iterator& other) const {
      return ((&s_ == &other.s_) && (e_ == other.e_));
    }
    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

   private:
    const_iterator(const joint_discrete_set_type& s, joint_element_type e)
        : s_(s), e_(e) {}

    void set_to_end_() { e_ = joint_element_type(Index(s_.size())); }

    void set_to_begin_() { e_ = joint_element_type(Index(0)); }

    const joint_discrete_set_type& s_;
    joint_element_type e_;
  };

  JointDiscreteSpace(const std::map<agent_type, local_space_type>& local_spaces)
      : m_(local_spaces), is_(num_local_elements(local_spaces)) {}

  const local_space_type& get_local_space(const agent_type& i) const {
    return m_.at(i);
  }

  const_iterator begin() const { return const_iterator::make_begin(*this); }
  const_iterator end() const { return const_iterator::make_end(*this); }

  size_type size() const {
    if (m_.empty()) return 0;

    auto size_prod =
        [](std::size_t x,
           const typename std::map<agent_type, local_space_type>::value_type&
               y) { return x * y.second.size(); };
    return std::accumulate(std::next(m_.begin()), m_.cend(),
                           m_.begin()->second.size(), size_prod);
  }

  joint_element_type get_joint_element(
      const std::map<agent_type, local_element_type>& locals) const {
    if (locals.size() != m_.size())
      throw std::runtime_error(
          "number of local actions in input does not match number of local "
          "spaces");

    index_type joint_index(0);
    for (const auto& x : locals)
      joint_index = is_.increment_joint_index(joint_index, x.first.index(),
                                              x.second.index());

    return joint_element_type({joint_index, ""});
  }

  const local_element_type& get_local_element(const joint_element_type& joint,
                                              const agent_type& i) const {
    auto local_idx = is_.local_index(joint.index(), i.index());
    return m_.at(i).at(local_idx);
  }

  size_type num_agents() const { return m_.size(); }

 private:
  std::map<agent_type, local_space_type> m_;
  IndexSpace<index_type> is_;
};

template <typename Index = std::size_t>
using DiscreteLocalActionSpace =
    DiscreteSpace<Index, DiscreteLocalAction<Index>>;

template <typename Index = std::size_t,
          typename LocalElement = DiscreteLocalAction<Index>,
          typename JointElement = DiscreteJointAction<Index>>
using DiscreteJointActionSpace =
    JointDiscreteSpace<Index, LocalElement, JointElement>;

template <typename Index = std::size_t>
using DiscreteLocalObservationSpace =
    DiscreteSpace<Index, DiscreteLocalObservation<Index>>;

template <typename Index = std::size_t,
          typename LocalElement = DiscreteLocalObservation<Index>,
          typename JointElement = DiscreteJointObservation<Index>>
using DiscreteJointObservationSpace =
    JointDiscreteSpace<Index, LocalElement, JointElement>;

}  // namespace npgi

#endif  // DISCRETESPACES_HPP