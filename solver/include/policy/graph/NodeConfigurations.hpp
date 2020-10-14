#ifndef NODECONFIGURATIONS_HPP
#define NODECONFIGURATIONS_HPP
#include <iterator>
#include <map>
#include <ostream>
#include <vector>
#include "EdgeConfigurations.hpp"

namespace npgi {

template <typename Action, typename Observation, typename Node>
struct NodeConfiguration;

template <typename Action, typename Observation, typename Node>
std::ostream& operator<<(std::ostream&,
                         const NodeConfiguration<Action, Observation, Node>&);

template <typename Action, typename Observation, typename Node>
struct NodeConfiguration {
  Action action;
  EdgeConfiguration<Observation, Node> edge_cfg;

  bool operator==(const NodeConfiguration& other) const {
    return ((action == other.action) && (edge_cfg == other.edge_cfg));
  }

  friend std::ostream& operator<<<Action, Observation, Node>(
      std::ostream&, const NodeConfiguration<Action, Observation, Node>&);
};

template <typename Action, typename Observation, typename Node>
std::ostream& operator<<(
    std::ostream& os, const NodeConfiguration<Action, Observation, Node>& x) {
  os << x.action << " -- " << x.edge_cfg;
  return os;
}

template <typename Action, typename Observation, typename Node>
class NodeConfigurationSet {
  using edge_configuration_set_t = EdgeConfigurationSet<Observation, Node>;
  using edge_configuration_set_const_iterator =
      typename edge_configuration_set_t::const_iterator;
  using valid_configuration_set_t = std::map<Action, edge_configuration_set_t>;
  using valid_configuration_set_const_iterator =
      typename valid_configuration_set_t::const_iterator;
  friend class const_iterator;

 public:
  using node_configuration_type = NodeConfiguration<Action, Observation, Node>;
  using size_type = std::size_t;
  using value_type = node_configuration_type;

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = node_configuration_type;
    using difference_type = long long;
    using pointer = node_configuration_type*;
    using reference = node_configuration_type&;
    using const_reference = const node_configuration_type&;

    const_iterator(const const_iterator& other)
        : config(other.config),
          ait(other.ait),
          // ep(other.ep),
          first(other.first),
          second(other.second),
          eit(other.eit),
          set(other.set) {}

    friend void swap(const_iterator& first, const_iterator& second) {
      using std::swap;
      swap(first.config, second.config);
      swap(first.ait, second.ait);
      // swap(first.ep, second.ep);
      swap(first.first, second.first);
      swap(first.second, second.second);
      swap(first.eit, second.eit);
      swap(first.set, second.set);
    }

    const_iterator& operator=(const_iterator other) {
      swap(*this, other);
      return *this;
    }

    const_iterator(const_iterator&& other) noexcept : const_iterator() {
      swap(*this, other);
    }

    const_iterator& operator++() {
      if (++eit != second) {
        config.edge_cfg = *eit;
        return *this;
      }
      if (++ait == set->s.end()) {
        set_to_end_();
        return *this;
      }

      first = ait->second.begin();
      second = ait->second.end();
      eit = first;

      config.action = ait->first;
      config.edge_cfg = *eit;
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator ret(*this);
      ++(*this);
      return ret;
    }

    const_iterator& operator--() {
      if (eit != first) {
        --eit;
        config.edge_cfg = *eit;
        return *this;
      }
      if (ait != set->s.begin()) {
        set_to_begin_();
        return *this;
      }

      --ait;
      first = ait->second.begin();
      second = ait->second.end();
      eit = std::prev(second);

      config.action = ait->first;
      config.edge_cfg = *eit;
      return *this;
    }

    const_iterator operator--(int) {
      const_iterator ret(*this);
      --(*this);
      return ret;
    }

    bool operator==(const const_iterator& other) const {
      return ((set == other.set) && (ait == other.ait) && (first == other.first) && (second == other.second) &&
              (eit == other.eit) && (config == other.config));
    }

    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

    const_reference operator*() const { return config; }

    static const_iterator make_begin(const NodeConfigurationSet* set) {
      valid_configuration_set_const_iterator ait = set->s.begin();
      edge_configuration_set_const_iterator cfg_begin = ait->second.begin();
      edge_configuration_set_const_iterator cfg_end = ait->second.end();

      node_configuration_type cfg;
      cfg.action = ait->first;
      cfg.edge_cfg = *cfg_begin;

      return const_iterator(cfg, ait, cfg_begin, cfg_end, cfg_begin, set);
    }

    static const_iterator make_end(const NodeConfigurationSet* set) {
      valid_configuration_set_const_iterator al = std::prev(set->s.end());
      edge_configuration_set_const_iterator cfg_begin = al->second.begin();
      edge_configuration_set_const_iterator cfg_end = al->second.end();

      node_configuration_type cfg;
      cfg.action = al->first;
      cfg.edge_cfg = *std::prev(cfg_end);

      return const_iterator(cfg, set->s.end(), cfg_begin, cfg_end, cfg_end,
                            set);
    }

   private:
    const_iterator() {}
    const_iterator(const node_configuration_type& cfg,
                   valid_configuration_set_const_iterator ait,
                   edge_configuration_set_const_iterator begin,
                   edge_configuration_set_const_iterator end,
                   edge_configuration_set_const_iterator it,
                   const NodeConfigurationSet* s)
        : config(cfg), ait(ait), first(begin), second(end), eit(it), set(s) {}

    void set_to_end_() {
      ait = set->s.end();

      valid_configuration_set_const_iterator al = std::prev(set->s.end());
      first = al->second.begin();
      second = al->second.end();
      eit = second;

      config.action = std::prev(ait)->first;
      config.edge_cfg = *std::prev(eit);
    }

    void set_to_begin_() {
      ait = set->s.begin();

      first = ait->second.begin();
      second = ait->second.end();
      eit = first;

      config.action = ait->first;
      config.edge_cfg = *eit;
    }

    node_configuration_type config;
    valid_configuration_set_const_iterator ait;
    // std::pair<edge_configuration_set_const_iterator,
    //           edge_configuration_set_const_iterator>
    //     ep;

        edge_configuration_set_const_iterator first;
        edge_configuration_set_const_iterator second;


    edge_configuration_set_const_iterator eit;
    const NodeConfigurationSet* set;
  };

  NodeConfigurationSet() : s() {}

  NodeConfigurationSet(const std::vector<Action>& a,
                       const EdgeConfigurationSet<Observation, Node>& edge_cfg)
      : s([&a, &edge_cfg]() {
          valid_configuration_set_t m;
          for (const auto& act : a) m.emplace(act, edge_cfg);
          return m;
        }()) {}

  void set_valid_edge_configurations(
      const Action& a, const EdgeConfigurationSet<Observation, Node>& cfgs) {
    s[a] = cfgs;
  }

  const_iterator begin() const { return const_iterator::make_begin(this); }
  const_iterator end() const { return const_iterator::make_end(this); }

  size_type size() const {
    if (s.empty()) return 0;

    auto size_prod = [](
        size_type s, const typename valid_configuration_set_t::value_type& x) {
      return s + x.second.size();
    };
    return std::accumulate(s.begin(), s.end(), size_type(0), size_prod);
  }

  bool empty() const { return (size() == 0); }

 private:
  valid_configuration_set_t s;
};

template <typename Action, typename Observation, typename Node>
NodeConfigurationSet<Action, Observation, Node> make_node_configuration_set(
    const std::vector<Action>& a,
    const EdgeConfigurationSet<Observation, Node>& edge_cfg) {
  return NodeConfigurationSet<Action, Observation, Node>(a, edge_cfg);
}

}  // namespace npgi

#endif  // NODECONFIGURATIONS_HPP