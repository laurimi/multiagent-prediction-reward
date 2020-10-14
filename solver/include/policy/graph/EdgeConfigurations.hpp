#ifndef EDGECONFIGURATIONS_HPP
#define EDGECONFIGURATIONS_HPP
#include <iterator>
#include <map>
#include <ostream>
#include <vector>

namespace npgi {

template <typename Observation, typename Node>
struct EdgeConfiguration;

template <typename Observation, typename Node>
std::ostream& operator<<(std::ostream&,
                         const EdgeConfiguration<Observation, Node>&);

template <typename Observation, typename Node>
struct EdgeConfiguration {
  std::map<Observation, Node> next_nodes;


  bool operator==(const EdgeConfiguration& other) const
  {
  	return (next_nodes == other.next_nodes);
  }

  friend std::ostream& operator<< <Observation, Node>(std::ostream&,
                                  const EdgeConfiguration<Observation, Node>&);
};

template <typename Observation, typename Node>
std::ostream& operator<<(std::ostream& os,
                         const EdgeConfiguration<Observation, Node>& x) {
  for (const auto & [ o, n ] : x.next_nodes)
    os << "[" << o << " -> " << n << "]";
  return os;
}

template <typename Observation, typename Node>
class EdgeConfigurationSet {
  using node_vec_t = std::vector<Node>;
  using node_vec_const_iterator = typename node_vec_t::const_iterator;
  using valid_node_map_t = std::map<Observation, node_vec_t>;
  friend class const_iterator;

 public:
  using edge_configuration_type = EdgeConfiguration<Observation, Node>;
  using size_type = std::size_t;
  using value_type = edge_configuration_type;

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = edge_configuration_type;
    using difference_type = long long;
    using pointer = edge_configuration_type*;
    using reference = edge_configuration_type&;
    using const_reference = const edge_configuration_type&;

    using observation_config_map =
        std::map<Observation, node_vec_const_iterator>;

    const_iterator() {}

    const_iterator(const const_iterator& other)
        : config(other.config),
          observation_config(other.observation_config),
          set(other.set) {}

    friend void swap(const_iterator& first, const_iterator& second) {
      using std::swap;
      swap(first.config, second.config);
      swap(first.observation_config, second.observation_config);
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
      for (auto & [ obs, nit ] : observation_config) {
        auto nxt = std::next(nit);
        if (nxt != set->n.at(obs).end()) {
          nit = nxt;
          config.next_nodes.at(obs) = *nit;
          return *this;
        }
        nit = set->n.at(obs).begin();
        config.next_nodes.at(obs) = *nit;
      }
      set_to_end_();
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator ret(*this);
      ++(*this);
      return ret;
    }

    const_iterator& operator--() {
      for (auto & [ obs, nit ] : observation_config) {
        if (nit != set->n.at(obs).begin()) {
          nit = std::prev(nit);
          config.next_nodes.at(obs) = *nit;
          return *this;
        }
        nit = std::prev(set->n.at(obs).end());
        config.next_nodes.at(obs) = *nit;
      }
      set_to_begin_();
      return *this;
    }

    const_iterator operator--(int) {
      const_iterator ret(*this);
      --(*this);
      return ret;
    }

    bool operator==(const const_iterator& other) const {
      return ((set == other.set) &&
              (observation_config == other.observation_config) &&
              (config == other.config));
    }

    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

    const_reference operator*() const { return config; }

    static const_iterator make_begin(const EdgeConfigurationSet* s) {
      edge_configuration_type cfg;
      observation_config_map obs_cfg;
      for (const auto & [ obs, nvec ] : s->n) {
      	if (!nvec.empty())
      	{
        	cfg.next_nodes.emplace(obs, nvec.front());
        	obs_cfg.emplace(obs, nvec.begin());
      	}
      }
      return const_iterator(cfg, obs_cfg, s);
    }

    static const_iterator make_end(const EdgeConfigurationSet* s) {
      edge_configuration_type cfg;
      observation_config_map obs_cfg;
      for (const auto & [ obs, nvec ] : s->n) {
      	if (!nvec.empty())
      	{
        	cfg.next_nodes.emplace(obs, nvec.back());
        	obs_cfg.emplace(obs, nvec.end());
      	}
      }
      return const_iterator(cfg, obs_cfg, s);
    }

   private:
    const_iterator(const edge_configuration_type& cfg,
                   const observation_config_map& obs_cfg,
                   const EdgeConfigurationSet* s)
        : config(cfg), observation_config(obs_cfg), set(s) {}

    void set_to_end_() {
      for (auto & [ obs, nit ] : observation_config) {
        nit = set->n.at(obs).end();
        config.next_nodes.at(obs) = *std::prev(nit);
      }
    }

    void set_to_begin_() {
      for (auto & [ obs, nit ] : observation_config) {
        nit = set->n.at(obs).begin();
        config.next_nodes.at(obs) = *nit;
      }
    }

    edge_configuration_type config;
    std::map<Observation, node_vec_const_iterator> observation_config;
    const EdgeConfigurationSet* set;
  };

  EdgeConfigurationSet() : n() {}

  EdgeConfigurationSet(const std::vector<Observation>& o,
                       const std::vector<Node>& next_nodes)
      : n([&o, &next_nodes]() {
          valid_node_map_t m;
          for (const auto& obs : o) m.emplace(obs, next_nodes);
          return m;
        }()) {}

  void set_valid_next_nodes(const Observation& o,
                            const std::vector<Node>& next_nodes) {
    n[o] = next_nodes;
  }

  const_iterator begin() const { return const_iterator::make_begin(this); }
  const_iterator end() const { return const_iterator::make_end(this); }

  size_type size() const {
    if (n.empty()) return 0;
    auto size_prod = [](size_type s,
                        const typename valid_node_map_t::value_type& x) {
      return s * x.second.size();
    };
    return std::accumulate(n.begin(), n.end(), size_type(1), size_prod);
  }

  bool empty() const { return (size() == 0); }

 private:
  valid_node_map_t n;
};

template <typename Observation, typename Node>
EdgeConfigurationSet<Observation, Node> make_edge_configuration_set(
    const std::vector<Observation>& o, const std::vector<Node>& next_nodes) {
  return EdgeConfigurationSet<Observation, Node>(o, next_nodes);
}

}  // namespace npgi
#endif  // EDGECONFIGURATIONS_HPP