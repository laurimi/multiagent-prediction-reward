#ifndef FSC_HPP
#define FSC_HPP
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <vector>
#include "utilities/GraphUtils.hpp"
#include "NodeConfigurations.hpp"
#include "utilities/SamplingUtilities.hpp"

namespace npgi {
template <typename Action, typename TimeStep>
struct FSCNode {
  Action action_;
  TimeStep time_;
  bool operator==(const FSCNode<Action, TimeStep>& other) const
  {
    return ((action_ == other.action_) && (time_ == other.time_));
  }
};

template <typename Action, typename Observation, typename TimeStep = int>
using fsc_graph_t =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                          FSCNode<Action, TimeStep>, Observation>;

using fsc_node_t =
    boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                 boost::bidirectionalS>::vertex_descriptor;

template <typename Graph>
std::vector<fsc_node_t> get_layer(int l, const Graph& g) {
  auto vp =
      filter_vertices([l, &g](fsc_node_t v) { return (g[v].time_ == l); }, g);

  if (vp.first != vp.second)
    return std::vector<fsc_node_t>(vp.first, vp.second);
  else
    return std::vector<fsc_node_t>();
}

template <typename Action, typename Observation, typename TimeStep>
void set_configuration(
    const NodeConfiguration<Action, Observation, fsc_node_t>& config,
    const fsc_node_t node, fsc_graph_t<Action, Observation, TimeStep>& g) {
  g[node].action_ = config.action;
  boost::clear_out_edges(node, g);
  for (const auto& [obs, next_node] : config.edge_cfg.next_nodes)
    boost::add_edge(node, next_node, obs, g);
}

template <typename Action, typename Observation, typename TimeStep>
bool same_policy(typename fsc_graph_t<Action, Observation, TimeStep>::vertex_descriptor v,
                 typename fsc_graph_t<Action, Observation, TimeStep>::vertex_descriptor u,
                 const fsc_graph_t<Action, Observation, TimeStep>& g) {
  if (v == u) return true;
  if (!(g[v] == g[u])) return false;
  // same layer and same action, recursion: check local policies at next layer
  // nodes
  for (auto vep = boost::out_edges(v, g); vep.first != vep.second;
       ++vep.first) {
    for (auto uep = boost::out_edges(u, g); uep.first != uep.second;
         ++uep.first) {
      // check if Observations same
      if (g[*vep.first] != g[*uep.first]) continue;
      // check if next nodes' policies  different
      if (!same_policy(boost::target(*vep.first, g),
                       boost::target(*uep.first, g), g))
        return false;
    }
  }
  return true;
}

template <typename Action, typename Observation, typename TimeStep>
std::vector<fsc_node_t> get_identical_policy_nodes(
    fsc_node_t q, const std::vector<fsc_node_t>& q_others,
    const fsc_graph_t<Action, Observation, TimeStep>& fsc) {
  std::vector<fsc_node_t> q_identical;
  for (auto q_other : q_others) {
    if (same_policy(q, q_other, fsc)) {
      q_identical.push_back(q_other);
    }
  }
  return q_identical;
}

template <typename Action, typename Observation, typename TimeStep>
void redirect(fsc_node_t from, fsc_node_t to,
                                 fsc_graph_t<Action, Observation, TimeStep>& fsc) {
  redirect_in_edges(from, to, fsc);
}

template <typename Action, typename Observation, typename TimeStep>
class VertexWriter {
 public:
  using node_t = typename fsc_graph_t<Action, Observation, TimeStep>::vertex_descriptor;
  VertexWriter(const fsc_graph_t<Action, Observation, TimeStep>* g) : g_(g) {}
  void operator()(std::ostream& out, const node_t& v) const {
    out << "[label=\"" << (*g_)[v].action_ << "\", layer=\"" << (*g_)[v].time_
        << "\"]";
  }

 private:
  const fsc_graph_t<Action, Observation, TimeStep>* g_;
};

template <typename Action, typename Observation, typename TimeStep>
class EdgeWriter {
 public:
  using edge_t = typename fsc_graph_t<Action, Observation, TimeStep>::edge_descriptor;
  EdgeWriter(const fsc_graph_t<Action, Observation, TimeStep>* g) : g_(g) {}
  void operator()(std::ostream& out, const edge_t& e) const {
    out << "[label=\"" << (*g_)[e] << "\"]";
  }

 private:
  const fsc_graph_t<Action, Observation, TimeStep>* g_;
};

template <typename Action, typename Observation, typename TimeStep>
std::ostream& operator<<(std::ostream& os,
                         const fsc_graph_t<Action, Observation, TimeStep>& g) {
  VertexWriter<Action, Observation, TimeStep> vw(&g);
  EdgeWriter<Action, Observation, TimeStep> ew(&g);
  boost::write_graphviz(os, g, vw, ew);
  return os;
}

template <typename Action, typename Observation, typename TimeStep>
std::istream& operator>>(std::istream& is, fsc_graph_t<Action, Observation, TimeStep>& g) {
  using Node =
      typename boost::vertex_bundle_type<fsc_graph_t<Action, Observation, TimeStep>>::type;

  boost::dynamic_properties dp(boost::ignore_other_properties);
  dp.property("label", get(&Node::action_, g));
  dp.property("layer", get(&Node::time_, g));
  dp.property("label", get(boost::edge_bundle, g));
  if (!boost::read_graphviz(is, g, dp)) is.setstate(std::ios::failbit);
  return is;
}

}  // namespace npgi

#endif  // FSC_HPP