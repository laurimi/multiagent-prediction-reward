#ifndef FSCUTILS_HPP
#define FSCUTILS_HPP
#include <boost/graph/adjacency_list.hpp>
#include "FSC.hpp"
#include "NodeConfigurations.hpp"
#include "decpomdp/discrete/DiscreteElements.hpp"
#include "decpomdp/discrete/DiscreteSpaces.hpp"
namespace npgi {

template <typename Index, typename TimeStep>
NodeConfigurationSet<DiscreteLocalAction<Index>,
                        DiscreteLocalObservation<Index>,
                        fsc_node_t>
get_configurations(
    fsc_node_t q, const DiscreteLocalActionSpace<Index>& a,
    const DiscreteLocalObservationSpace<Index>& z,
    const fsc_graph_t<DiscreteLocalAction<Index>,
                      DiscreteLocalObservation<Index>, TimeStep>& g) {
  const int next_layer = g[q].time_ + 1;
  const auto next_nodes = get_layer(next_layer, g);
  auto edge_cfgs = make_edge_configuration_set(
      std::vector<DiscreteLocalObservation<Index>>(z.begin(), z.end()),
      next_nodes);
  return make_node_configuration_set(std::vector<DiscreteLocalAction<Index>>(a.begin(), a.end()), edge_cfgs);;
}

template <typename Index, typename TimeStep>
fsc_graph_t<DiscreteLocalAction<Index>, DiscreteLocalObservation<Index>, TimeStep>
get_fsc(const std::vector<std::size_t>& layer_width, const Agent<Index>& agent,
        const JointActionSpaceFlat<Index, TimeStep>& A,
        const JointObservationSpaceFlat<Index, TimeStep>& Z) {
  fsc_graph_t<DiscreteLocalAction<Index>, DiscreteLocalObservation<Index>, TimeStep> g;

  for (std::size_t l = 0; l < layer_width.size(); ++l)
    add_layer(g, l, layer_width[l], A.local_action_space(agent, l),
              Z.local_observation_space(agent, l + 1));

  // add end node
  const std::size_t last_layer = layer_width.size();
  add_layer(g, layer_width.size(), 1, A.local_action_space(agent, last_layer),
            Z.local_observation_space(agent, last_layer + 1));
  return g;
}

template <typename Graph, typename Index>
void add_layer(Graph& g, int layer_idx, std::size_t layer_width,
               const DiscreteLocalActionSpace<Index>& a,
               const DiscreteLocalObservationSpace<Index>& z) {
  std::vector<typename Graph::vertex_descriptor> added;
  for (std::size_t i = 0; i < layer_width; ++i) {
    added.push_back(boost::add_vertex({*a.begin(), layer_idx}, g));
  }

  if (layer_idx == 0) return;

  auto prev_layer = get_layer(layer_idx - 1, g);
  if (prev_layer.empty()) throw std::runtime_error("Previous layer empty!");

  const typename Graph::vertex_descriptor dst = added.front();
  for (const auto& src : prev_layer) {
    for (auto it = z.begin(); it != z.end(); ++it) {
      boost::add_edge(src, dst, *it, g);
    }
  }
}

template <typename Graph, typename Index, typename TimeStep,
          typename RandomNumberGenerator>
void randomize(Graph& g, const Agent<Index>& agent,
               const JointActionSpaceFlat<Index, TimeStep>& A,
               const JointObservationSpaceFlat<Index, TimeStep>& Z,
               RandomNumberGenerator& gen) {
  auto vit = boost::vertices(g);
  std::vector<fsc_node_t> nodes(vit.first, vit.second);
  // sort by layer, have to randomize starting from the end.
  std::sort(nodes.begin(), nodes.end(),
            [&g](const fsc_node_t& a, const fsc_node_t& b) {
              return g[a].time_ > g[b].time_;
            });
  for (auto q : nodes) {
  	auto layer = g[q].time_;
    randomize_node_configuration(q, g, A.local_action_space(agent, layer),
              Z.local_observation_space(agent, layer + 1), gen);
  }
}

template <typename Graph, typename Index, typename RandomNumberGenerator>
void randomize_node_configuration(fsc_node_t node, Graph& g,
                                  const DiscreteLocalActionSpace<Index>& a,
                                  const DiscreteLocalObservationSpace<Index>& z,
                                  RandomNumberGenerator& gen) {
  using local_action_type = typename DiscreteLocalActionSpace<Index>::local_element_type;
  using local_observation_type = typename DiscreteLocalObservationSpace<Index>::local_element_type;

  const int next_layer = g[node].time_ + 1;
  const auto next_nodes = get_layer(next_layer, g);

  auto edge_cfgs = make_edge_configuration_set(std::vector<local_observation_type>(z.begin(), z.end()), next_nodes);
  auto node_cfgs = make_node_configuration_set(std::vector<local_action_type>(a.begin(), a.end()), edge_cfgs);
  set_unique_random_configuration(node_cfgs, node, g, gen);
}

template <typename Graph, typename NodeConfigurationSet,
          typename RandomNumberGenerator>
void set_unique_random_configuration(const NodeConfigurationSet& confs,
                                     fsc_node_t node, Graph& g,
                                     RandomNumberGenerator& gen) {
  auto this_l = get_layer(g[node].time_, g);
  this_l.erase(std::remove(this_l.begin(), this_l.end(), node),
               this_l.end());  // without node
  do {
    set_configuration(*reservoir_sample(confs.begin(), confs.end(), gen), node,
                      g);
  } while (std::any_of(
      this_l.begin(), this_l.end(),
      [&node, &g](fsc_node_t other) { return same_policy(node, other, g); }));
}

}  // namespace npgi

#endif  // FSCUTILS_HPP