#ifndef GRAPHUTILS_HPP
#define GRAPHUTILS_HPP
#include <boost/graph/adjacency_list.hpp>
#include <boost/iterator/filter_iterator.hpp>

namespace npgi {
template <typename Graph>
std::pair<typename Graph::edge_descriptor, bool> find_out_edge(
    typename Graph::vertex_descriptor src,
    const typename boost::edge_bundle_type<Graph>::type& edge_property,
    const Graph& g) {
  auto eop = boost::out_edges(src, g);
  auto ef =
      std::find_if(eop.first, eop.second,
                   [&g, &edge_property](typename Graph::edge_descriptor e) {
                     return (g[e] == edge_property);
                   });
  return std::make_pair(*ef, ef != eop.second);
}

template <class Graph, class VertexPredicate>
std::pair<
    typename boost::filter_iterator<
        VertexPredicate, typename boost::graph_traits<Graph>::vertex_iterator>,
    typename boost::filter_iterator<
        VertexPredicate, typename boost::graph_traits<Graph>::vertex_iterator> >
filter_vertices(VertexPredicate pred, const Graph& g) {
  auto vit = boost::vertices(g);
  return std::make_pair(
      boost::make_filter_iterator(pred, vit.first, vit.second),
      boost::make_filter_iterator(pred, vit.second, vit.second));
}

template <typename Graph>
std::pair<typename Graph::out_edge_iterator, bool> out_edge_exists(
    typename Graph::vertex_descriptor v,
    const typename boost::edge_bundle_type<Graph>::type& edge_property,
    const Graph& g) {
  auto eop = boost::out_edges(v, g);
  auto ef =
      std::find_if(eop.first, eop.second,
                   [&g, &edge_property](typename Graph::edge_descriptor e) {
                     return g[e] == edge_property;
                   });
  return std::make_pair(ef, ef != eop.second);
}

template <typename Graph>
void redirect_in_edges(typename Graph::vertex_descriptor from,
                       typename Graph::vertex_descriptor to, Graph& g) {
  typedef std::pair<typename Graph::vertex_descriptor,
                    typename boost::edge_bundle_type<Graph>::type>
      source_edge_pair;
  std::vector<source_edge_pair> svp;

  for (auto in : boost::make_iterator_range(boost::in_edges(from, g)))
    svp.emplace_back(std::make_pair(boost::source(in, g), g[in]));

  boost::clear_in_edges(from, g);
  for (auto& se : svp) boost::add_edge(se.first, to, se.second, g);
}

}  // namespace npgi

#endif  // GRAPHUTILS_HPP