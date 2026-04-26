"""Topology analysis for route affinity graphs."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import networkx as nx


class TopologyAnalyzer:
    """Split a graph into its main trunk and residual minor branches."""

    def __init__(self, graph_or_affinity) -> None:
        self.graph = graph_or_affinity if isinstance(graph_or_affinity, nx.Graph) else graph_or_affinity.graph

    def decompose_branches(self, seed_hexes: Sequence[str] | None = None) -> dict:
        if self.graph.number_of_nodes() == 0:
            return {
                "main_branch": {
                    "node_count": 0,
                    "total_count": 0,
                    "total_value": 0,
                    "hexes": [],
                },
                "minor_branches": [],
            }

        forest = self._maximum_spanning_forest()
        main_branch = self._choose_main_branch(forest, seed_hexes=seed_hexes)
        main_set = set(main_branch)

        residual = self.graph.copy()
        residual.remove_nodes_from(main_set)

        branches = []
        for component in nx.connected_components(residual):
            hexes = self._stable_component_order(component)
            branches.append(
                {
                    "node_count": len(hexes),
                    "total_count": self._sum_node_attr(hexes, "count"),
                    "total_value": self._sum_node_attr(hexes, "value"),
                    "hexes": hexes,
                    "connects_to_main_at": self._main_attachment(component, main_set),
                }
            )

        branches.sort(key=lambda item: (item["total_count"], item["node_count"]), reverse=True)
        for idx, branch in enumerate(branches, start=1):
            branch["branch_id"] = idx

        return {
            "main_branch": {
                "node_count": len(main_branch),
                "total_count": self._sum_node_attr(main_branch, "count"),
                "total_value": self._sum_node_attr(main_branch, "value"),
                "hexes": main_branch,
            },
            "minor_branches": branches,
        }

    def _maximum_spanning_forest(self) -> nx.Graph:
        weighted = nx.Graph()
        weighted.add_nodes_from(self.graph.nodes(data=True))

        for u, v, data in self.graph.edges(data=True):
            copied = dict(data)
            copied["_trunk_strength"] = self._edge_strength(u, v, data)
            weighted.add_edge(u, v, **copied)

        if weighted.number_of_edges() == 0:
            return weighted

        return nx.maximum_spanning_tree(weighted, weight="_trunk_strength")

    def _choose_main_branch(
        self,
        forest: nx.Graph,
        seed_hexes: Sequence[str] | None = None,
    ) -> List[str]:
        if forest.number_of_nodes() == 1:
            return list(forest.nodes)

        candidates = []
        seed_path = self._seed_path(forest, seed_hexes)
        if seed_path:
            candidates.append(seed_path)

        for component in nx.connected_components(forest):
            subgraph = forest.subgraph(component)
            candidates.append(self._diameter_path(subgraph))

        return max(
            candidates,
            key=lambda path: (len(path), self._sum_node_attr(path, "count")),
            default=[],
        )

    def _seed_path(self, forest: nx.Graph, seed_hexes: Sequence[str] | None) -> List[str]:
        if not seed_hexes:
            return []

        present = [cell for cell in seed_hexes if cell in forest]
        if len(present) < 2:
            return []

        best_path: List[str] = []
        # Limit pair scans for very long seed routes while keeping endpoint intent.
        candidates = present[:25] + present[-25:]
        for idx, source in enumerate(candidates):
            for target in candidates[idx + 1 :]:
                if not nx.has_path(forest, source, target):
                    continue
                path = nx.shortest_path(forest, source, target)
                if len(path) > len(best_path):
                    best_path = path

        return best_path

    def _diameter_path(self, graph: nx.Graph) -> List[str]:
        if graph.number_of_nodes() == 0:
            return []
        if graph.number_of_nodes() == 1:
            return list(graph.nodes)

        start = next(iter(graph.nodes))
        first = self._farthest_by_hops(graph, start)
        second = self._farthest_by_hops(graph, first)
        return nx.shortest_path(graph, first, second)

    def _farthest_by_hops(self, graph: nx.Graph, source: str) -> str:
        lengths = nx.single_source_shortest_path_length(graph, source)
        return max(lengths, key=lambda node: (lengths[node], self._node_metric(node)))

    def _edge_strength(self, u: str, v: str, data: dict) -> float:
        traversal_count = float(data.get("count", 0) or 0)
        endpoint_strength = (self._node_metric(u) + self._node_metric(v)) / 2.0
        distance = max(float(data.get("distance", data.get("weight", 1)) or 1), 1.0)
        return traversal_count * 1000.0 + endpoint_strength - distance * 0.001

    def _node_metric(self, node: str) -> float:
        return float(self.graph.nodes[node].get("count", 0) or 0)

    def _sum_node_attr(self, nodes: Iterable[str], attr: str) -> float:
        return sum(float(self.graph.nodes[node].get(attr, 0) or 0) for node in nodes)

    def _stable_component_order(self, component: Iterable[str]) -> List[str]:
        return sorted(
            component,
            key=lambda node: (
                -float(self.graph.nodes[node].get("count", 0) or 0),
                str(node),
            ),
        )

    def _main_attachment(self, component: Iterable[str], main_set: set[str]) -> str | None:
        options = []
        for node in component:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in main_set:
                    continue
                data = self.graph[node][neighbor]
                options.append(
                    (
                        -float(data.get("count", 0) or 0),
                        float(data.get("distance", data.get("weight", 0)) or 0),
                        neighbor,
                    )
                )

        if not options:
            return None
        options.sort()
        return options[0][2]
