"""Dominant route corridor extraction."""

from __future__ import annotations

from typing import List, Sequence

import networkx as nx


class CorridorExtractor:
    """Extract compact Dijkstra clusters that cover a target metric share."""

    def __init__(self, graph_or_affinity) -> None:
        self.graph = graph_or_affinity if isinstance(graph_or_affinity, nx.Graph) else graph_or_affinity.graph

    def extract_x_percent_corridor(
        self,
        target_pct: float = 0.8,
        weight_attr: str = "count",
        seed_hexes: Sequence[str] | None = None,
    ) -> List[str]:
        del seed_hexes

        if self.graph.number_of_nodes() == 0:
            return []
        if self.graph.number_of_nodes() <= 2:
            return list(self.graph.nodes)

        pct = self._normalize_pct(target_pct)
        total = self._total_metric(weight_attr)
        if total <= 0:
            return list(self.graph.nodes)

        target = total * pct
        best_hexes = None
        best_score = float("inf")
        best_accumulated = 0.0

        for center in self.graph.nodes:
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.graph,
                    center,
                    weight="weight",
                )
            except Exception:
                continue

            selected = []
            accumulated = 0.0
            path_sum = 0.0
            for node, distance in sorted(distances.items(), key=lambda item: item[1]):
                selected.append(node)
                accumulated += self._metric(node, weight_attr)
                path_sum += float(distance or 0)
                if accumulated >= target:
                    break

            if accumulated >= target and (
                path_sum < best_score
                or (path_sum == best_score and accumulated > best_accumulated)
            ):
                best_score = path_sum
                best_accumulated = accumulated
                best_hexes = selected[:]

        return best_hexes if best_hexes else list(self.graph.nodes)

    @staticmethod
    def _normalize_pct(target_pct: float) -> float:
        pct = float(target_pct)
        if pct > 1:
            pct = pct / 100.0
        if pct <= 0 or pct > 1:
            raise ValueError("target_pct must be in the range (0, 1] or (0, 100].")
        return pct

    def _total_metric(self, weight_attr: str) -> float:
        return sum(self._metric(node, weight_attr) for node in self.graph.nodes)

    def _metric(self, node: str, weight_attr: str) -> float:
        return float(self.graph.nodes[node].get(weight_attr, 0) or 0)
