"""Core H3 affinity graph implementation."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Iterable, List, Sequence

import networkx as nx

from ._h3 import cell_area, cell_to_latlng, grid_distance, is_valid_cell
from .spatial_ingestor import SpatialIngestor


class AffinityGraph:
    """NetworkX-backed graph for H3 route affinity analysis."""

    def __init__(
        self,
        initial_hexes: dict[str, float] | Iterable[str] | None = None,
        resolution: int = 9,
        hex_resolution: int | None = None,
        ingestor: SpatialIngestor | None = None,
    ) -> None:
        self.graph = nx.Graph()
        self.node_add_count = 0
        self.total_value_sum = 0.0
        self.route_count = 0
        self.insertion_log: list[dict] = []
        self.resolution = int(hex_resolution if hex_resolution is not None else resolution)
        self.ingestor = ingestor or SpatialIngestor(resolution=self.resolution)

        if initial_hexes:
            if isinstance(initial_hexes, dict):
                for hex_id, value in initial_hexes.items():
                    self.add_hex(hex_id, value=value)
            else:
                for hex_id in initial_hexes:
                    self.add_hex(hex_id)

    def _grid_dist(self, a: str, b: str) -> int:
        return grid_distance(a, b)

    def _find_all_nearest(
        self,
        new_hex: str,
        candidates: Iterable[str] | None = None,
    ) -> tuple[list[str], float]:
        candidate_nodes = list(self.graph.nodes if candidates is None else candidates)
        distances = []
        for existing in candidate_nodes:
            if existing == new_hex:
                continue
            distances.append((existing, self._grid_dist(new_hex, existing)))

        if not distances:
            return [], float("inf")

        distances.sort(key=lambda item: item[1])
        min_dist = distances[0][1]
        return [node for node, dist in distances if dist == min_dist], min_dist

    def add_hex(self, h3_hex: str, value: float = 0.0, reroute_edges: bool = True) -> None:
        """Add one H3 cell and attach it to all nearest existing cells."""
        self._validate_hex(h3_hex)
        existing_nodes = list(self.graph.nodes)
        was_new = self._upsert_node(h3_hex, value=value)

        if not was_new:
            self.insertion_log.append(
                {
                    "action": "increment",
                    "node": h3_hex,
                    "new_count": self.graph.nodes[h3_hex]["count"],
                }
            )
            return

        if not existing_nodes:
            self.insertion_log.append({"action": "first_node", "node": h3_hex})
            return

        if len(existing_nodes) == 1:
            other = existing_nodes[0]
            distance = self._grid_dist(h3_hex, other)
            self._add_or_update_edge(h3_hex, other, distance, count_increment=0, kind="attachment")
            self.insertion_log.append(
                {
                    "action": "second_node",
                    "node": h3_hex,
                    "connected_to": [other],
                    "distance": distance,
                }
            )
            return

        nearest_nodes, min_dist = self._find_all_nearest(h3_hex, candidates=existing_nodes)
        log = {
            "action": "insert",
            "node": h3_hex,
            "nearest_nodes": nearest_nodes[:],
            "nearest_dist": min_dist,
            "edges_added": [],
            "edges_removed": [],
            "edges_rerouted": [],
        }

        for nearest in nearest_nodes:
            distance = self._grid_dist(h3_hex, nearest)
            self._add_or_update_edge(h3_hex, nearest, distance, count_increment=0, kind="attachment")
            log["edges_added"].append((h3_hex, nearest, distance))

        if reroute_edges:
            self._reroute_attachment_edges(h3_hex, nearest_nodes, log)

        self.insertion_log.append(log)

    add_node = add_hex

    def add_route(
        self,
        route_hexes: Sequence[str],
        value: float = 0.0,
        route_id: str | None = None,
    ) -> List[str]:
        """Normalize a route hex array, then insert each hex with ``add_hex``."""
        return self.add_hex_array(route_hexes, value=value, route_id=route_id)

    def add_hex_array(
        self,
        hex_array: Sequence[str],
        value: float = 0.0,
        route_id: str | None = None,
    ) -> List[str]:
        """Normalize an H3 array and add it using the original per-node procedure."""
        route = self.ingestor.ingest_h3_array(hex_array)
        if not route:
            self.insertion_log.append({"action": "empty_hex_array", "route_id": route_id})
            return []

        self.route_count += 1
        current_route_id = route_id or self.route_count
        for cell in route:
            self.add_hex(cell, value=value)

        self.insertion_log.append(
            {
                "action": "add_hex_array",
                "route_id": current_route_id,
                "route_length": len(route),
                "hexes": route[:],
            }
        )
        return route

    def add_latlng_sequence(
        self,
        coords: Sequence[tuple[float, float]],
        resolution: int | None = None,
        value: float = 0.0,
        route_id: str | None = None,
    ) -> List[str]:
        route = self.ingestor.ingest_latlng_sequence(coords, resolution=resolution)
        return self.add_route(route, value=value, route_id=route_id)

    def add_encoded_polyline(
        self,
        polyline_str: str,
        resolution: int | None = None,
        value: float = 0.0,
        route_id: str | None = None,
    ) -> List[str]:
        route = self.ingestor.ingest_encoded_polyline(polyline_str, resolution=resolution)
        return self.add_route(route, value=value, route_id=route_id)

    add_polyline = add_encoded_polyline

    def get_route_affinity_score(self, route_a: Sequence[str], route_b: Sequence[str]) -> float:
        """Return Jaccard similarity of two normalized H3 routes."""
        set_a = set(self.ingestor.ingest_h3_array(route_a))
        set_b = set(self.ingestor.ingest_h3_array(route_b))
        union = set_a | set_b
        if not union:
            return 1.0
        return round(len(set_a & set_b) / len(union), 6)

    def extract_x_percent_corridor(
        self,
        target_pct: float = 0.8,
        use_values: bool = False,
        seed_hexes: Sequence[str] | None = None,
    ) -> List[str]:
        from .corridor_extractor import CorridorExtractor

        weight_attr = "value" if use_values else "count"
        return CorridorExtractor(self).extract_x_percent_corridor(
            target_pct=target_pct,
            weight_attr=weight_attr,
            seed_hexes=seed_hexes,
        )

    def decompose_branches(self, seed_hexes: Sequence[str] | None = None) -> dict:
        from .topology_analyzer import TopologyAnalyzer

        return TopologyAnalyzer(self).decompose_branches(seed_hexes=seed_hexes)

    def get_appropriate_hexes(
        self,
        cutoff: float = 0.8,
        use_values: bool = False,
        top_k_centers: int | None = None,
    ) -> List[str]:
        """Return the most compact Dijkstra cluster covering the requested metric share.

        By default this runs Dijkstra from every node and picks the center with
        the smallest accumulated path cost. ``top_k_centers`` can be supplied
        only when you intentionally want a faster approximate scan.
        """
        if not self.graph.nodes:
            return []
        if len(self.graph.nodes) <= 2:
            return list(self.graph.nodes)

        metric_key = "value" if use_values else "count"
        total_metric = self.total_value_sum if use_values else self.node_add_count
        if total_metric == 0:
            return list(self.graph.nodes)

        cutoff = cutoff / 100.0 if cutoff > 1 else cutoff
        if cutoff <= 0 or cutoff > 1:
            raise ValueError("cutoff must be in the range (0, 1] or (0, 100].")

        target = total_metric * cutoff
        centers = list(self.graph.nodes)
        if top_k_centers is not None:
            centers = sorted(
                centers,
                key=lambda node: self.graph.nodes[node].get(metric_key, 0),
                reverse=True,
            )[:top_k_centers]

        best_hexes = None
        best_score = float("inf")
        best_accumulated = 0.0
        for center in centers:
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
            total_path_dist = 0.0
            for node, distance in sorted(distances.items(), key=lambda item: item[1]):
                selected.append(node)
                accumulated += float(self.graph.nodes[node].get(metric_key, 0) or 0)
                total_path_dist += float(distance or 0)
                if accumulated >= target:
                    break

            if accumulated >= target and (
                total_path_dist < best_score
                or (total_path_dist == best_score and accumulated > best_accumulated)
            ):
                best_score = total_path_dist
                best_accumulated = accumulated
                best_hexes = selected[:]

        return best_hexes if best_hexes else list(self.graph.nodes)

    def get_appropriate_hexes_with_stats(
        self,
        cutoff: float = 0.8,
        use_values: bool = False,
        top_k_centers: int | None = None,
    ) -> dict:
        hexes = self.get_appropriate_hexes(
            cutoff=cutoff,
            use_values=use_values,
            top_k_centers=top_k_centers,
        )
        return self._stats_for_hexes(hexes)

    def get_graph_stats(self) -> dict:
        distances = [
            float(data.get("distance", data.get("weight", 0)) or 0)
            for _, _, data in self.graph.edges(data=True)
        ]
        edge_counts = [float(data.get("count", 0) or 0) for _, _, data in self.graph.edges(data=True)]
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "total_count": self.node_add_count,
            "total_value": round(self.total_value_sum, 2),
            "route_count": self.route_count,
            "is_connected": nx.is_connected(self.graph) if self.graph.number_of_nodes() > 1 else True,
            "avg_edge_weight": round(fmean(distances), 2) if distances else 0,
            "max_edge_weight": max(distances, default=0),
            "total_edge_traversals": int(sum(edge_counts)),
        }

    def get_total_node_count_weight(self) -> int:
        return sum(int(self.graph.nodes[node].get("count", 0) or 0) for node in self.graph.nodes)

    def get_edge_distances_summary(self) -> tuple[list[float], float]:
        distances = [
            float(data.get("weight", data.get("distance", 0)) or 0)
            for _, _, data in self.graph.edges(data=True)
        ]
        return distances, sum(distances)

    def remove_hex(self, h3_hex: str) -> bool:
        """Remove a node and adjust aggregate counters."""
        if h3_hex not in self.graph:
            return False

        data = self.graph.nodes[h3_hex]
        self.node_add_count = max(0, self.node_add_count - int(data.get("count", 0) or 0))
        self.total_value_sum = max(0.0, self.total_value_sum - float(data.get("value", 0) or 0))
        self.graph.remove_node(h3_hex)
        self.insertion_log.append({"action": "remove_hex", "node": h3_hex})
        return True

    def set_hex_metric(
        self,
        h3_hex: str,
        count: int | None = None,
        value: float | None = None,
    ) -> None:
        """Edit an existing node's count and/or value while keeping totals aligned."""
        if h3_hex not in self.graph:
            raise KeyError(f"Unknown H3 cell: {h3_hex}")

        node = self.graph.nodes[h3_hex]
        if count is not None:
            new_count = int(count)
            if new_count < 0:
                raise ValueError("count cannot be negative.")
            self.node_add_count += new_count - int(node.get("count", 0) or 0)
            node["count"] = new_count

        if value is not None:
            new_value = float(value)
            self.total_value_sum += new_value - float(node.get("value", 0) or 0)
            node["value"] = new_value

        self.insertion_log.append({"action": "set_hex_metric", "node": h3_hex})

    def neighbors(self, h3_hex: str) -> list[str]:
        return list(self.graph.neighbors(h3_hex))

    def shortest_path(self, source: str, target: str) -> list[str]:
        return nx.shortest_path(self.graph, source, target, weight="distance")

    def to_dict(self) -> dict:
        return {
            "node_add_count": self.node_add_count,
            "total_value_sum": self.total_value_sum,
            "route_count": self.route_count,
            "nodes": [
                {"id": node, **dict(data)}
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **dict(data)}
                for u, v, data in self.graph.edges(data=True)
            ],
            "insertion_log": self.insertion_log,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AffinityGraph":
        graph = cls()
        graph.node_add_count = int(payload.get("node_add_count", 0) or 0)
        graph.total_value_sum = float(payload.get("total_value_sum", 0) or 0)
        graph.route_count = int(payload.get("route_count", 0) or 0)
        graph.insertion_log = list(payload.get("insertion_log", []))

        for node in payload.get("nodes", []):
            data = dict(node)
            node_id = data.pop("id")
            graph.graph.add_node(node_id, **data)

        for edge in payload.get("edges", []):
            data = dict(edge)
            source = data.pop("source")
            target = data.pop("target")
            graph.graph.add_edge(source, target, **data)

        return graph

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "AffinityGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _get_geo_pos(self) -> dict[str, tuple[float, float]]:
        return {
            node: (cell_to_latlng(node)[1], cell_to_latlng(node)[0])
            for node in self.graph.nodes
        }

    def visualize_graph(
        self,
        title: str = "H3 Hex Graph",
        highlight_hexes: Sequence[str] | str | None = None,
        figsize: tuple[float, float] = (14, 10),
        show_edge_weights: bool = True,
        show_labels: bool = True,
    ):
        """Plot the graph using H3 longitude/latitude positions for QC."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        pos = self._get_geo_pos()
        highlighted = set(highlight_hexes or [])

        node_colors = ["#2ecc71" if node in highlighted else "#e74c3c" for node in self.graph.nodes]
        node_sizes = [
            200 + int(self.graph.nodes[node].get("count", 1) or 1) * 80
            for node in self.graph.nodes
        ]
        edge_colors = [
            "#27ae60" if u in highlighted and v in highlighted else "#bdc3c7"
            for u, v in self.graph.edges()
        ]
        edge_widths = [
            2.0 if u in highlighted and v in highlighted else 0.8
            for u, v in self.graph.edges()
        ]

        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=edge_widths, ax=ax)
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax,
            edgecolors="black",
            linewidths=0.5,
        )

        if show_labels:
            labels = {
                node: f"{str(node)[-4:]}\nc:{self.graph.nodes[node].get('count', 0)}"
                for node in self.graph.nodes
            }
            nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=7, ax=ax)

        if show_edge_weights:
            edge_labels = {
                (u, v): f"{data.get('weight', data.get('distance', ''))}"
                for u, v, data in self.graph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=7,
                font_color="red",
                ax=ax,
            )

        if highlighted:
            patches = [
                mpatches.Patch(color="#2ecc71", label="Selected"),
                mpatches.Patch(color="#e74c3c", label="Not selected"),
            ]
            ax.legend(handles=patches, loc="upper left", fontsize=9)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        return fig

    def visualize_step_by_step(
        self,
        hex_list: Sequence[str],
        labels: Sequence[str] | None = None,
        figsize: tuple[float, float] = (16, 5),
        value: float = 0.0,
    ):
        """Visualize your original insertion and edge-restructuring procedure."""
        import matplotlib.pyplot as plt

        if not hex_list:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.set_title("Step-by-Step Insertion")
            ax.axis("off")
            return fig

        step_count = len(hex_list)
        fig, axes = plt.subplots(1, step_count, figsize=figsize)
        if step_count == 1:
            axes = [axes]

        temp = self.__class__(hex_resolution=self.resolution)
        display_labels = list(labels or [])
        if len(display_labels) < step_count:
            display_labels.extend(chr(65 + idx) for idx in range(len(display_labels), step_count))
        label_map: dict[str, str] = {}

        for step, hex_id in enumerate(hex_list):
            label = display_labels[step]
            temp.add_node(hex_id, value=value)
            label_map[hex_id] = label
            ax = axes[step]
            pos = temp._get_geo_pos()
            log = temp.insertion_log[-1]

            colors = [
                "#e67e22" if node == hex_id and log["action"] != "increment" else "#3498db"
                for node in temp.graph.nodes
            ]

            nx.draw_networkx_edges(temp.graph, pos, edge_color="#2c3e50", width=2, ax=ax)
            nx.draw_networkx_nodes(
                temp.graph,
                pos,
                node_color=colors,
                node_size=500,
                ax=ax,
                edgecolors="black",
                linewidths=1.5,
            )
            nx.draw_networkx_labels(
                temp.graph,
                pos,
                labels={node: label_map.get(node, str(node)[-4:]) for node in temp.graph.nodes},
                font_size=12,
                font_weight="bold",
                ax=ax,
            )
            edge_labels = {
                (u, v): f"{data.get('weight', data.get('distance', ''))}"
                for u, v, data in temp.graph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                temp.graph,
                pos,
                edge_labels=edge_labels,
                font_size=9,
                font_color="red",
                ax=ax,
            )

            title_lines = [f"Step {step + 1}: Add {label}"]
            if log["action"] == "insert":
                nearest_labels = [label_map.get(node, str(node)[-4:]) for node in log["nearest_nodes"]]
                title_lines.append(f"Near: {','.join(nearest_labels)} (d={log['nearest_dist']})")
                for u, v, weight in log.get("edges_removed", []):
                    title_lines.append(f"CUT {label_map.get(u, str(u)[-4:])}-{label_map.get(v, str(v)[-4:])}(d={weight})")
                for u, v, weight in log.get("edges_rerouted", []):
                    title_lines.append(f"ADD {label_map.get(u, str(u)[-4:])}-{label_map.get(v, str(v)[-4:])}(d={weight})")
            elif log["action"] == "increment":
                title_lines.append(f"EXISTS count={log['new_count']}")
            elif log["action"] == "second_node":
                connected = [label_map.get(node, str(node)[-4:]) for node in log["connected_to"]]
                title_lines.append(f"Connect: {','.join(connected)} (d={log['distance']})")

            ax.set_title("\n".join(title_lines), fontsize=8, fontweight="bold")
            ax.axis("off")

        fig.suptitle(
            "Step-by-Step Insertion with Edge Restructuring",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        return fig

    def plot_graph(
        self,
        title: str = "H3 Hex Graph",
        highlight_hexes: Sequence[str] | None = None,
        figsize: tuple[float, float] = (14, 10),
        show_edge_weights: bool = True,
    ):
        """Compatibility plotting method."""
        return self.visualize_graph(
            title=title,
            highlight_hexes=highlight_hexes,
            figsize=figsize,
            show_edge_weights=show_edge_weights,
        )

    def plot_h3_cells(
        self,
        cells: Sequence[str] | str | None = None,
        *,
        title: str = "H3 Cell Footprint",
        highlight_hexes: Sequence[str] | None = None,
        figsize: tuple[float, float] = (10, 8),
        show_labels: bool = True,
        label_full_hex: bool = False,
    ):
        """Plot H3 cells as real geospatial hex polygons.

        If ``cells`` is omitted, all graph nodes are plotted.
        """
        from .plotting import plot_h3_cells

        target_cells = list(self.graph.nodes) if cells is None else cells
        return plot_h3_cells(
            target_cells,
            title=title,
            selected_cells=highlight_hexes,
            figsize=figsize,
            show_labels=show_labels,
            label_full_hex=label_full_hex,
        )

    def plot_h3_cells_map(
        self,
        cells: Sequence[str] | str | None = None,
        *,
        title: str = "H3 Cells Map",
        highlight_hexes: Sequence[str] | str | None = None,
        figsize: tuple[float, float] = (16, 16),
        basemap: bool = True,
    ):
        """Plot graph H3 cells through GeoPandas with optional Contextily basemap."""
        from .plotting import plot_h3_cells_map

        target_cells = list(self.graph.nodes) if cells is None else cells
        return plot_h3_cells_map(
            target_cells,
            title=title,
            selected_cells=highlight_hexes,
            figsize=figsize,
            basemap=basemap,
        )

    def get_latlng(self, cells: Sequence[str] | str | None = None) -> list[tuple[float, float]]:
        """Return graph H3 centers as ``(lat, lng)`` tuples."""
        from .geometry import get_latlng

        target_cells = list(self.graph.nodes) if cells is None else cells
        return get_latlng(target_cells)

    def convex_hull(self, cells: Sequence[str] | str | None = None):
        """Return a Shapely convex hull around graph H3 cell centers."""
        from .geometry import h3_convex_hull

        target_cells = list(self.graph.nodes) if cells is None else cells
        return h3_convex_hull(target_cells)

    def convex_hull_geojson(self, cells: Sequence[str] | str | None = None) -> dict | None:
        """Return graph H3 center convex hull as GeoJSON-like mapping."""
        from .geometry import h3_convex_hull_geojson

        target_cells = list(self.graph.nodes) if cells is None else cells
        return h3_convex_hull_geojson(target_cells)

    def _upsert_node(self, h3_hex: str, value: float = 0.0) -> bool:
        self.node_add_count += 1
        self.total_value_sum += float(value)

        if h3_hex in self.graph:
            self.graph.nodes[h3_hex]["count"] = int(self.graph.nodes[h3_hex].get("count", 0)) + 1
            self.graph.nodes[h3_hex]["value"] = float(self.graph.nodes[h3_hex].get("value", 0)) + float(value)
            return False

        self.graph.add_node(h3_hex, count=1, value=float(value))
        return True

    def _add_or_update_edge(
        self,
        source: str,
        target: str,
        distance: float,
        count_increment: int = 1,
        kind: str = "route",
    ) -> None:
        if source == target:
            return

        if self.graph.has_edge(source, target):
            data = self.graph[source][target]
            existing_distance = float(data.get("distance", data.get("weight", distance)) or distance)
            data["distance"] = min(existing_distance, float(distance))
            data["weight"] = data["distance"]
            data["count"] = int(data.get("count", 0) or 0) + int(count_increment)
            data["kind"] = kind if data.get("kind") == kind else "mixed"
            return

        self.graph.add_edge(
            source,
            target,
            weight=float(distance),
            distance=float(distance),
            count=int(count_increment),
            kind=kind,
        )

    def _reroute_attachment_edges(self, new_hex: str, nearest_nodes: Sequence[str], log: dict) -> None:
        edges_to_remove = set()
        edges_to_add = []

        for nearest in nearest_nodes:
            for neighbor in list(self.graph.neighbors(nearest)):
                if neighbor == new_hex:
                    continue

                edge_data = self.graph[nearest][neighbor]
                if edge_data.get("kind") == "route" or int(edge_data.get("count", 0) or 0) > 0:
                    continue

                old_distance = float(edge_data.get("distance", edge_data.get("weight", 0)) or 0)
                new_distance = self._grid_dist(new_hex, neighbor)
                if new_distance < old_distance:
                    edges_to_remove.add((nearest, neighbor))
                    if not self.graph.has_edge(new_hex, neighbor):
                        edges_to_add.append((new_hex, neighbor, new_distance))

        for source, target in edges_to_remove:
            if self.graph.has_edge(source, target):
                old_distance = self.graph[source][target].get("distance", self.graph[source][target].get("weight"))
                self.graph.remove_edge(source, target)
                log["edges_removed"].append((source, target, old_distance))

        for source, target, distance in edges_to_add:
            if not self.graph.has_edge(source, target):
                self._add_or_update_edge(source, target, distance, count_increment=0, kind="attachment")
                log["edges_rerouted"].append((source, target, distance))

    def _stats_for_hexes(self, hexes: Sequence[str]) -> dict:
        if not hexes:
            return {
                "hexes": [],
                "total_count": 0,
                "total_value": 0,
                "num_hexes": 0,
                "count_coverage_pct": 0,
                "value_coverage_pct": 0,
                "area_km2": None,
            }

        selected = set(hexes)
        total_count = sum(int(self.graph.nodes[cell].get("count", 0) or 0) for cell in selected)
        total_value = sum(float(self.graph.nodes[cell].get("value", 0) or 0) for cell in selected)

        area = None
        try:
            area = round(sum(cell_area(cell, unit="km^2") for cell in selected), 4)
        except Exception:
            pass

        return {
            "hexes": list(selected),
            "total_count": total_count,
            "total_value": round(total_value, 2),
            "num_hexes": len(selected),
            "count_coverage_pct": round(total_count / self.node_add_count * 100, 2)
            if self.node_add_count
            else 0,
            "value_coverage_pct": round(total_value / self.total_value_sum * 100, 2)
            if self.total_value_sum
            else 0,
            "area_km2": area,
        }

    def _validate_hex(self, h3_hex: str) -> None:
        if not is_valid_cell(h3_hex):
            raise ValueError(f"Invalid H3 cell: {h3_hex}")
