
from __future__ import annotations

import heapq

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "Path search requires pandas and numpy: pip install 'sameer-graph-lib[route]'"
    ) from exc


def _edge_cost(graph, u, v, metric, grain_values) -> float:
    value = graph.edge_value(u, v, metric, **grain_values)
    return float(value)


def _neighbours(graph, node, direction):
    inward = direction in ("in", "to", "sources", "upstream")
    for partner, u, v, _ in graph.incident(node, "in" if inward else "out"):
        yield partner, u, v


def search(graph, start, direction="out", cost=None, budget=None, max_hops=3,
           min_hops=1, edge_filter=None, node_filter=None, **grain_values) -> dict:
    cost = cost or graph.schema.ride_time_metric
    if cost is None:
        raise ValueError(
            "No cost metric: pass cost= (or name ride_time_metric= when building "
            "the graph) so the search knows what to accumulate along a path"
        )
    if start not in graph.graph:
        raise KeyError(f"Unknown cluster: {start!r}")
    if max_hops is not None and max_hops < 1:
        raise ValueError("max_hops must be at least 1")

    source = graph
    if edge_filter or node_filter:
        source = graph.keep(edge_filter=edge_filter, node_filter=node_filter,
                            drop_isolated=False, **grain_values)
        if start not in source.graph:
            raise ValueError(
                f"The filter removed the start cluster {start!r}; loosen it"
            )

    inward = direction in ("in", "to", "sources", "upstream")
    best_state: dict = {(start, 0): 0.0}
    found: dict = {}
    queue = [(0.0, 0, start, (start,))]

    while queue:
        spent, hops, node, path = heapq.heappop(queue)
        if spent > best_state.get((node, hops), np.inf):
            continue
        if hops >= min_hops and node != start:
            known = found.get(node)
            if known is None or spent < known["cost"]:
                ordered = tuple(reversed(path)) if inward else path
                found[node] = {"cost": spent, "hops": hops, "path": list(ordered)}
        if max_hops is not None and hops >= max_hops:
            continue

        for partner, u, v in _neighbours(source, node, direction):
            if partner in path:
                continue
            step = _edge_cost(source, u, v, cost, grain_values)
            if np.isnan(step) or step < 0:
                continue
            total = spent + step
            if budget is not None and total > budget:
                continue
            state = (partner, hops + 1)
            if total < best_state.get(state, np.inf):
                best_state[state] = total
                heapq.heappush(queue, (total, hops + 1, partner, path + (partner,)))

    return found


def reach_frame(graph, start, direction="out", cost=None, metrics=(), **kwargs):
    cost = cost or graph.schema.ride_time_metric
    if cost is None:
        raise ValueError(
            "No cost metric: pass cost= (or name ride_time_metric= when building "
            "the graph) so the search knows what to accumulate along a path"
        )
    grain = {k: v for k, v in kwargs.items() if k in graph.schema.grain_names}
    found = search(graph, start, direction=direction, cost=cost, **kwargs)

    rows = []
    for cluster, hit in found.items():
        row = {
            "cluster": cluster,
            cost: hit["cost"],
            "hops": hit["hops"],
            "path": " -> ".join(str(p) for p in hit["path"]),
        }
        for metric in metrics:
            row[metric] = path_total(graph, hit["path"], metric, **grain)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values([cost, "hops"], ignore_index=True)


def path_total(graph, path, metric, **grain_values) -> float:
    values = [graph.edge_value(u, v, metric, **grain_values)
              for u, v in zip(path, path[1:])]
    return float(np.sum(values)) if values and not np.isnan(values).any() else np.nan


def paths(graph, source, target, cost=None, budget=None, max_hops=3, min_hops=1,
          edge_filter=None, node_filter=None, metrics=(), **grain_values):
    cost = cost or graph.schema.ride_time_metric
    if cost is None:
        raise ValueError(
            "No cost metric: pass cost= (or name ride_time_metric= when building "
            "the graph) so the search knows what to accumulate along a path"
        )
    for cluster in (source, target):
        if cluster not in graph.graph:
            raise KeyError(f"Unknown cluster: {cluster!r}")
    if max_hops is not None and max_hops < 1:
        raise ValueError("max_hops must be at least 1")

    walk = graph
    if edge_filter or node_filter:
        walk = graph.keep(edge_filter=edge_filter, node_filter=node_filter,
                          drop_isolated=False, **grain_values)
        if source not in walk.graph or target not in walk.graph:
            return pd.DataFrame()

    out = []

    def step(node, path, spent):
        if node == target and len(path) - 1 >= min_hops:
            out.append((list(path), spent, len(path) - 1))
            return
        if max_hops is not None and len(path) - 1 >= max_hops:
            return
        for partner, u, v in _neighbours(walk, node, "out"):
            if partner in path:
                continue
            hop = _edge_cost(walk, u, v, cost, grain_values)
            if np.isnan(hop) or hop < 0:
                continue
            if budget is not None and spent + hop > budget:
                continue
            step(partner, path + [partner], spent + hop)

    step(source, [source], 0.0)

    rows = []
    for path, spent, hops in out:
        row = {"path": " -> ".join(str(p) for p in path), "hops": hops, cost: spent}
        for metric in metrics:
            row[metric] = path_total(graph, path, metric, **grain_values)
        row["clusters"] = path
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values([cost, "hops"], ignore_index=True)


def reach_subgraph(graph, start, direction="out", cost=None, **kwargs):
    import networkx as nx

    cost = cost or graph.schema.ride_time_metric
    if cost is None:
        raise ValueError(
            "No cost metric: pass cost= (or name ride_time_metric= when building "
            "the graph) so the search knows what to accumulate along a path"
        )
    grain = {k: v for k, v in kwargs.items() if k in graph.schema.grain_names}
    found = search(graph, start, direction=direction, cost=cost, **kwargs)
    inward = direction in ("in", "to", "sources", "upstream")
    sign = -1 if inward else 1

    tree = nx.DiGraph()
    tree.add_node(start, level=0, depth=0, side="focus", is_focus=True,
                  value=0.0, cost=0.0, hops=0)
    for cluster, hit in found.items():
        tree.add_node(cluster, level=sign * hit["hops"], depth=hit["hops"],
                      side="source" if inward else "drop", is_focus=False,
                      value=hit["cost"], cost=hit["cost"], hops=hit["hops"])
    for cluster, hit in found.items():
        path = hit["path"]
        for u, v in zip(path, path[1:]):
            if u in tree and v in tree and not tree.has_edge(u, v):
                tree.add_edge(u, v, value=_edge_cost(graph, u, v, cost, grain),
                              metric=cost, side="source" if inward else "drop",
                              depth=abs(tree.nodes[v]["level"]) or 1,
                              routes=graph.graph[u][v].get("count", 1))
    tree.graph.update({
        "focus": [start],
        "metric": cost,
        "grain": dict(grain),
        "upstream": [1] if inward else [],
        "downstream": [] if inward else [1],
        "edge_filter": dict(kwargs.get("edge_filter") or {}),
        "node_filter": dict(kwargs.get("node_filter") or {}),
        "direction": "in" if inward else "out",
        "budget": kwargs.get("budget"),
    })
    return tree
