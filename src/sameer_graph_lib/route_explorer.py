
from __future__ import annotations

from typing import Iterable, Sequence

from .route_graph import RouteGraph, RouteSchema, RouteTensor


class RouteExplorer:

    def __init__(
        self,
        data=None,
        *,
        graph: RouteGraph | None = None,
        schema: RouteSchema | None = None,
        pickup_col: str = "pickup_cluster",
        drop_col: str = "drop_cluster",
        grain_cols: Sequence[str] = (),
        metrics: Sequence[str] | None = None,
        metric: str | None = None,
        directed: bool = True,
        allow_self_loops: bool = True,
        grain: dict | None = None,
        **schema_kwargs,
    ):
        self.pickup_col = pickup_col
        self.drop_col = drop_col
        self._grain = dict(grain or {})

        if graph is not None:
            self.graph = graph
        elif data is not None:
            schema_kwargs.setdefault("exclude_cols", (pickup_col, drop_col))
            schema = schema or RouteSchema.from_frame(
                data, grain_cols=grain_cols, metrics=metrics, **schema_kwargs)
            self.graph = RouteGraph(schema, directed=directed, allow_self_loops=allow_self_loops)
            self.graph.add_frame(data, pickup_col=pickup_col, drop_col=drop_col)
        elif schema is not None:
            self.graph = RouteGraph(schema, directed=directed, allow_self_loops=allow_self_loops)
        else:
            raise ValueError("Pass a DataFrame, a schema, or an existing RouteGraph")

        self._metric = metric or self.graph.schema.default_metric

    @classmethod
    def from_dataframe(cls, df, **kwargs) -> "RouteExplorer":
        return cls(df, **kwargs)

    @classmethod
    def ask(cls, data, *, input_fn=None, output_fn=print, **kwargs) -> "RouteExplorer":
        from .columns import ask_columns

        chosen = ask_columns(data, input_fn=input_fn, output_fn=output_fn,
                             **{k: kwargs.pop(k) for k in
                                ("pickup_col", "drop_col", "grain_cols", "metrics",
                                 "mean_metrics", "sum_metrics", "length_metric",
                                 "ride_time_metric", "weight_col")
                                if k in kwargs})
        if not chosen["pickup_col"] or not chosen["drop_col"]:
            raise ValueError(
                "A route graph needs a pickup and a drop column. For data with "
                "a single cluster column, use HexMetricGraph instead."
            )
        for key in ("length_metric", "ride_time_metric", "weight_col",
                    "sum_metrics", "mean_metrics"):
            if chosen.get(key):
                kwargs.setdefault(key, chosen[key])
        return cls(data, pickup_col=chosen["pickup_col"], drop_col=chosen["drop_col"],
                   grain_cols=chosen["grain_cols"], metrics=chosen["metrics"], **kwargs)

    @staticmethod
    def preview(data, **overrides) -> str:
        from .columns import describe_columns

        return describe_columns(data, **overrides)

    @classmethod
    def from_csv(cls, path, *, read_kwargs=None, **kwargs) -> "RouteExplorer":
        import pandas as pd

        return cls(pd.read_csv(path, **(read_kwargs or {})), **kwargs)

    @classmethod
    def from_graph(cls, graph: RouteGraph, **kwargs) -> "RouteExplorer":
        return cls(graph=graph, **kwargs)

    def add(self, data) -> "RouteExplorer":
        self.graph.add_frame(data, pickup_col=self.pickup_col, drop_col=self.drop_col)
        return self

    def _view(self, **changes) -> "RouteExplorer":
        options = {
            "graph": self.graph,
            "pickup_col": self.pickup_col,
            "drop_col": self.drop_col,
            "metric": self._metric,
            "grain": dict(self._grain),
        }
        options.update(changes)
        return RouteExplorer(**options)

    def where(self, **grain_values) -> "RouteExplorer":
        merged = dict(self._grain)
        merged.update(grain_values)
        unknown = set(merged) - set(self.schema.grain_names)
        if unknown:
            raise KeyError(f"Unknown grain(s) {sorted(unknown)}; have {self.schema.grain_names}")
        return self._view(grain=merged)

    def using(self, metric: str) -> "RouteExplorer":
        if metric != "count":
            self.schema.validate_metric(metric)
        return self._view(metric=metric)

    def keep(self, edge_filter=None, node_filter=None, node_direction="both",
             drop_isolated=True, **grain_values) -> "RouteExplorer":
        _, grain = self._resolve(None, grain_values)
        filtered = self.graph.keep(edge_filter=edge_filter, node_filter=node_filter,
                                   node_direction=node_direction,
                                   drop_isolated=drop_isolated, **grain)
        return self._view(graph=filtered)

    def cut(self, edge_filter=None, node_filter=None, node_direction="both",
            drop_isolated=True, **grain_values) -> "RouteExplorer":
        _, grain = self._resolve(None, grain_values)
        self.graph.cut(edge_filter=edge_filter, node_filter=node_filter,
                       node_direction=node_direction, drop_isolated=drop_isolated,
                       **grain)
        return self

    def matching_routes(self, edge_filter=None, node_filter=None, **grain_values) -> list:
        _, grain = self._resolve(None, grain_values)
        return self.graph.matching_routes(edge_filter=edge_filter,
                                          node_filter=node_filter, **grain)

    def matching_clusters(self, node_filter, direction="both", **grain_values) -> list:
        _, grain = self._resolve(None, grain_values)
        return self.graph.matching_clusters(node_filter, direction=direction, **grain)

    def all_grains(self) -> "RouteExplorer":
        return self._view(grain={})

    def _resolve(self, metric=None, grain_values=None):
        grain = dict(self._grain)
        grain.update(grain_values or {})
        return (metric or self._metric), grain

    @property
    def schema(self) -> RouteSchema:
        return self.graph.schema

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def grain(self) -> dict:
        return dict(self._grain)

    @property
    def metrics(self) -> list:
        return self.graph.metric_names()

    @property
    def clusters(self) -> list:
        return list(self.graph.graph.nodes)

    @property
    def routes(self) -> list:
        return self.graph.routes()

    def __len__(self) -> int:
        return self.graph.graph.number_of_nodes()

    def __contains__(self, cluster) -> bool:
        return cluster in self.graph.graph

    def __repr__(self) -> str:
        stats = self.graph.get_graph_stats()
        grain = f", grain={self._grain}" if self._grain else ""
        return (f"RouteExplorer(clusters={stats['num_nodes']}, routes={stats['num_edges']}, "
                f"metric={self._metric!r}{grain})")

    def sources(self, cluster, top=10, metric=None, rank_by="edge", **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.partners_frame(cluster, direction="in", metric=metric,
                                         top=top, rank_by=rank_by, **grain)

    def drops(self, cluster, top=10, metric=None, rank_by="edge", **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.partners_frame(cluster, direction="out", metric=metric,
                                         top=top, rank_by=rank_by, **grain)

    def top_routes(self, top=20, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.rank_routes(metric=metric, top=top, **grain)

    def clusters_frame(self, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.nodes_frame(metric=metric, **grain)

    def node_split(self, cluster, *, metrics=None, diff=None, diff_order="out-in",
                   **grain_values) -> dict:
        _, grain = self._resolve(None, grain_values)
        return self.graph.node_split(cluster, metrics=metrics, diff=diff,
                                     diff_order=diff_order, **grain)

    def node_metrics_frame(self, *, metrics=None, diff=None, diff_order="out-in",
                           nodes=None, **grain_values):
        _, grain = self._resolve(None, grain_values)
        return self.graph.node_metrics_frame(metrics=metrics, diff=diff,
                                             diff_order=diff_order, nodes=nodes,
                                             **grain)

    @property
    def weight_col(self):
        return self.schema.weight_col

    def routes_frame(self, metrics=None, **grain_values):
        _, grain = self._resolve(None, grain_values)
        return self.graph.edges_frame(metrics=metrics, **grain)

    def matrix(self, top=15, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.flow_matrix(metric=metric, top=top, **grain)

    def frame(self, drop_empty=True):
        return self.graph.to_frame(drop_empty=drop_empty)

    def profile(self, cluster=None, pickup=None, drop=None, direction="both", metric=None):
        metric, _ = self._resolve(metric, None)
        if cluster is not None:
            tensor = self.graph.node_tensor(cluster, direction=direction)
        elif pickup is not None and drop is not None:
            tensor = self.graph.graph[pickup][drop]["tensor"]
        else:
            raise ValueError("Pass cluster=... or both pickup=... and drop=...")
        return tensor.profile(metric=metric)

    def summary(self, cluster=None, direction="both", **grain_values):
        _, grain = self._resolve(None, grain_values)
        if cluster is None:
            return self.graph.total_summary(**grain)
        return self.graph.node_summary(cluster, direction=direction, **grain)

    def route(self, pickup, drop, **grain_values):
        _, grain = self._resolve(None, grain_values)
        return self.graph.route_summary(pickup, drop, **grain)

    def value(self, pickup, drop, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.edge_value(pickup, drop, metric, **grain)

    def stats(self) -> dict:
        return self.graph.get_graph_stats()

    def describe(self) -> str:
        stats = self.stats()
        lines = [
            f"clusters        : {stats['num_nodes']}",
            f"routes          : {stats['num_edges']}",
            f"rows folded in  : {self.graph.route_count} group(s)",
            f"grains          : " + ", ".join(f"{k} ({v})" for k, v in stats["grains"].items()),
            f"metrics         : {len(self.metrics)} available, default {self._metric!r}",
            f"bucket coverage : {stats['avg_coverage']:.1%} of grain cells have data",
        ]
        if self._grain:
            lines.append(f"grain filter    : {self._grain}")
        return "\n".join(lines)

    def flow(self, focus, upstream=5, downstream=5, metric=None, extra_metrics=(), **kwargs):
        metric, grain = self._resolve(metric, None)
        return self.graph.flow_frame(focus, upstream=upstream, downstream=downstream,
                                     metric=metric, extra_metrics=extra_metrics,
                                     **grain, **kwargs)

    def subgraph(self, focus, upstream=5, downstream=5, metric=None, **kwargs):
        metric, grain = self._resolve(metric, None)
        return self.graph.flow_subgraph(focus, upstream=upstream, downstream=downstream,
                                        metric=metric, **grain, **kwargs)

    def tree(self, focus, upstream=5, downstream=5, metric=None, **kwargs) -> str:
        metric, grain = self._resolve(metric, None)
        return self.graph.flow_tree(focus, upstream=upstream, downstream=downstream,
                                    metric=metric, **grain, **kwargs)

    def print_tree(self, focus, upstream=5, downstream=5, metric=None, **kwargs) -> None:
        print(self.tree(focus, upstream=upstream, downstream=downstream,
                        metric=metric, **kwargs))

    def reach_to(self, cluster, budget=None, cost=None, max_hops=3, metrics=(), **kwargs):
        _, grain = self._resolve(None, None)
        return self.graph.reach_to(cluster, budget=budget, cost=cost,
                                   max_hops=max_hops, metrics=metrics, **grain, **kwargs)

    def reach_from(self, cluster, budget=None, cost=None, max_hops=3, metrics=(), **kwargs):
        _, grain = self._resolve(None, None)
        return self.graph.reach_from(cluster, budget=budget, cost=cost,
                                     max_hops=max_hops, metrics=metrics, **grain, **kwargs)

    def paths(self, source, target, budget=None, cost=None, max_hops=3,
              metrics=(), **kwargs):
        _, grain = self._resolve(None, None)
        return self.graph.paths(source, target, budget=budget, cost=cost,
                                max_hops=max_hops, metrics=metrics, **grain, **kwargs)

    def path_total(self, path, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.path_total(path, metric, **grain)

    def plot_reach(self, cluster, **kwargs):
        _, grain = self._resolve(None, None)
        return self.graph.plot_reach(cluster, **grain, **kwargs)

    def plot(self, focus, upstream=5, downstream=5, metric=None, **kwargs):
        metric, grain = self._resolve(metric, None)
        return self.graph.plot_flow(focus, upstream=upstream, downstream=downstream,
                                    metric=metric, **grain, **kwargs)

    def plot_partners(self, cluster, top=10, metric=None, **kwargs):
        metric, grain = self._resolve(metric, None)
        return self.graph.plot_partners(cluster, top=top, metric=metric, **grain, **kwargs)

    def plot_profile(self, cluster=None, pickup=None, drop=None, metric=None, **kwargs):
        metric, _ = self._resolve(metric, None)
        return self.graph.plot_profile(node=cluster, pickup=pickup, drop=drop,
                                       metric=metric, **kwargs)

    def plot_matrix(self, top=15, metric=None, **kwargs):
        metric, grain = self._resolve(metric, None)
        return self.graph.plot_matrix(metric=metric, top=top, **grain, **kwargs)

    def plot_graph(self, metric=None, **kwargs):
        metric, grain = self._resolve(metric, None)
        return self.graph.plot_graph(metric=metric, **grain, **kwargs)

    def plot_columns(self, y, x=None, group=None, **kwargs):
        from .plotter import plot_columns

        return plot_columns(self.frame(), y=y, x=x, group=group, **kwargs)

    def plot_distribution(self, column, group=None, **kwargs):
        from .plotter import plot_distribution

        return plot_distribution(self.frame(), column, group=group, **kwargs)

    def compare_distributions(self, column, group, **kwargs):
        from .plotter import compare_distributions

        return compare_distributions(self.frame(), column, group=group, **kwargs)

    @staticmethod
    def save(fig, path, dpi: int = 180) -> str:
        import matplotlib.pyplot as plt

        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)
