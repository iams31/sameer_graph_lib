"""A small, friendly front door to :class:`RouteGraph`.

``RouteExplorer`` is the one-liner version of the route stack: give it a
DataFrame and it builds the schema, the tensors and the graph for you, then
answers the questions you normally ask next.

    >>> ex = RouteExplorer(df, metric="orders")
    >>> ex.sources("A1", top=5)          # where A1 orders come from
    >>> ex.drops("A1", top=5)            # where A1 orders go
    >>> ex.plot("A1", upstream=[5, 3], downstream=4)
    >>> ex.where(hour=[8, 9, 10]).plot("A1")      # same, morning peak only

Every view method honours two sticky defaults: the metric (``using``) and the
grain filter (``where``). Both return a new explorer that shares the same
graph, so nothing is rebuilt and the original is left untouched.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .route_graph import RouteGraph, RouteSchema, RouteTensor


class RouteExplorer:
    """Build, query and plot a route graph without touching the internals."""

    def __init__(
        self,
        data=None,
        *,
        graph: RouteGraph | None = None,
        schema: RouteSchema | None = None,
        pickup_col: str = "pickup_cluster",
        drop_col: str = "drop_cluster",
        grain_cols: Sequence[str] = ("week_period", "hour"),
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
            schema = schema or RouteSchema.from_frame(
                data, grain_cols=grain_cols,
                exclude_cols=(pickup_col, drop_col), **schema_kwargs)
            self.graph = RouteGraph(schema, directed=directed, allow_self_loops=allow_self_loops)
            self.graph.add_frame(data, pickup_col=pickup_col, drop_col=drop_col)
        elif schema is not None:
            self.graph = RouteGraph(schema, directed=directed, allow_self_loops=allow_self_loops)
        else:
            raise ValueError("Pass a DataFrame, a schema, or an existing RouteGraph")

        # no metric named: use the schema default, so nothing hardcodes a column
        self._metric = metric or self.graph.schema.default_metric

    # ---------------- construction ---------------- #
    @classmethod
    def from_dataframe(cls, df, **kwargs) -> "RouteExplorer":
        return cls(df, **kwargs)

    @classmethod
    def from_csv(cls, path, *, read_kwargs=None, **kwargs) -> "RouteExplorer":
        """Read a CSV and build the explorer in one step."""
        import pandas as pd

        return cls(pd.read_csv(path, **(read_kwargs or {})), **kwargs)

    @classmethod
    def from_graph(cls, graph: RouteGraph, **kwargs) -> "RouteExplorer":
        return cls(graph=graph, **kwargs)

    def add(self, data) -> "RouteExplorer":
        """Fold more rows into the same graph (schema must already cover them)."""
        self.graph.add_frame(data, pickup_col=self.pickup_col, drop_col=self.drop_col)
        return self

    # ---------------- sticky defaults ---------------- #
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
        """A view restricted to grain buckets, e.g. ``where(hour=[8, 9])``."""
        merged = dict(self._grain)
        merged.update(grain_values)
        unknown = set(merged) - set(self.schema.grain_names)
        if unknown:
            raise KeyError(f"Unknown grain(s) {sorted(unknown)}; have {self.schema.grain_names}")
        return self._view(grain=merged)

    def using(self, metric: str) -> "RouteExplorer":
        """A view whose default metric is ``metric``."""
        if metric != "count":
            self.schema.validate_metric(metric)
        return self._view(metric=metric)

    def all_grains(self) -> "RouteExplorer":
        """Drop any sticky grain filter."""
        return self._view(grain={})

    def _resolve(self, metric=None, grain_values=None):
        grain = dict(self._grain)
        grain.update(grain_values or {})
        return (metric or self._metric), grain

    # ---------------- attributes ---------------- #
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
        """Every metric name you can pass as ``metric=``."""
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

    # ---------------- tables ---------------- #
    def sources(self, cluster, top=10, metric=None, **grain_values):
        """Where this cluster's orders come from, ranked."""
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.partners_frame(cluster, direction="in", metric=metric,
                                         top=top, **grain)

    def drops(self, cluster, top=10, metric=None, **grain_values):
        """Where this cluster's orders go, ranked."""
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.partners_frame(cluster, direction="out", metric=metric,
                                         top=top, **grain)

    def top_routes(self, top=20, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.rank_routes(metric=metric, top=top, **grain)

    def clusters_frame(self, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.nodes_frame(metric=metric, **grain)

    def routes_frame(self):
        """One row per route with every metric collapsed."""
        return self.graph.edges_frame()

    def matrix(self, top=15, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.flow_matrix(metric=metric, top=top, **grain)

    def frame(self, drop_empty=True):
        """Long frame: one row per route per grain bucket (needs xarray)."""
        return self.graph.to_frame(drop_empty=drop_empty)

    def profile(self, cluster=None, pickup=None, drop=None, direction="both", metric=None):
        """One metric across two grains as a table."""
        metric, _ = self._resolve(metric, None)
        if cluster is not None:
            tensor = self.graph.node_tensor(cluster, direction=direction)
        elif pickup is not None and drop is not None:
            tensor = self.graph.graph[pickup][drop]["tensor"]
        else:
            raise ValueError("Pass cluster=... or both pickup=... and drop=...")
        return tensor.profile(metric=metric)

    # ---------------- summaries ---------------- #
    def summary(self, cluster=None, direction="both", **grain_values):
        """Every metric for one cluster, or for the whole graph."""
        _, grain = self._resolve(None, grain_values)
        if cluster is None:
            return self.graph.total_summary(**grain)
        return self.graph.node_summary(cluster, direction=direction, **grain)

    def route(self, pickup, drop, **grain_values):
        """Every metric for one route."""
        _, grain = self._resolve(None, grain_values)
        return self.graph.route_summary(pickup, drop, **grain)

    def value(self, pickup, drop, metric=None, **grain_values):
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.edge_value(pickup, drop, metric, **grain)

    def stats(self) -> dict:
        return self.graph.get_graph_stats()

    def describe(self) -> str:
        """A short human-readable overview of the graph."""
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

    # ---------------- flow ---------------- #
    def flow(self, focus, upstream=5, downstream=5, metric=None, extra_metrics=(), **kwargs):
        """The top-k expansion around ``focus`` as a DataFrame."""
        metric, grain = self._resolve(metric, None)
        return self.graph.flow_frame(focus, upstream=upstream, downstream=downstream,
                                     metric=metric, extra_metrics=extra_metrics,
                                     **grain, **kwargs)

    def subgraph(self, focus, upstream=5, downstream=5, metric=None, **kwargs):
        """The same expansion as a NetworkX DiGraph."""
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

    # ---------------- plots ---------------- #
    def plot(self, focus, upstream=5, downstream=5, metric=None, **kwargs):
        """Draw the focus clusters with their top sources and drops."""
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

    # ---------------- generic DataFrame plotting ---------------- #
    def plot_columns(self, y, x=None, group=None, **kwargs):
        """Run the stateless plotter over the long per-bucket frame.

        Handy for questions like *how does the order count move across the
        hour, weekday vs weekend*::

            ex.plot_columns(["orders", "speed"], x="hour",
                            group="week_period", agg="mean", mode="grid")
        """
        from .plotter import plot_columns

        return plot_columns(self.frame(), y=y, x=x, group=group, **kwargs)

    def plot_distribution(self, column, group=None, **kwargs):
        """Distribution of one metric across route/bucket rows."""
        from .plotter import plot_distribution

        return plot_distribution(self.frame(), column, group=group, **kwargs)

    def compare_distributions(self, column, group, **kwargs):
        """Compare one metric's distribution across groups, one colour each."""
        from .plotter import compare_distributions

        return compare_distributions(self.frame(), column, group=group, **kwargs)

    # ---------------- misc ---------------- #
    @staticmethod
    def save(fig, path, dpi: int = 180) -> str:
        """Save a returned figure and close it."""
        import matplotlib.pyplot as plt

        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)
