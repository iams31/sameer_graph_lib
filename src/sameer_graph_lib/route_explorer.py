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

        # no metric named: use the schema default, so nothing hardcodes a column
        self._metric = metric or self.graph.schema.default_metric

    # ---------------- construction ---------------- #
    @classmethod
    def from_dataframe(cls, df, **kwargs) -> "RouteExplorer":
        return cls(df, **kwargs)

    @classmethod
    def ask(cls, data, *, input_fn=None, **kwargs) -> "RouteExplorer":
        """Build after confirming which columns are which.

        Column names differ between exports, so rather than assuming
        ``pickup_cluster`` / ``week_period``, this proposes a mapping read from
        the frame and lets you correct it::

            ex = RouteExplorer.ask(df)

        Answers may be column names or the numbers in the listing; blank
        accepts the proposal. Outside an interactive session the proposal is
        used as-is, so scripts and notebooks with saved output still work.
        """
        from .columns import ask_columns

        chosen = ask_columns(data, input_fn=input_fn,
                             **{k: kwargs.pop(k) for k in
                                ("pickup_col", "drop_col", "grain_cols", "metrics",
                                 "length_metric", "ride_time_metric", "weight_col")
                                if k in kwargs})
        if not chosen["pickup_col"] or not chosen["drop_col"]:
            raise ValueError(
                "A route graph needs a pickup and a drop column. For data with "
                "a single cluster column, use HexMetricGraph instead."
            )
        for key in ("length_metric", "ride_time_metric", "weight_col"):
            if chosen.get(key):
                kwargs.setdefault(key, chosen[key])
        return cls(data, pickup_col=chosen["pickup_col"], drop_col=chosen["drop_col"],
                   grain_cols=chosen["grain_cols"], metrics=chosen["metrics"], **kwargs)

    @staticmethod
    def preview(data, **overrides) -> str:
        """What :meth:`ask` would propose, as a table, without prompting."""
        from .columns import describe_columns

        return describe_columns(data, **overrides)

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

    def keep(self, edge_filter=None, node_filter=None, node_direction="both",
             drop_isolated=True, **grain_values) -> "RouteExplorer":
        """A NEW explorer over only the routes and clusters a filter accepts.

        A test is a number (``>=``), a ``(operator, value)`` pair or a
        ``(low, high)`` range; several metrics are combined with AND::

            fast = ex.keep(edge_filter={"speed": (">", 0.25)})
            busy = ex.keep(node_filter={"orders": 5_000})
            both = ex.keep(edge_filter={"orders": 500, "speed": (0.2, 0.6)})

        The original explorer and its graph are untouched, and the sticky
        grain filter applies, so ``ex.where(hour=[8, 9]).keep(...)`` tests the
        morning numbers only.
        """
        _, grain = self._resolve(None, grain_values)
        filtered = self.graph.keep(edge_filter=edge_filter, node_filter=node_filter,
                                   node_direction=node_direction,
                                   drop_isolated=drop_isolated, **grain)
        return self._view(graph=filtered)

    def cut(self, edge_filter=None, node_filter=None, node_direction="both",
            drop_isolated=True, **grain_values) -> "RouteExplorer":
        """Remove what the filter MATCHES from this graph, in place.

        The mirror of :meth:`keep`, for pruning a large graph rather than
        copying it::

            ex.cut(edge_filter={"orders": ("<", 100)})     # drop thin routes
            ex.cut(node_filter={"row_count": ("<", 5)})    # drop sparse clusters
        """
        _, grain = self._resolve(None, grain_values)
        self.graph.cut(edge_filter=edge_filter, node_filter=node_filter,
                       node_direction=node_direction, drop_isolated=drop_isolated,
                       **grain)
        return self

    def matching_routes(self, edge_filter=None, node_filter=None, **grain_values) -> list:
        """Which routes a filter would accept, without changing anything."""
        _, grain = self._resolve(None, grain_values)
        return self.graph.matching_routes(edge_filter=edge_filter,
                                          node_filter=node_filter, **grain)

    def matching_clusters(self, node_filter, direction="both", **grain_values) -> list:
        """Which clusters a node filter would accept."""
        _, grain = self._resolve(None, grain_values)
        return self.graph.matching_clusters(node_filter, direction=direction, **grain)

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
    def sources(self, cluster, top=10, metric=None, rank_by="edge", **grain_values):
        """Where this cluster's orders come from, ranked.

        ``rank_by="node"`` ranks by each source cluster's own total instead of
        by the route into this one.
        """
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.partners_frame(cluster, direction="in", metric=metric,
                                         top=top, rank_by=rank_by, **grain)

    def drops(self, cluster, top=10, metric=None, rank_by="edge", **grain_values):
        """Where this cluster's orders go, ranked.

        ``rank_by="node"`` ranks by each drop cluster's own total instead of by
        the route out of this one.
        """
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
        """Inbound and outbound for one cluster, metric by metric.

        Only the cluster is needed; the rest are optional::

            ex.node_split("A1")
            ex.node_split("A1", metrics=["orders", "avg_distance"], diff=True)
        """
        _, grain = self._resolve(None, grain_values)
        return self.graph.node_split(cluster, metrics=metrics, diff=diff,
                                     diff_order=diff_order, **grain)

    def node_metrics_frame(self, *, metrics=None, diff=None, diff_order="out-in",
                           nodes=None, **grain_values):
        """One row per cluster, with in/out (and optional diff) per metric."""
        _, grain = self._resolve(None, grain_values)
        return self.graph.node_metrics_frame(metrics=metrics, diff=diff,
                                             diff_order=diff_order, nodes=nodes,
                                             **grain)

    @property
    def weight_col(self):
        """The column the mean metrics are weighted by."""
        return self.schema.weight_col

    def routes_frame(self, metrics=None, **grain_values):
        """One row per route. ``metrics`` picks the columns to include."""
        _, grain = self._resolve(None, grain_values)
        return self.graph.edges_frame(metrics=metrics, **grain)

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
        """The top-k expansion around ``focus`` as a DataFrame.

        Accepts ``edge_filter`` / ``node_filter`` like :meth:`plot`.
        """
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

    # ---------------- reachability ---------------- #
    def reach_to(self, cluster, budget=None, cost=None, max_hops=3, metrics=(), **kwargs):
        """Clusters that can REACH this one, with the cheapest route to it.

        The two-hour question::

            ex.reach_to("A1", budget=120)                    # minutes, if that is the unit
            ex.reach_to("A1", budget=120, max_hops=2)        # and at most two legs
            ex.reach_to("A1", budget=120, edge_filter={"orders": 100})

        ``cost`` defaults to the schema ride-time metric and is summed along
        the path. Cheapest first.
        """
        _, grain = self._resolve(None, None)
        return self.graph.reach_to(cluster, budget=budget, cost=cost,
                                   max_hops=max_hops, metrics=metrics, **grain, **kwargs)

    def reach_from(self, cluster, budget=None, cost=None, max_hops=3, metrics=(), **kwargs):
        """Clusters this one can REACH, under the same constraints."""
        _, grain = self._resolve(None, None)
        return self.graph.reach_from(cluster, budget=budget, cost=cost,
                                     max_hops=max_hops, metrics=metrics, **grain, **kwargs)

    def paths(self, source, target, budget=None, cost=None, max_hops=3,
              metrics=(), **kwargs):
        """Every qualifying route between two clusters, not just the best one."""
        _, grain = self._resolve(None, None)
        return self.graph.paths(source, target, budget=budget, cost=cost,
                                max_hops=max_hops, metrics=metrics, **grain, **kwargs)

    def path_total(self, path, metric=None, **grain_values):
        """Sum one metric along a path, e.g. its total ride time."""
        metric, grain = self._resolve(metric, grain_values)
        return self.graph.path_total(path, metric, **grain)

    def plot_reach(self, cluster, **kwargs):
        """Draw everything within reach, laid out by hop count."""
        _, grain = self._resolve(None, None)
        return self.graph.plot_reach(cluster, **grain, **kwargs)

    # ---------------- plots ---------------- #
    def plot(self, focus, upstream=5, downstream=5, metric=None, **kwargs):
        """Draw the focus clusters with their top sources and drops.

        ``edge_filter`` / ``node_filter`` restrict which routes the expansion
        may walk, so the top-k is picked from what survives::

            ex.plot("A1", upstream=5, edge_filter={"speed": (">", 0.25)})
        """
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
