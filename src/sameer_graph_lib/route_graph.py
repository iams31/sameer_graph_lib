"""Grain-aware route graph.

A :class:`RouteGraph` is a directed ``pickup_cluster -> drop_cluster`` graph in
which every edge carries a :class:`RouteTensor`: a dense metric cube indexed by
one or more *grain* dimensions (``week_period`` x ``hour`` by default).

Sum metrics are accumulated, mean metrics are accumulated as weighted sums so
that merging two tensors stays exact, and ``speed`` is derived as a ratio of
means (total distance / total time).

The tensor layer needs pandas and numpy; the ``xarray`` views (``raw``,
``data``, ``as_dataset``) are imported lazily, so the rest of the module works
without it::

    pip install 'sameer-graph-lib[route]'
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import networkx as nx

try:  # pandas/numpy are hard requirements for this module only
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Route graph features require pandas and numpy: "
        "pip install 'sameer-graph-lib[route]'"
    ) from exc


#: Columns added together across rows. These are only the fallback names used
#: when a frame is not self-describing - :meth:`RouteSchema.from_frame` reads
#: whatever numeric columns your frame actually has.
SUM_METRICS = [
    "requests", "orders", "accepted_requests", "accepted_orders",
    "completed_orders", "cancelled_orders", "unfulfilled", "supply_count",
]

#: Columns averaged across rows, weighted by :attr:`RouteSchema.weight_col`.
MEAN_METRICS = [
    "avg_distance", "avg_duration", "avg_price", "avg_wait_time",
    "avg_rating", "accept_rate", "fulfil_rate", "surge_pct",
]

#: Categorical columns kept as per-edge counters.
CATEGORICAL = ["segment"]

#: Columns that identify the route rather than measure it.
ID_COLS = ("pickup_cluster", "drop_cluster", "city")

#: Name fragments that mark a column as an average rather than a total.
MEAN_HINTS = ("avg", "mean", "median", "rate", "ratio", "pct", "percent",
              "share", "score", "index", "per_")

ROWS = "__rows"

#: Metric aliases that resolve against the schema instead of a column name.
DERIVED_METRICS = ("length", "ride_time", "speed", "row_count", "count")


def classify_metrics(df, exclude=()) -> tuple[list, list]:
    """Split a frame's numeric columns into sum-like and mean-like metrics.

    A column is treated as an average when its name says so (``avg_``,
    ``_rate``, ``pct``, ...) or when it is one of :data:`MEAN_METRICS`;
    everything else numeric is summed.
    """
    skip = set(exclude)
    sums, means = [], []
    for column in df.columns:
        if column in skip or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        name = str(column).lower()
        if (column in MEAN_METRICS or name.endswith("%")
                or any(hint in name for hint in MEAN_HINTS)):
            means.append(column)
        else:
            sums.append(column)
    return sums, means


def _require_xarray():
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Dataset views require xarray: pip install 'sameer-graph-lib[route]'"
        ) from exc
    return xr


def _as_list(value) -> list:
    """Wrap a scalar grain value in a list, pass sequences through."""
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, set, frozenset, range, np.ndarray, pd.Index, pd.Series)
    ):
        return [value]
    return list(value)


@dataclass
class RouteSchema:
    """Grain dimensions plus the metric columns a tensor should carry."""

    grains: dict
    sum_metrics: list = field(default_factory=lambda: list(SUM_METRICS))
    mean_metrics: list = field(default_factory=lambda: list(MEAN_METRICS))
    weight_col: str | None = "requests"
    length_metric: str = "avg_distance"
    ride_time_metric: str = "avg_duration"
    speed_scale: float = 1.0

    def __post_init__(self):
        if not self.grains:
            raise ValueError("Schema needs at least one grain dimension")
        self.grains = {k: list(v) for k, v in self.grains.items()}
        self._indexers = {k: pd.Index(v) for k, v in self.grains.items()}
        for m in (self.length_metric, self.ride_time_metric):
            if m not in self.mean_metrics:
                raise ValueError(f"{m!r} must be one of mean_metrics")

    @classmethod
    def from_frame(cls, df, grain_cols=("week_period", "hour"), exclude_cols=(), **kwargs):
        """Infer the grains and the metric columns from a DataFrame.

        Every numeric column that is not a grain, an id or excluded becomes a
        metric, classified by :func:`classify_metrics`. Pass ``sum_metrics`` /
        ``mean_metrics`` explicitly to override the split.
        """
        grains = {}
        for col in grain_cols:
            if col not in df:
                raise KeyError(f"Grain column {col!r} is not in the frame")
            if col == "hour":
                hours = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
                grains[col] = sorted(set(range(24)) | set(hours))
            else:
                grains[col] = sorted(df[col].dropna().unique().tolist())

        skip = set(grain_cols) | set(exclude_cols) | set(CATEGORICAL) | set(ID_COLS)
        inferred_sums, inferred_means = classify_metrics(df, exclude=skip)
        kwargs.setdefault("sum_metrics", inferred_sums)
        mean = list(kwargs.get("mean_metrics", inferred_means))
        # the length and ride-time metrics stay in the schema even when the
        # frame lacks them, so speed is NaN instead of the schema refusing to build
        for key, fallback in (("length_metric", "avg_distance"),
                              ("ride_time_metric", "avg_duration")):
            metric = kwargs.get(key, fallback)
            if metric not in mean:
                mean.append(metric)
        kwargs["mean_metrics"] = mean
        return cls(grains=grains, **kwargs)

    @property
    def grain_names(self):
        return list(self.grains)

    @property
    def shape(self):
        return tuple(len(v) for v in self.grains.values())

    @property
    def default_metric(self) -> str:
        """Metric used when a call does not name one: the first sum metric."""
        if self.sum_metrics:
            return self.sum_metrics[0]
        if self.mean_metrics:
            return self.mean_metrics[0]
        return "row_count"

    @property
    def metric_names(self) -> list[str]:
        """Every metric ``summary()`` can return, including derived ones."""
        return ["row_count", *self.sum_metrics, *self.mean_metrics, "speed"]

    def empty_arrays(self):
        names = [ROWS, *self.sum_metrics]
        for m in self.mean_metrics:
            names += [f"{m}__wsum", f"{m}__w"]
        return {n: np.zeros(self.shape, dtype=float) for n in names}

    def index_of(self, dim, values):
        if dim == "hour":
            values = pd.to_numeric(values, errors="coerce")
        indexer = self._indexers[dim]
        values = pd.Index(values)
        if indexer.dtype.kind in "iu" and values.dtype.kind == "f":
            # 9.0 must match the integer hour 9; NaN falls back to a miss
            numeric = values.to_numpy(dtype=float)
            safe = np.where(np.isfinite(numeric), numeric, -1.0)
            values = pd.Index(np.rint(safe).astype("int64"))
        return indexer.get_indexer(values)

    def row_weights(self, df):
        if not self.weight_col or self.weight_col not in df:
            return np.ones(len(df))
        w = pd.to_numeric(df[self.weight_col], errors="coerce").fillna(0).clip(lower=0)
        return w.where(w > 0, 1.0).to_numpy(float)   # zero-volume rows still count once

    def compatible(self, other):
        return (self.grains == other.grains
                and self.sum_metrics == other.sum_metrics
                and self.mean_metrics == other.mean_metrics)

    def resolve_metric(self, metric: str) -> str:
        """Map ``length``/``ride_time`` aliases onto the configured columns."""
        if metric == "length":
            return self.length_metric
        if metric == "ride_time":
            return self.ride_time_metric
        return metric

    def validate_metric(self, metric: str) -> str:
        resolved = self.resolve_metric(metric)
        if resolved in self.metric_names or resolved in DERIVED_METRICS:
            return resolved
        raise KeyError(
            f"Unknown metric {metric!r}. Available: {', '.join(self.metric_names)}"
        )


class RouteTensor:
    """Dense metric cube for one route, indexed by the schema grains."""

    def __init__(self, schema: RouteSchema, arrays=None):
        self.schema = schema
        self.arrays = arrays if arrays is not None else schema.empty_arrays()

    @classmethod
    def from_frame(cls, schema, df):
        return cls(schema).add_rows(df)

    def add_rows(self, df):
        s = self.schema
        pos = [s.index_of(g, df[g]) for g in s.grain_names]
        valid = np.logical_and.reduce([p >= 0 for p in pos])
        if not valid.all():
            warnings.warn(f"{int((~valid).sum())} row(s) dropped: grain value not in schema")
        df, pos = df.loc[valid], [p[valid] for p in pos]
        w = s.row_weights(df)

        np.add.at(self.arrays[ROWS], tuple(pos), 1.0)
        for m in s.sum_metrics:
            if m not in df:
                continue
            v = pd.to_numeric(df[m], errors="coerce").to_numpy(float)
            ok = ~np.isnan(v)
            np.add.at(self.arrays[m], tuple(p[ok] for p in pos), v[ok])
        for m in s.mean_metrics:
            if m not in df:
                continue
            v = pd.to_numeric(df[m], errors="coerce").to_numpy(float)
            ok = ~np.isnan(v)
            at = tuple(p[ok] for p in pos)
            np.add.at(self.arrays[f"{m}__wsum"], at, v[ok] * w[ok])
            np.add.at(self.arrays[f"{m}__w"], at, w[ok])
        return self

    def merge(self, other: "RouteTensor"):
        if not self.schema.compatible(other.schema):
            raise ValueError("Cannot merge tensors with different schemas")
        for k, arr in other.arrays.items():
            self.arrays[k] += arr
        return self

    def copy(self):
        return RouteTensor(self.schema, {k: v.copy() for k, v in self.arrays.items()})

    def __add__(self, other: "RouteTensor") -> "RouteTensor":
        return self.copy().merge(other)

    @property
    def coverage(self):
        """Fraction of grain buckets that have at least one row."""
        return float((self.arrays[ROWS] > 0).mean())

    @property
    def row_total(self) -> float:
        """How many source rows went into this tensor."""
        return float(self.arrays[ROWS].sum())

    @property
    def is_empty(self) -> bool:
        return self.row_total == 0

    @property
    def raw(self):
        """xarray view of the accumulators (rows, sums, weighted sums)."""
        xr = _require_xarray()
        dims = self.schema.grain_names
        return xr.Dataset({k: (dims, v) for k, v in self.arrays.items()},
                          coords=self.schema.grains)

    @property
    def data(self):
        """Resolved metrics per bucket (NaN where a bucket has no data)."""
        xr = _require_xarray()
        s, dims = self.schema, self.schema.grain_names
        rows = self.arrays[ROWS]
        out = {"row_count": rows}
        for m in s.sum_metrics:
            out[m] = np.where(rows > 0, self.arrays[m], np.nan)
        for m in s.mean_metrics:
            w = self.arrays[f"{m}__w"]
            out[m] = np.divide(self.arrays[f"{m}__wsum"], w,
                               out=np.full_like(w, np.nan), where=w > 0)
        out["speed"] = self._speed(out[s.length_metric], out[s.ride_time_metric])
        return xr.Dataset({k: (dims, v) for k, v in out.items()}, coords=s.grains)

    def get(self, **grain_values):
        """Metrics for a specific bucket, e.g. get(hour=9, week_period='weekday')."""
        return self.data.sel(**grain_values)

    def selector(self, grain_values):
        """``np.ix_`` selector for a grain filter, or ``None`` for everything."""
        s = self.schema
        if not grain_values:
            return None
        unknown = set(grain_values) - set(s.grain_names)
        if unknown:
            raise KeyError(
                f"Unknown grain(s) {sorted(unknown)}. Available: {s.grain_names}"
            )
        picks = []
        for dim in s.grain_names:
            if dim in grain_values:
                wanted = _as_list(grain_values[dim])
                found = s.index_of(dim, pd.Index(wanted))
                missing = [v for v, p in zip(wanted, found) if p < 0]
                if missing:
                    raise KeyError(f"{dim}={missing} not present in the schema")
                picks.append(np.asarray(found))
            else:
                picks.append(np.arange(len(s.grains[dim])))
        return np.ix_(*picks)

    def totals(self, **grain_values) -> dict:
        """Summed accumulators over all buckets, or a grain selection."""
        sel = self.selector(grain_values)
        if sel is None:
            return {k: float(v.sum()) for k, v in self.arrays.items()}
        return {k: float(v[sel].sum()) for k, v in self.arrays.items()}

    def summary(self, **grain_values):
        """Collapse buckets (all, or a selection like hour=[8, 9, 10]) into one dict."""
        s = self.schema
        tot = self.totals(**grain_values)
        has = tot[ROWS] > 0
        out = {"row_count": tot[ROWS]}
        for m in s.sum_metrics:
            out[m] = tot[m] if has else np.nan
        for m in s.mean_metrics:
            w = tot[f"{m}__w"]
            out[m] = tot[f"{m}__wsum"] / w if w > 0 else np.nan
        out["speed"] = float(self._speed(out[s.length_metric], out[s.ride_time_metric]))
        return out

    def frame(self, drop_empty: bool = True):
        """Long DataFrame with one row per grain bucket."""
        df = self.data.to_dataframe().reset_index()
        return df[df["row_count"] > 0].reset_index(drop=True) if drop_empty else df

    def profile(self, metric: str = "speed", index=None, columns=None):
        """Pivot one metric across two grains, e.g. week_period x hour."""
        s = self.schema
        metric = s.validate_metric(metric)
        names = s.grain_names
        index = index or names[0]
        table = self.data[metric]
        if len(names) == 1:
            return table.to_pandas()
        columns = columns or next(n for n in names if n != index)
        return table.transpose(index, columns).to_pandas()

    def _speed(self, lm, ride_time):
        """Ratio of means: avg LM / avg ride time (i.e. total distance / total time)."""
        lm, rt = np.asarray(lm, dtype=float), np.asarray(ride_time, dtype=float)
        spd = np.divide(lm, rt, out=np.full_like(rt, np.nan), where=(rt > 0) & ~np.isnan(lm))
        return self.schema.speed_scale * spd


class RouteGraph:
    """Directed pickup -> drop graph whose edges carry :class:`RouteTensor`."""

    def __init__(self, schema: RouteSchema, initial_routes=None,
                 directed=True, allow_self_loops=True):
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.schema = schema
        self.allow_self_loops = allow_self_loops
        self.insertion_log = []
        self.route_count = 0

        if initial_routes:
            for (pickup, drop), tensor in initial_routes.items():
                self.add_route(pickup, drop, tensor)

    # ---------------- construction ---------------- #
    @classmethod
    def from_dataframe(cls, df, schema=None, pickup_col="pickup_cluster",
                       drop_col="drop_cluster", grain_cols=("week_period", "hour"),
                       **graph_kwargs):
        schema = schema or RouteSchema.from_frame(
            df, grain_cols=grain_cols, exclude_cols=(pickup_col, drop_col))
        g = cls(schema, **graph_kwargs)
        g.add_frame(df, pickup_col=pickup_col, drop_col=drop_col)
        return g

    def add_frame(self, df, pickup_col="pickup_cluster", drop_col="drop_cluster"):
        df = df.dropna(subset=[pickup_col, drop_col])
        for (pickup, drop), grp in df.groupby([pickup_col, drop_col], sort=False):
            attrs = {}
            if "city" in grp:
                attrs["cities"] = set(grp["city"].dropna())
            for c in CATEGORICAL:
                if c in grp:
                    attrs[c] = Counter(grp[c].dropna())
            self.add_route(pickup, drop, RouteTensor.from_frame(self.schema, grp), **attrs)
        return self

    def add_route(self, pickup, drop, tensor: RouteTensor, **attrs):
        self._validate_cluster(pickup)
        self._validate_cluster(drop)
        if pickup == drop and not self.allow_self_loops:
            return

        self._upsert_node(pickup, role="pickup", cities=attrs.get("cities"))
        self._upsert_node(drop, role="drop", cities=attrs.get("cities"))
        was_new = self._add_or_update_edge(pickup, drop, tensor, **attrs)
        self.route_count += 1

        edge = self.graph[pickup][drop]
        self.insertion_log.append({
            "action": "new_route" if was_new else "merge_route",
            "pickup": pickup, "drop": drop,
            "count": edge["count"], "length": edge["length"],
            "ride_time": edge["ride_time"], "speed": edge["speed"],
            "coverage": edge["tensor"].coverage,
        })

    def _upsert_node(self, cluster, role, cities=None):
        if cluster not in self.graph:
            self.graph.add_node(cluster, count=0, pickup_count=0, drop_count=0, cities=set())
        node = self.graph.nodes[cluster]
        node["count"] += 1
        node[f"{role}_count"] += 1
        if cities:
            node["cities"] |= set(cities)

    def _add_or_update_edge(self, source, target, tensor, **attrs):
        if self.graph.has_edge(source, target):
            data = self.graph[source][target]
            data["tensor"].merge(tensor)
            data["count"] += 1
            if attrs.get("cities"):
                data["cities"] |= attrs["cities"]
            for c in CATEGORICAL:
                if c in attrs:
                    data[c].update(attrs[c])
            was_new = False
        else:
            self.graph.add_edge(
                source, target, tensor=tensor.copy(), count=1, kind="route",
                cities=set(attrs.get("cities") or ()),
                **{c: Counter(attrs.get(c) or {}) for c in CATEGORICAL},
            )
            was_new = True
        self._refresh_edge_metrics(source, target)
        return was_new

    def _refresh_edge_metrics(self, source, target):
        data, s = self.graph[source][target], self.schema
        summ = data["tensor"].summary()
        data["metrics"] = summ
        data["length"] = data["distance"] = data["weight"] = summ[s.length_metric]
        data["ride_time"] = summ[s.ride_time_metric]
        data["speed"] = summ["speed"]

    def _validate_cluster(self, cluster):
        if cluster is None or (isinstance(cluster, float) and np.isnan(cluster)):
            raise ValueError(f"Invalid cluster id: {cluster!r}")

    # ---------------- queries ---------------- #
    def bucket(self, pickup, drop, **grain_values):
        """Metrics for one route in one bucket, e.g. bucket('A', 'B', hour=9)."""
        return self.graph[pickup][drop]["tensor"].get(**grain_values)

    def edge_metrics(self, pickup, drop, **grain_values):
        """{length, ride_time, speed} for a route, overall or for buckets (hour=[8, 9])."""
        data, s = self.graph[pickup][drop], self.schema
        if not grain_values:
            return {k: data[k] for k in ("length", "ride_time", "speed")}
        summ = data["tensor"].summary(**grain_values)
        return {"length": summ[s.length_metric],
                "ride_time": summ[s.ride_time_metric],
                "speed": summ["speed"]}

    def edge_length(self, pickup, drop, **grain_values):
        return self.edge_metrics(pickup, drop, **grain_values)["length"]

    def edge_ride_time(self, pickup, drop, **grain_values):
        return self.edge_metrics(pickup, drop, **grain_values)["ride_time"]

    def edge_speed(self, pickup, drop, **grain_values):
        return self.edge_metrics(pickup, drop, **grain_values)["speed"]

    def speed_profile(self, pickup, drop):
        """Speed per bucket as a week_period x hour DataFrame."""
        return self.bucket(pickup, drop)["speed"].to_pandas()

    def route_summary(self, pickup, drop, **grain_values):
        return self.graph[pickup][drop]["tensor"].summary(**grain_values)

    def node_tensor(self, cluster, direction="out"):
        """Merge the tensors of a cluster outgoing ('out'), incoming ('in') or all routes."""
        g = self.graph
        if direction == "out":
            edges = g.out_edges(cluster, data=True) if g.is_directed() else g.edges(cluster, data=True)
        elif direction == "in":
            edges = g.in_edges(cluster, data=True) if g.is_directed() else g.edges(cluster, data=True)
        else:
            edges = list(g.edges(cluster, data=True))
            if g.is_directed():
                edges += list(g.in_edges(cluster, data=True))
        total = RouteTensor(self.schema)
        seen = set()
        for u, v, d in edges:
            if (u, v) not in seen:
                seen.add((u, v))
                total.merge(d["tensor"])
        return total

    def shortest_route(self, source, target, by="length", **grain_values):
        key = {"length": self.schema.length_metric,
               "ride_time": self.schema.ride_time_metric}[by]

        def weight(u, v, d):
            val = d["tensor"].summary(**grain_values)[key] if grain_values else d[by]
            return None if val is None or np.isnan(val) else val   # None hides the edge

        path = nx.shortest_path(self.graph, source, target, weight=weight)
        total = sum(weight(u, v, self.graph[u][v]) for u, v in zip(path, path[1:]))
        return path, total

    def as_dataset(self):
        """All route tensors stacked on a 'route' dimension."""
        xr = _require_xarray()
        edges = list(self.graph.edges(data=True))
        if not edges:
            return xr.Dataset()
        ds = xr.concat([d["tensor"].data for _, _, d in edges], dim="route")
        return ds.assign_coords(
            route=[f"{u}->{v}" for u, v, _ in edges],
            pickup=("route", [u for u, _, _ in edges]),
            drop=("route", [v for _, v, _ in edges]),
        )

    def to_frame(self, drop_empty=True):
        df = self.as_dataset().to_dataframe().reset_index()
        return df[df["row_count"] > 0].reset_index(drop=True) if drop_empty else df

    def edges_frame(self):
        """One row per route with its collapsed metrics and length."""
        rows = []
        for u, v, d in self.graph.edges(data=True):
            rows.append({"pickup_cluster": u, "drop_cluster": v,
                         "length": d["length"], "ride_time": d["ride_time"],
                         "speed": d["speed"], "count": d["count"],
                         "coverage": d["tensor"].coverage, **d["tensor"].summary()})
        return pd.DataFrame(rows)

    # ---------------- metric access ---------------- #
    def metric_names(self) -> list[str]:
        """Metrics you can pass as ``metric=`` anywhere in this class."""
        return [*self.schema.metric_names, "count"]
    def resolve_metric(self, metric=None) -> str:
        """Fall back to the schema default when a call does not name a metric."""
        return metric if metric is not None else self.schema.default_metric

    def _edge_data(self, pickup, drop):
        if not self.graph.has_edge(pickup, drop):
            raise KeyError(f"No route {pickup!r} -> {drop!r}")
        return self.graph[pickup][drop]

    def _value_from_edge(self, data, metric, grain_values):
        if metric == "count":
            return float(data.get("count", 0) or 0)
        resolved = self.schema.validate_metric(metric)
        if grain_values:
            summ = data["tensor"].summary(**grain_values)
        else:
            summ = data.get("metrics") or data["tensor"].summary()
        return float(summ[resolved])

    def edge_value(self, pickup, drop, metric=None, **grain_values) -> float:
        """One metric for one route, optionally restricted to grain buckets."""
        metric = self.resolve_metric(metric)
        return self._value_from_edge(self._edge_data(pickup, drop), metric, grain_values)

    def incident(self, node, direction="out"):
        """Yield ``(partner, source, target, data)`` for a node's routes."""
        g = self.graph
        if node not in g:
            raise KeyError(f"Unknown cluster: {node!r}")
        if not g.is_directed():
            for partner in g.neighbors(node):
                yield partner, node, partner, g[node][partner]
            return
        if direction in ("out", "drop", "drops", "downstream"):
            for _, v, data in g.out_edges(node, data=True):
                yield v, node, v, data
        elif direction in ("in", "source", "sources", "upstream"):
            for u, _, data in g.in_edges(node, data=True):
                yield u, u, node, data
        elif direction in ("both", "all"):
            seen = set()
            for u, v, data in list(g.out_edges(node, data=True)) + list(g.in_edges(node, data=True)):
                if (u, v) in seen:
                    continue
                seen.add((u, v))
                yield (v if u == node else u), u, v, data
        else:
            raise ValueError(f"direction must be 'in', 'out' or 'both', got {direction!r}")

    def partners(self, node, direction="out", metric=None, top=None,
                 min_value=None, exclude=None, ascending=False, **grain_values):
        """Ranked neighbours of a node as ``[(partner, value), ...]``.

        ``direction='in'`` answers *where do this cluster's orders come from*,
        ``direction='out'`` answers *where do they go*.
        """
        metric = self.resolve_metric(metric)
        skip = set(exclude or ())
        rows = []
        for partner, _, _, data in self.incident(node, direction):
            if partner in skip:
                continue
            value = self._value_from_edge(data, metric, grain_values)
            if min_value is not None and not (value >= min_value):
                continue
            rows.append((partner, value))

        miss = float("inf") if ascending else float("-inf")
        rows.sort(key=lambda item: miss if np.isnan(item[1]) else item[1], reverse=not ascending)
        return rows[:top] if top else rows

    def top_drops(self, node, top=5, metric=None, **kwargs):
        """Top clusters this cluster sends orders to."""
        metric = self.resolve_metric(metric)
        return self.partners(node, direction="out", metric=metric, top=top, **kwargs)

    def top_sources(self, node, top=5, metric=None, **kwargs):
        """Top clusters this cluster receives orders from."""
        metric = self.resolve_metric(metric)
        return self.partners(node, direction="in", metric=metric, top=top, **kwargs)

    def partners_frame(self, node, direction="out", metric=None, top=None,
                       extra_metrics=(), **grain_values):
        """Ranked partners as a DataFrame, with a share-of-total column."""
        metric = self.resolve_metric(metric)
        ranked = self.partners(node, direction=direction, metric=metric, top=top, **grain_values)
        total = float(np.nansum([v for _, v in ranked])) if ranked else 0.0
        rows = []
        for rank, (partner, value) in enumerate(ranked, start=1):
            row = {
                "rank": rank,
                "cluster": node,
                "partner": partner,
                "direction": direction,
                metric: value,
                "share": (value / total) if total else np.nan,
            }
            pickup, drop = (node, partner) if direction.startswith("out") else (partner, node)
            data = self.graph[pickup][drop]
            for extra in extra_metrics:
                row[extra] = self._value_from_edge(data, extra, grain_values)
            row["routes"] = data.get("count", 1)
            rows.append(row)
        return pd.DataFrame(rows)

    # ---------------- cluster level views ---------------- #
    def node_summary(self, node, direction="out", **grain_values) -> dict:
        """Every metric for a cluster's inbound, outbound or combined flow."""
        return self.node_tensor(node, direction=direction).summary(**grain_values)

    def node_value(self, node, metric=None, direction="out", **grain_values) -> float:
        """One metric for a cluster, correctly weighted across its routes."""
        metric = self.resolve_metric(metric)
        if metric == "count":
            return float(sum(1 for _ in self.incident(node, direction)))
        resolved = self.schema.validate_metric(metric)
        return float(self.node_summary(node, direction=direction, **grain_values)[resolved])

    def nodes_frame(self, metric=None, **grain_values):
        """One row per cluster: inbound vs outbound volume and net balance."""
        metric = self.resolve_metric(metric)
        rows = []
        for node in self.graph.nodes:
            out_value = self.node_value(node, metric=metric, direction="out", **grain_values)
            in_value = self.node_value(node, metric=metric, direction="in", **grain_values)
            data = self.graph.nodes[node]
            rows.append({
                "cluster": node,
                f"out_{metric}": out_value,
                f"in_{metric}": in_value,
                "net": np.nansum([out_value, -in_value]) if not (
                    np.isnan(out_value) and np.isnan(in_value)) else np.nan,
                "out_routes": self.graph.out_degree(node) if self.graph.is_directed()
                else self.graph.degree(node),
                "in_routes": self.graph.in_degree(node) if self.graph.is_directed()
                else self.graph.degree(node),
                "pickup_count": data.get("pickup_count", 0),
                "drop_count": data.get("drop_count", 0),
                "cities": ", ".join(sorted(str(c) for c in data.get("cities", ()))) or None,
            })
        frame = pd.DataFrame(rows)
        return frame.sort_values(f"out_{metric}", ascending=False, ignore_index=True) if rows else frame

    def rank_routes(self, metric=None, top=None, ascending=False, **grain_values):
        """Routes ranked by any metric, as a DataFrame."""
        metric = self.resolve_metric(metric)
        rows = []
        for u, v, data in self.graph.edges(data=True):
            rows.append({
                "pickup_cluster": u,
                "drop_cluster": v,
                metric: self._value_from_edge(data, metric, grain_values),
                "routes": data.get("count", 1),
                "coverage": data["tensor"].coverage,
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame = frame.sort_values(metric, ascending=ascending, na_position="last", ignore_index=True)
        return frame.head(top) if top else frame

    def flow_matrix(self, metric=None, nodes=None, top=None, **grain_values):
        """Pickup x drop matrix of one metric (rows = pickup, columns = drop)."""
        metric = self.resolve_metric(metric)
        keep = set(nodes) if nodes is not None else None
        rows = []
        for u, v, data in self.graph.edges(data=True):
            if keep is not None and (u not in keep or v not in keep):
                continue
            rows.append({"pickup_cluster": u, "drop_cluster": v,
                         "value": self._value_from_edge(data, metric, grain_values)})
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        matrix = frame.pivot_table(index="pickup_cluster", columns="drop_cluster",
                                   values="value", aggfunc="sum")
        if top:
            order = matrix.sum(axis=1).sort_values(ascending=False).head(top).index
            cols = matrix.sum(axis=0).sort_values(ascending=False).head(top).index
            matrix = matrix.loc[order, cols]
        return matrix

    def total_summary(self, **grain_values) -> dict:
        """Every metric across the whole graph."""
        total = RouteTensor(self.schema)
        for _, _, data in self.graph.edges(data=True):
            total.merge(data["tensor"])
        return total.summary(**grain_values)

    def get_graph_stats(self) -> dict:
        """Headline counts, mirroring AffinityGraph.get_graph_stats."""
        coverage = [data["tensor"].coverage for _, _, data in self.graph.edges(data=True)]
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "route_count": self.route_count,
            "directed": self.graph.is_directed(),
            "grains": {k: len(v) for k, v in self.schema.grains.items()},
            "sum_metrics": len(self.schema.sum_metrics),
            "mean_metrics": len(self.schema.mean_metrics),
            "avg_coverage": round(float(np.mean(coverage)), 4) if coverage else 0.0,
        }

    def routes(self):
        """``[(pickup, drop), ...]`` for every edge."""
        return [(u, v) for u, v, _ in self.graph.edges(data=True)]

    def has_route(self, pickup, drop) -> bool:
        return self.graph.has_edge(pickup, drop)

    def simple_graph(self, metric=None, **grain_values):
        """Plain NetworkX graph with one scalar weight per edge, for export."""
        metric = self.resolve_metric(metric)
        out = nx.DiGraph() if self.graph.is_directed() else nx.Graph()
        out.add_nodes_from(self.graph.nodes)
        for u, v, data in self.graph.edges(data=True):
            value = self._value_from_edge(data, metric, grain_values)
            out.add_edge(u, v, weight=value, **{metric: value, "routes": data.get("count", 1)})
        return out

    def subgraph(self, nodes) -> "RouteGraph":
        """A new RouteGraph restricted to ``nodes`` (tensors are shared, not copied)."""
        keep = set(nodes)
        out = RouteGraph(self.schema, directed=self.graph.is_directed(),
                         allow_self_loops=self.allow_self_loops)
        out.graph = self.graph.subgraph(keep).copy()
        out.route_count = out.graph.number_of_edges()
        return out

    # ---------------- flow expansion ---------------- #
    @staticmethod
    def _level_plan(spec) -> list:
        """Normalize a per-level top-k spec into a list, one entry per hop.

        ``3`` -> one level keeping the top 3; ``[5, 3, 2]`` -> three levels
        keeping 5 then 3 then 2; ``None``/``0`` -> no levels; ``None`` inside a
        list -> keep every partner at that level.
        """
        if spec is None or spec is False:
            return []
        if isinstance(spec, bool):
            return [None]
        if isinstance(spec, (int, np.integer)):
            return [int(spec)] if int(spec) > 0 else []
        plan = []
        for k in spec:
            if k is None:
                plan.append(None)
            elif int(k) <= 0:
                break
            else:
                plan.append(int(k))
        return plan

    def flow_subgraph(self, focus, upstream=5, downstream=5, metric=None,
                      per_parent=True, min_value=None, include_cross_edges=False,
                      exclude_self_loops=True, **grain_values):
        """Expand around ``focus`` and keep only the top partners at each hop.

        ``upstream`` walks against the arrows (where orders come from) and
        ``downstream`` walks along them (where orders go). Both accept an int
        for a single level or a list for one top-k per level, so
        ``upstream=[5, 3, 2]`` keeps the top 5 sources of the focus, the top 3
        sources of each of those, then the top 2 of each of those.

        Returns a ``nx.DiGraph`` whose nodes carry ``level`` (0 for focus,
        negative upstream, positive downstream) and whose edges carry ``value``.
        """
        metric = self.resolve_metric(metric)
        metric = "count" if metric == "count" else self.schema.validate_metric(metric)
        focus_nodes = [f for f in _as_list(focus)]
        missing = [f for f in focus_nodes if f not in self.graph]
        if missing:
            raise KeyError(f"Unknown cluster(s): {missing}")

        sub = nx.DiGraph()
        for node in focus_nodes:
            sub.add_node(node, level=0, depth=0, side="focus", value=np.nan, is_focus=True)

        def expand(direction, plan, sign):
            frontier = list(focus_nodes)
            side = "source" if sign < 0 else "drop"
            for hop, keep in enumerate(plan, start=1):
                if not frontier:
                    break
                picks = []
                if per_parent:
                    for parent in frontier:
                        picks += [(parent, p, v) for p, v in self.partners(
                            parent, direction=direction, metric=metric, top=keep,
                            min_value=min_value, **grain_values)]
                else:
                    pool = []
                    for parent in frontier:
                        pool += [(parent, p, v) for p, v in self.partners(
                            parent, direction=direction, metric=metric,
                            min_value=min_value, **grain_values)]
                    pool.sort(key=lambda item: -np.inf if np.isnan(item[2]) else item[2],
                              reverse=True)
                    picks = pool[:keep] if keep else pool

                next_frontier = []
                for parent, partner, value in picks:
                    if exclude_self_loops and partner == parent:
                        continue
                    if partner not in sub:
                        sub.add_node(partner, level=sign * hop, depth=hop, side=side,
                                     value=0.0, is_focus=False)
                        next_frontier.append(partner)
                    if not np.isnan(value):
                        node = sub.nodes[partner]
                        node["value"] = float(np.nansum([node.get("value", 0.0), value]))
                    src, dst = (parent, partner) if sign > 0 else (partner, parent)
                    sub.add_edge(src, dst, value=value, metric=metric,
                                 depth=hop, side=side,
                                 routes=self.graph[src][dst].get("count", 1))
                frontier = next_frontier

        expand("in", self._level_plan(upstream), -1)
        expand("out", self._level_plan(downstream), 1)

        if include_cross_edges:
            selected = list(sub.nodes)
            for u in selected:
                for v in selected:
                    if u == v or sub.has_edge(u, v) or not self.graph.has_edge(u, v):
                        continue
                    sub.add_edge(u, v, side="cross", depth=None, metric=metric,
                                 value=self.edge_value(u, v, metric, **grain_values),
                                 routes=self.graph[u][v].get("count", 1))

        sub.graph.update({
            "focus": focus_nodes,
            "metric": metric,
            "grain": dict(grain_values),
            "upstream": self._level_plan(upstream),
            "downstream": self._level_plan(downstream),
        })
        return sub

    def _split_kwargs(self, kwargs):
        """Separate grain filters from plain options in a mixed kwargs dict."""
        grain = {k: v for k, v in kwargs.items() if k in self.schema.grain_names}
        options = {k: v for k, v in kwargs.items() if k not in grain}
        return grain, options

    def flow_frame(self, focus, upstream=5, downstream=5, metric=None,
                   extra_metrics=(), **kwargs):
        """The same expansion as :meth:`flow_subgraph`, as a tidy DataFrame."""
        metric = self.resolve_metric(metric)
        grain, options = self._split_kwargs(kwargs)
        sub = self.flow_subgraph(focus, upstream=upstream, downstream=downstream,
                                 metric=metric, **options, **grain)
        rows = []
        for u, v, data in sub.edges(data=True):
            row = {
                "side": data["side"],
                "hop": data["depth"],
                "pickup_cluster": u,
                "drop_cluster": v,
                metric: data["value"],
                "routes": data["routes"],
            }
            for extra in extra_metrics:
                row[extra] = self.edge_value(u, v, extra, **grain)
            rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["side", "hop", metric], ascending=[True, True, False],
                                 na_position="last", ignore_index=True)

    def flow_tree(self, focus, upstream=5, downstream=5, metric=None,
                  value_format="{:,.1f}", **kwargs) -> str:
        """A plain-text tree of the same expansion, handy for quick checks."""
        metric = self.resolve_metric(metric)
        grain, options = self._split_kwargs(kwargs)
        sub = self.flow_subgraph(focus, upstream=upstream, downstream=downstream,
                                 metric=metric, **options, **grain)
        lines = []

        def walk(node, side, prefix, seen):
            edges = sub.out_edges(node, data=True) if side == "drop" else sub.in_edges(node, data=True)
            partners = []
            for u, v, data in edges:
                if data.get("side") != side:
                    continue
                partner = v if side == "drop" else u
                if partner not in seen:
                    partners.append((partner, data["value"]))
            partners.sort(key=lambda item: -np.inf if np.isnan(item[1]) else item[1], reverse=True)
            for index, (partner, value) in enumerate(partners):
                last = index == len(partners) - 1
                arrow = "->" if side == "drop" else "<-"
                branch = "+-- " if last else "|-- "
                lines.append(f"{prefix}{branch}{arrow} {partner}  [{value_format.format(value)}]")
                walk(partner, side, prefix + ("    " if last else "|   "), seen | {partner})

        for node in sub.graph["focus"]:
            grain_note = f"  {sub.graph['grain']}" if sub.graph["grain"] else ""
            lines.append(f"{node}  ({metric}{grain_note})")
            if sub.graph["upstream"]:
                lines.append("  sources (orders coming in)")
                walk(node, "source", "  ", {node})
            if sub.graph["downstream"]:
                lines.append("  drops (orders going out)")
                walk(node, "drop", "  ", {node})
        return "\n".join(lines)

    # ---------------- plotting ---------------- #
    def plot_flow(self, focus, **kwargs):
        """Draw the focus clusters with their top sources and drops per level."""
        from .route_viz import plot_flow

        return plot_flow(self, focus, **kwargs)

    def plot_partners(self, node, **kwargs):
        """Back-to-back bars: top sources on the left, top drops on the right."""
        from .route_viz import plot_partners

        return plot_partners(self, node, **kwargs)

    def plot_profile(self, pickup=None, drop=None, node=None, **kwargs):
        """Heatmap of one metric across two grains for a route or a cluster."""
        from .route_viz import plot_profile

        return plot_profile(self, pickup=pickup, drop=drop, node=node, **kwargs)

    def plot_matrix(self, **kwargs):
        """Pickup x drop heatmap of one metric."""
        from .route_viz import plot_matrix

        return plot_matrix(self, **kwargs)

    def plot_graph(self, **kwargs):
        """Draw the whole route graph, sized and coloured by one metric."""
        from .route_viz import plot_route_graph

        return plot_route_graph(self, **kwargs)
