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

import operator
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

#: Whole words that mark a numeric column as a bucket rather than a measurement.
#: Only these are kept out of the metrics - a count column with few distinct
#: values (cancelled 0-11, riders 1-30) is still a measurement and stays.
DIMENSION_WORDS = frozenset({
    "hour", "hours", "hr", "hrs", "day", "days", "dow", "weekday", "week",
    "weeks", "month", "months", "quarter", "year", "years", "date", "datetime",
    "slot", "slots", "timeslot", "period", "bucket", "bin", "band", "tier",
    "cohort", "variant", "arm", "segment", "shift", "wave",
})


def _words(name) -> list:
    """Split a column name into its words: ``avg_ride_time`` -> avg, ride, time."""
    import re

    return [part for part in re.split(r"[^a-z0-9]+", str(name).lower()) if part]


def looks_like_dimension(name) -> bool:
    """Whether a column name denotes a bucket you slice by, not a measurement."""
    return any(word in DIMENSION_WORDS for word in _words(name))

#: Bucket used when the data has no grain dimension of its own.
FLAT_GRAIN = "__all__"

ROWS = "__rows"

#: Metric aliases that resolve against the schema instead of a column name.
DERIVED_METRICS = ("length", "ride_time", "speed", "row_count", "count")

#: Operators usable in a derived metric spec such as ``("orders", "/", "requests")``.
DERIVED_OPS = {
    "/": lambda a, b: np.divide(a, b, out=np.full(np.shape(np.asarray(b, dtype=float)),
                                                  np.nan, dtype=float),
                                where=(np.asarray(b, dtype=float) != 0)),
    "*": lambda a, b: np.asarray(a, dtype=float) * np.asarray(b, dtype=float),
    "+": lambda a, b: np.asarray(a, dtype=float) + np.asarray(b, dtype=float),
    "-": lambda a, b: np.asarray(a, dtype=float) - np.asarray(b, dtype=float),
}


def _as_derived(spec):
    """Turn a derived-metric spec into a function of the collapsed metrics.

    ``("orders", "/", "requests")`` becomes a division applied *after* the
    metrics are aggregated, which is the whole point: a ratio of totals is not
    the total of the ratios. A callable is passed through unchanged and is
    handed the dict of already-collapsed values.
    """
    if callable(spec):
        return spec
    if not (isinstance(spec, (tuple, list)) and len(spec) == 3):
        raise ValueError(
            f"A derived metric is a callable or (left, op, right), got {spec!r}"
        )
    left, op, right = spec
    if op not in DERIVED_OPS:
        raise ValueError(f"Unknown operator {op!r}; use one of {sorted(DERIVED_OPS)}")
    apply = DERIVED_OPS[op]

    def compute(values):
        return apply(values[left], values[right])

    compute.spec = (left, op, right)
    return compute


#: Operators a filter test can use.
COMPARISONS = {
    ">": operator.gt, "gt": operator.gt,
    ">=": operator.ge, "ge": operator.ge, "min": operator.ge,
    "<": operator.lt, "lt": operator.lt,
    "<=": operator.le, "le": operator.le, "max": operator.le,
    "==": operator.eq, "eq": operator.eq,
    "!=": operator.ne, "ne": operator.ne,
}


def _as_test(spec):
    """Turn a filter value into a predicate on one metric value.

    ``100`` means ``>= 100``, ``(">", 0.25)`` names an operator, and
    ``(10, 50)`` is an inclusive range. A NaN metric never passes.
    """
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise ValueError(f"A filter test needs 2 items, got {spec!r}")
        first, second = spec
        if isinstance(first, str):
            try:
                compare = COMPARISONS[first]
            except KeyError:
                raise ValueError(
                    f"Unknown operator {first!r}; use one of {sorted(COMPARISONS)}"
                ) from None
            return lambda value: bool(not np.isnan(value) and compare(value, second))
        low, high = float(first), float(second)
        return lambda value: bool(not np.isnan(value) and low <= value <= high)

    threshold = float(spec)
    return lambda value: bool(not np.isnan(value) and value >= threshold)


def suggest_grains(df, max_levels: int = 48, exclude=()) -> list:
    """Columns that look like grain dimensions: few repeated values.

    A grain is something you slice by - a day part, an hour, a segment - so the
    test is a small number of distinct values relative to the frame.
    """
    skip = set(exclude) | set(ID_COLS)
    out = []
    for column in df.columns:
        if column in skip:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        levels = series.nunique()
        if levels <= 1 or levels > max_levels or levels >= len(series):
            continue
        if pd.api.types.is_float_dtype(series):
            continue                                  # measurements, not buckets
        out.append(column)
    return out


def suggest_ids(df, exclude=()) -> list:
    """Columns that look like cluster or entity ids rather than measurements."""
    skip = set(exclude)
    hints = ("cluster", "hex", "id", "zone", "cell", "node", "region")
    return [c for c in df.columns
            if c not in skip and any(h in str(c).lower() for h in hints)]


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
    derived_metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.grains:
            raise ValueError("Schema needs at least one grain dimension")
        self.grains = {k: list(v) for k, v in self.grains.items()}
        self._indexers = {k: pd.Index(v) for k, v in self.grains.items()}
        for m in (self.length_metric, self.ride_time_metric):
            if m not in self.mean_metrics:
                raise ValueError(f"{m!r} must be one of mean_metrics")
        self.derived_metrics = dict(self.derived_metrics or {})
        self._derived = {name: _as_derived(spec)
                         for name, spec in self.derived_metrics.items()}
        clash = set(self._derived) & set(self.sum_metrics + self.mean_metrics)
        if clash:
            raise ValueError(
                f"Derived metric name(s) already used by a column: {sorted(clash)}"
            )

    @classmethod
    def from_frame(cls, df, grain_cols=(), exclude_cols=(), metrics=None, **kwargs):
        """Infer the grains and the metric columns from a DataFrame.

        Your column names are your own, so name them:

            RouteSchema.from_frame(df, grain_cols=("day_part", "hour_of_day"),
                                   metrics=["trips", "gmv", "avg_km"])

        Grains are optional. Pass none and the whole frame collapses into a
        single bucket; pass any columns you want to slice by, including a
        cohort column such as ``variant`` holding test and control.

        With ``metrics=None`` every numeric column becomes a metric except
        grains, ids, exclusions, and columns whose name denotes a bucket
        (``hour``, ``week``, ``slot``, ...). Integer counts are kept: few
        distinct values does not make a column a dimension. Name them with
        ``metrics=`` when the guess is wrong, or set the split yourself with
        ``sum_metrics`` / ``mean_metrics``.
        """
        grain_cols = tuple(grain_cols or ())
        if not grain_cols:
            grain_cols = (FLAT_GRAIN,)
            df = df.assign(**{FLAT_GRAIN: "all"})
        grains = {}
        for col in grain_cols:
            if col not in df:
                raise KeyError(
                    f"Grain column {col!r} is not in the frame. "
                    f"Pass grain_cols= with your own column names; "
                    f"candidates here: {suggest_grains(df)}"
                )
            if col == "hour":
                hours = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
                grains[col] = sorted(set(range(24)) | set(hours))
            else:
                # plain Python values, so repr and JSON stay readable
                levels = [v.item() if hasattr(v, "item") else v
                          for v in df[col].dropna().unique().tolist()]
                grains[col] = sorted(levels)

        if metrics is not None:
            chosen = list(metrics)
            if not chosen:
                raise ValueError("metrics= is empty; pass at least one column")
            absent = [m for m in chosen if m not in df.columns]
            if absent:
                raise KeyError(f"Metric column(s) not in the frame: {absent}")
            inferred_sums, inferred_means = classify_metrics(df[chosen])
            unusable = [m for m in chosen if m not in inferred_sums + inferred_means]
            if unusable:
                raise TypeError(
                    f"Metric column(s) are not numeric and cannot be measured: {unusable}"
                )
        else:
            skip = set(grain_cols) | set(exclude_cols) | set(CATEGORICAL) | set(ID_COLS)
            # summing `hour` is meaningless, so columns whose name denotes a
            # bucket stay out of the metrics even when they are not grains.
            # This is deliberately name-based: an integer column with few
            # distinct values is usually still a count, and counts are metrics.
            skip |= {c for c in df.columns if looks_like_dimension(c)}
            inferred_sums, inferred_means = classify_metrics(df, exclude=skip)

        kwargs.setdefault("sum_metrics", inferred_sums)
        mean = list(kwargs.get("mean_metrics", inferred_means))
        derived = kwargs.get("derived_metrics") or {}
        # the length and ride-time metrics stay in the schema even when the
        # frame lacks them, so speed is NaN instead of the schema refusing to build
        for key, fallback in (("length_metric", "avg_distance"),
                              ("ride_time_metric", "avg_duration")):
            metric = kwargs.get(key, fallback)
            if metric not in mean:
                mean.append(metric)
        kwargs["mean_metrics"] = mean
        kwargs["derived_metrics"] = derived
        return cls(grains=grains, **kwargs)

    @property
    def grain_names(self):
        return list(self.grains)

    @property
    def shape(self):
        return tuple(len(v) for v in self.grains.values())

    @property
    def is_flat(self) -> bool:
        """True when the schema has no real grain, just one bucket."""
        return self.grain_names == [FLAT_GRAIN]

    def ensure_grain(self, df):
        """Add the flat bucket column when the schema has no real grain."""
        if self.is_flat and FLAT_GRAIN not in df.columns:
            return df.assign(**{FLAT_GRAIN: "all"})
        return df

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
        return ["row_count", *self.sum_metrics, *self.mean_metrics, "speed",
                *self._derived]

    def apply_derived(self, values: dict) -> dict:
        """Add the derived metrics to a dict of already-collapsed metrics.

        Applied after aggregation at every level, so a ratio is always a ratio
        of the totals for whatever is in scope - one bucket, one route, a
        cluster's inbound side, or the whole graph.
        """
        for name, compute in self._derived.items():
            try:
                values[name] = compute(values)
            except KeyError as exc:
                raise KeyError(
                    f"Derived metric {name!r} needs {exc.args[0]!r}, "
                    "which is not one of the schema's metrics"
                ) from None
        return values

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
        s.apply_derived(out)                      # per bucket, after collapsing
        return xr.Dataset({k: (dims, np.asarray(v)) for k, v in out.items()},
                          coords=s.grains)

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
        s.apply_derived(out)                      # after the sums and the means
        return {k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else v)
                for k, v in out.items()}

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
    #: Keyword arguments that belong to the schema rather than the graph.
    SCHEMA_OPTIONS = ("metrics", "sum_metrics", "mean_metrics", "weight_col",
                      "length_metric", "ride_time_metric", "speed_scale",
                      "derived_metrics", "exclude_cols")

    @classmethod
    def from_dataframe(cls, df, schema=None, pickup_col="pickup_cluster",
                       drop_col="drop_cluster", grain_cols=(), **kwargs):
        """Build the schema, the tensors and the graph from one frame.

        ``grain_cols`` is optional: pass none and everything collapses into one
        bucket, or name the columns you want to slice by. Schema options
        (``metrics``, ``weight_col``, ``length_metric``, ...) may be passed
        straight through, alongside graph options such as ``directed``.
        """
        schema_kwargs = {k: kwargs.pop(k) for k in list(kwargs)
                         if k in cls.SCHEMA_OPTIONS}
        if schema is not None and schema_kwargs:
            raise TypeError(
                f"Pass either schema= or schema options, not both: "
                f"{sorted(schema_kwargs)}"
            )
        schema_kwargs.setdefault("exclude_cols", (pickup_col, drop_col))
        schema = schema or RouteSchema.from_frame(df, grain_cols=grain_cols,
                                                  **schema_kwargs)
        g = cls(schema, **kwargs)
        g.add_frame(df, pickup_col=pickup_col, drop_col=drop_col)
        return g

    def add_frame(self, df, pickup_col="pickup_cluster", drop_col="drop_cluster"):
        df = self.schema.ensure_grain(df.dropna(subset=[pickup_col, drop_col]))
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

    def edges_frame(self, metrics=None, **grain_values):
        """One row per route.

        ``metrics`` picks which columns to put on each edge; the default is
        every metric the schema carries. Grain filters narrow the window::

            graph.edges_frame(metrics=["orders", "speed"], hour=[8, 9])
        """
        wanted = [self.schema.validate_metric(m) for m in metrics] if metrics else None
        rows = []
        for u, v, d in self.graph.edges(data=True):
            summary = (d["tensor"].summary(**grain_values) if grain_values
                       else (d.get("metrics") or d["tensor"].summary()))
            row = {"pickup_cluster": u, "drop_cluster": v,
                   "count": d["count"], "coverage": d["tensor"].coverage}
            if wanted is None:
                row.update({"length": d["length"], "ride_time": d["ride_time"],
                            "speed": d["speed"]})
                row.update(summary)
            else:
                row.update({m: summary[m] for m in wanted})
            rows.append(row)
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
                 min_value=None, exclude=None, ascending=False, rank_by="edge",
                 node_direction="both", **grain_values):
        """Ranked neighbours of a node as ``[(partner, value), ...]``.

        ``direction='in'`` answers *where do this cluster's orders come from*,
        ``direction='out'`` answers *where do they go*.

        ``rank_by`` decides what the ranking measures:

        * ``"edge"`` (default) - the metric on the route between the two, so
          the top 5 are the five biggest routes into this cluster.
        * ``"node"`` - the metric on the partner cluster itself, so the top 5
          are the five biggest clusters that feed it, however small the route
          between them happens to be.

        ``node_direction`` picks which side of the partner is measured when
        ranking by node: ``"both"``, ``"in"`` or ``"out"``.
        """
        metric = self.resolve_metric(metric)
        if rank_by not in ("edge", "node"):
            raise ValueError("rank_by must be 'edge' or 'node'")
        skip = set(exclude or ())
        rows = []
        for partner, _, _, data in self.incident(node, direction):
            if partner in skip:
                continue
            value = (self._value_from_edge(data, metric, grain_values)
                     if rank_by == "edge"
                     else self.node_value(partner, metric=metric,
                                          direction=node_direction, **grain_values))
            if min_value is not None and not (value >= min_value):
                continue
            rows.append((partner, value))

        miss = float("inf") if ascending else float("-inf")
        rows.sort(key=lambda item: miss if np.isnan(item[1]) else item[1], reverse=not ascending)
        return rows[:top] if top else rows

    def top_drops(self, node, top=5, metric=None, rank_by="edge", **kwargs):
        """Top clusters this cluster sends orders to.

        ``rank_by="node"`` ranks them by their own volume instead of by the
        route from here.
        """
        metric = self.resolve_metric(metric)
        return self.partners(node, direction="out", metric=metric, top=top,
                             rank_by=rank_by, **kwargs)

    def top_sources(self, node, top=5, metric=None, rank_by="edge", **kwargs):
        """Top clusters this cluster receives orders from.

        ``rank_by="node"`` ranks them by their own volume instead of by the
        route into here.
        """
        metric = self.resolve_metric(metric)
        return self.partners(node, direction="in", metric=metric, top=top,
                             rank_by=rank_by, **kwargs)

    def partners_frame(self, node, direction="out", metric=None, top=None,
                       extra_metrics=(), rank_by="edge", node_direction="both",
                       **grain_values):
        """Ranked partners as a DataFrame, with a share-of-total column.

        ``rank_by="node"`` ranks by the partner cluster's own total rather than
        by the route between them.
        """
        metric = self.resolve_metric(metric)
        ranked = self.partners(node, direction=direction, metric=metric, top=top,
                               rank_by=rank_by, node_direction=node_direction,
                               **grain_values)
        total = float(np.nansum([v for _, v in ranked])) if ranked else 0.0
        rows = []
        for rank, (partner, value) in enumerate(ranked, start=1):
            row = {
                "rank": rank,
                "cluster": node,
                "partner": partner,
                "direction": direction,
                "ranked_by": rank_by,
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

    def node_split(self, node, *, metrics=None, diff=None, diff_order="out-in",
                   **grain_values) -> dict:
        """Every chosen metric for one cluster, inbound and outbound side by side.

        Sums are summed and means are weighted-averaged, both from the same
        tensors the routes carry - so the inbound average is weighted by
        :attr:`RouteSchema.weight_col` across the inbound routes, not a plain
        average of their averages.

            graph.node_split("A1")                       # every metric, no diff
            graph.node_split("A1", metrics=["orders"], diff=True)
            {"orders": {"in": 30.0, "out": 30.0, "diff": 0.0}}

        Everything past the cluster is optional: ``metrics`` defaults to every
        metric the schema carries, ``diff`` picks which of them also get the
        difference (``True`` for all, or a list of names), and ``diff_order``
        flips which way it is taken.
        """
        if diff_order not in ("out-in", "in-out"):
            raise ValueError("diff_order must be 'out-in' or 'in-out'")
        wanted = [self.schema.validate_metric(m)
                  for m in (metrics or self.schema.metric_names)]
        if diff is True or diff == "all":
            differenced = set(wanted)
        elif diff:
            differenced = {self.schema.validate_metric(m) for m in diff}
        else:
            differenced = set()

        inbound = self.node_summary(node, direction="in", **grain_values)
        outbound = self.node_summary(node, direction="out", **grain_values)

        out = {}
        for metric in wanted:
            entry = {"in": float(inbound[metric]), "out": float(outbound[metric])}
            if metric in differenced:
                first, second = ((entry["out"], entry["in"]) if diff_order == "out-in"
                                 else (entry["in"], entry["out"]))
                entry["diff"] = first - second
            out[metric] = entry
        return out

    def node_metrics_frame(self, *, metrics=None, diff=None, diff_order="out-in",
                           nodes=None, **grain_values):
        """One row per cluster with ``in_``/``out_`` columns for each metric.

        The companion of :meth:`node_split` across the whole graph. Called bare
        it gives every metric for every cluster; every argument is optional::

            graph.node_metrics_frame()
            graph.node_metrics_frame(metrics=["orders", "requests"], diff=["orders"])

        gives ``in_orders``, ``out_orders``, ``diff_orders``, ``in_requests``,
        ``out_requests``, ``in_avg_distance``, ``out_avg_distance``. Sums are
        summed, means are weighted by the schema's weight column, and a side
        with no routes is NaN rather than zero.
        """
        wanted = [self.schema.validate_metric(m)
                  for m in (metrics or self.schema.metric_names)]
        clusters = list(nodes) if nodes is not None else list(self.graph.nodes)
        rows = []
        for cluster in clusters:
            if cluster not in self.graph:
                raise KeyError(f"Unknown cluster: {cluster!r}")
            split = self.node_split(cluster, metrics=wanted, diff=diff,
                                    diff_order=diff_order, **grain_values)
            row = {"cluster": cluster,
                   "in_routes": self.graph.in_degree(cluster)
                   if self.graph.is_directed() else self.graph.degree(cluster),
                   "out_routes": self.graph.out_degree(cluster)
                   if self.graph.is_directed() else self.graph.degree(cluster)}
            for metric in wanted:
                row[f"in_{metric}"] = split[metric]["in"]
                row[f"out_{metric}"] = split[metric]["out"]
                if "diff" in split[metric]:
                    row[f"diff_{metric}"] = split[metric]["diff"]
            rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        # sort by the headline metric when it is present, else the first asked for
        lead = (self.schema.default_metric if self.schema.default_metric in wanted
                else wanted[0])
        return frame.sort_values(f"out_{lead}", ascending=False,
                                 na_position="last", ignore_index=True)

    @property
    def weight_col(self):
        """The column the mean metrics are weighted by, fixed when built."""
        return self.schema.weight_col

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


    # ---------------- filtering ---------------- #
    def edge_passes(self, pickup, drop, edge_filter, **grain_values) -> bool:
        """Whether one route satisfies every test in ``edge_filter``."""
        return self._passes(self._edge_data(pickup, drop), edge_filter, grain_values)

    def _passes(self, data, spec, grain_values) -> bool:
        for metric, test in (spec or {}).items():
            if not _as_test(test)(self._value_from_edge(data, metric, grain_values)):
                return False
        return True

    def _node_passes(self, node, spec, direction, grain_values) -> bool:
        for metric, test in (spec or {}).items():
            value = self.node_value(node, metric=metric, direction=direction, **grain_values)
            if not _as_test(test)(value):
                return False
        return True

    def matching_routes(self, edge_filter=None, node_filter=None,
                        node_direction="both", **grain_values) -> list:
        """``[(pickup, drop), ...]`` for the routes a filter accepts."""
        clusters = None
        if node_filter:
            clusters = set(self.matching_clusters(node_filter, direction=node_direction,
                                                  **grain_values))
        out = []
        for u, v, data in self.graph.edges(data=True):
            if clusters is not None and (u not in clusters or v not in clusters):
                continue
            if self._passes(data, edge_filter, grain_values):
                out.append((u, v))
        return out

    def matching_clusters(self, node_filter, direction="both", **grain_values) -> list:
        """Clusters a node filter accepts."""
        return [node for node in self.graph.nodes
                if self._node_passes(node, node_filter, direction, grain_values)]

    def keep(self, edge_filter=None, node_filter=None, node_direction="both",
             drop_isolated=True, **grain_values) -> "RouteGraph":
        """A NEW graph holding only what the filter accepts.

        Tests are written as ``{metric: test}`` where a test is a number
        (``>=``), a ``(operator, value)`` pair such as ``(">", 0.25)``, or a
        ``(low, high)`` range. Several metrics are combined with AND, and a
        metric that is NaN never passes.

            fast = graph.keep(edge_filter={"speed": (">", 0.25)})
            busy = graph.keep(node_filter={"orders": 5_000})
            peak = graph.keep(edge_filter={"orders": 500}, hour=[8, 9, 10])

        The original graph is untouched and tensors are shared, so this is
        cheap. ``drop_isolated`` removes clusters left with no routes.
        """
        out = RouteGraph(self.schema, directed=self.graph.is_directed(),
                         allow_self_loops=self.allow_self_loops)
        out.graph = self.graph.copy()
        out.insertion_log = list(self.insertion_log)
        out._apply_filter(edge_filter, node_filter, "keep", node_direction,
                          drop_isolated, grain_values)
        return out

    def cut(self, edge_filter=None, node_filter=None, node_direction="both",
            drop_isolated=True, **grain_values) -> "RouteGraph":
        """Remove what the filter MATCHES from this graph, in place.

        The mirror of :meth:`keep`: ``keep`` says what to hold on to, ``cut``
        says what to throw away.

            graph.cut(edge_filter={"orders": ("<", 100)})   # drop thin routes
            graph.cut(node_filter={"row_count": ("<", 5)})  # drop sparse clusters

        Returns the same graph so calls can be chained.
        """
        self._apply_filter(edge_filter, node_filter, "cut", node_direction,
                           drop_isolated, grain_values)
        return self

    def _apply_filter(self, edge_filter, node_filter, mode, node_direction,
                      drop_isolated, grain_values):
        if not edge_filter and not node_filter:
            raise ValueError("Pass edge_filter=... and/or node_filter=...")
        for spec in (edge_filter, node_filter):
            for metric in (spec or {}):
                if metric != "count":
                    self.schema.validate_metric(metric)

        graph = self.graph
        before = (graph.number_of_nodes(), graph.number_of_edges())

        if node_filter:
            matched = set(self.matching_clusters(node_filter, direction=node_direction,
                                                 **grain_values))
            graph.remove_nodes_from(set(graph.nodes) - matched if mode == "keep" else matched)

        if edge_filter:
            matched = {(u, v) for u, v, data in graph.edges(data=True)
                       if self._passes(data, edge_filter, grain_values)}
            everything = set(graph.edges())
            graph.remove_edges_from(everything - matched if mode == "keep" else matched)

        if drop_isolated:
            graph.remove_nodes_from([n for n in list(graph.nodes) if graph.degree(n) == 0])

        self.route_count = graph.number_of_edges()
        self.insertion_log.append({
            "action": f"{mode}_filter",
            "edge_filter": dict(edge_filter or {}),
            "node_filter": dict(node_filter or {}),
            "grain": dict(grain_values),
            "clusters_removed": before[0] - graph.number_of_nodes(),
            "routes_removed": before[1] - graph.number_of_edges(),
        })

    # ---------------- reachability ---------------- #
    def reach_to(self, target, budget=None, cost=None, max_hops=3, metrics=(),
                 **kwargs):
        """Clusters that can REACH ``target``, cheapest qualifying path each.

        The everyday question this answers is *where can orders come from and
        still arrive within two hours*::

            graph.reach_to("A1", budget=120)            # minutes, if that is your unit
            graph.reach_to("A1", budget=120, max_hops=2, edge_filter={"orders": 100})

        ``cost`` defaults to the schema ride-time metric and is summed along
        the path. Returns a DataFrame with the cost, the hop count and the
        route taken, cheapest first.
        """
        from .route_paths import reach_frame

        return reach_frame(self, target, direction="in", cost=cost, budget=budget,
                           max_hops=max_hops, metrics=metrics, **kwargs)

    def reach_from(self, source, budget=None, cost=None, max_hops=3, metrics=(),
                   **kwargs):
        """Clusters reachable FROM ``source`` within the same constraints."""
        from .route_paths import reach_frame

        return reach_frame(self, source, direction="out", cost=cost, budget=budget,
                           max_hops=max_hops, metrics=metrics, **kwargs)

    def paths(self, source, target, budget=None, cost=None, max_hops=3,
              metrics=(), **kwargs):
        """Every qualifying route from ``source`` to ``target``, not just the best."""
        from .route_paths import paths

        return paths(self, source, target, cost=cost, budget=budget,
                     max_hops=max_hops, metrics=metrics, **kwargs)

    def path_total(self, path, metric=None, **grain_values) -> float:
        """Sum one metric along a path, e.g. the total ride time of a route."""
        from .route_paths import path_total

        return path_total(self, path, self.resolve_metric(metric), **grain_values)

    def reach_subgraph(self, start, direction="out", budget=None, cost=None,
                       max_hops=3, **kwargs):
        """The cheapest-path tree as a DiGraph, ready to plot."""
        from .route_paths import reach_subgraph

        return reach_subgraph(self, start, direction=direction, cost=cost,
                              budget=budget, max_hops=max_hops, **kwargs)

    def plot_reach(self, start, **kwargs):
        """Draw the reachable set, laid out by hops from the start."""
        from .route_viz import plot_reach

        return plot_reach(self, start, **kwargs)

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
                      exclude_self_loops=True, edge_filter=None, node_filter=None,
                      node_direction="both", rank_by="edge", mirror=True,
                      **grain_values):
        """Expand around ``focus`` and keep only the top partners at each hop.

        ``upstream`` walks against the arrows (where orders come from) and
        ``downstream`` walks along them (where orders go). Both accept an int
        for a single level or a list for one top-k per level, so
        ``upstream=[5, 3, 2]`` keeps the top 5 sources of the focus, the top 3
        sources of each of those, then the top 2 of each of those.

        ``rank_by`` decides what the top-k measures at each hop: ``"edge"``
        keeps the biggest routes, ``"node"`` keeps the biggest clusters.

        ``mirror`` (on by default) splits a cluster that is reached on both
        sides into two nodes, one upstream and one downstream, so the picture
        stays left-to-right. Without it the cluster is claimed by whichever
        side found it first and the other side's routes point backwards across
        the figure. A cluster reached twice on the *same* side is still one
        node - only the two directions are separated. Split nodes are keyed
        ``(side, cluster)`` and carry the plain id in their ``cluster``
        attribute; everything else keeps the cluster id as its key.

        ``edge_filter`` / ``node_filter`` apply :meth:`keep` before walking, so
        the expansion never crosses a route that fails the filter and the
        top-k at each hop is chosen from the survivors. This is a different
        thing from the top-k itself: the filter is an absolute test on any
        metric, the top-k is a ranking on one.

        Returns a ``nx.DiGraph`` whose nodes carry ``level`` (0 for focus,
        negative upstream, positive downstream) and whose edges carry ``value``.
        """
        metric = self.resolve_metric(metric)
        metric = "count" if metric == "count" else self.schema.validate_metric(metric)
        focus_nodes = [f for f in _as_list(focus)]
        missing = [f for f in focus_nodes if f not in self.graph]
        if missing:
            raise KeyError(f"Unknown cluster(s): {missing}")

        source = self
        if edge_filter or node_filter:
            source = self.keep(edge_filter=edge_filter, node_filter=node_filter,
                               node_direction=node_direction, drop_isolated=False,
                               **grain_values)
            gone = [f for f in focus_nodes if f not in source.graph]
            if gone:
                raise ValueError(
                    f"The filter removed the focus cluster(s) {gone}; "
                    "loosen it or pick another focus"
                )

        sub = nx.DiGraph()
        focus_key = {f: (("focus", f) if mirror else f) for f in focus_nodes}
        for node in focus_nodes:
            sub.add_node(focus_key[node], cluster=node, level=0, depth=0,
                         side="focus", value=np.nan, is_focus=True)

        def key_for(cluster, side):
            """Node key: split by side only when mirroring, never for a focus."""
            if cluster in focus_key:
                return focus_key[cluster]
            return (side, cluster) if mirror else cluster

        def expand(direction, plan, sign):
            frontier = [(focus_key[f], f) for f in focus_nodes]
            side = "source" if sign < 0 else "drop"
            for hop, keep in enumerate(plan, start=1):
                if not frontier:
                    break
                picks = []
                if per_parent:
                    for parent_key, parent in frontier:
                        picks += [(parent_key, parent, p, v) for p, v in source.partners(
                            parent, direction=direction, metric=metric, top=keep,
                            min_value=min_value, rank_by=rank_by,
                            node_direction=node_direction, **grain_values)]
                else:
                    pool = []
                    for parent_key, parent in frontier:
                        pool += [(parent_key, parent, p, v) for p, v in source.partners(
                            parent, direction=direction, metric=metric,
                            min_value=min_value, rank_by=rank_by,
                            node_direction=node_direction, **grain_values)]
                    pool.sort(key=lambda item: -np.inf if np.isnan(item[3]) else item[3],
                              reverse=True)
                    picks = pool[:keep] if keep else pool

                next_frontier = []
                focus_set = set(focus_nodes)
                for parent_key, parent, partner, value in picks:
                    if exclude_self_loops and partner == parent:
                        continue
                    if hop > 1 and partner in focus_set:
                        # looping back to the focus from a deeper hop would draw
                        # a route already shown, pointing the wrong way
                        continue
                    partner_key = key_for(partner, side)
                    if partner_key not in sub:
                        sub.add_node(partner_key, cluster=partner, level=sign * hop,
                                     depth=hop, side=side, value=0.0, is_focus=False)
                        next_frontier.append((partner_key, partner))
                    if not np.isnan(value):
                        node = sub.nodes[partner_key]
                        node["value"] = float(np.nansum([node.get("value", 0.0), value]))
                    src, dst = ((parent_key, partner_key) if sign > 0
                                else (partner_key, parent_key))
                    u, v = (parent, partner) if sign > 0 else (partner, parent)
                    edge_value = (value if rank_by == "edge"
                                  else source.edge_value(u, v, metric, **grain_values))
                    sub.add_edge(src, dst, value=edge_value, metric=metric,
                                 rank_value=value, depth=hop, side=side,
                                 routes=source.graph[u][v].get("count", 1))
                frontier = next_frontier

        expand("in", self._level_plan(upstream), -1)
        expand("out", self._level_plan(downstream), 1)

        if include_cross_edges:
            selected = [(k, sub.nodes[k]["cluster"]) for k in sub.nodes]
            for u_key, u in selected:
                for v_key, v in selected:
                    if u == v or sub.has_edge(u_key, v_key):
                        continue
                    if not source.graph.has_edge(u, v):
                        continue
                    sub.add_edge(u_key, v_key, side="cross", depth=None, metric=metric,
                                 value=source.edge_value(u, v, metric, **grain_values),
                                 routes=source.graph[u][v].get("count", 1))

        if mirror:
            # only a cluster that really is on both sides stays split
            seen: dict = {}
            for node, data in sub.nodes(data=True):
                seen.setdefault(data["cluster"], []).append(node)
            plain = {keys[0]: cluster for cluster, keys in seen.items() if len(keys) == 1}
            if plain:
                sub = nx.relabel_nodes(sub, plain, copy=True)

        sub.graph.update({
            "focus": [focus_key[f] if focus_key[f] in sub else f for f in focus_nodes],
            "focus_clusters": focus_nodes,
            "mirror": mirror,
            "metric": metric,
            "grain": dict(grain_values),
            "upstream": self._level_plan(upstream),
            "downstream": self._level_plan(downstream),
            "edge_filter": dict(edge_filter or {}),
            "node_filter": dict(node_filter or {}),
            "rank_by": rank_by,
        })
        return sub

    def _split_kwargs(self, kwargs):
        """Separate grain filters from plain options in a mixed kwargs dict."""
        grain = {k: v for k, v in kwargs.items() if k in self.schema.grain_names}
        options = {k: v for k, v in kwargs.items() if k not in grain}
        return grain, options

    # ---------------- plotting ---------------- #

    def flow_frame(self, focus, upstream=5, downstream=5, metric=None,
                   extra_metrics=(), **kwargs):
        """The same expansion as :meth:`flow_subgraph`, as a tidy DataFrame."""
        metric = self.resolve_metric(metric)
        grain, options = self._split_kwargs(kwargs)
        sub = self.flow_subgraph(focus, upstream=upstream, downstream=downstream,
                                 metric=metric, **options, **grain)
        rows = []
        for u, v, data in sub.edges(data=True):
            pickup = sub.nodes[u].get("cluster", u)
            drop = sub.nodes[v].get("cluster", v)
            row = {
                "side": data["side"],
                "hop": data["depth"],
                "pickup_cluster": pickup,
                "drop_cluster": drop,
                metric: data["value"],
                "routes": data["routes"],
            }
            for extra in extra_metrics:
                row[extra] = self.edge_value(pickup, drop, extra, **grain)
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
                partner_key = v if side == "drop" else u
                partner = sub.nodes[partner_key].get("cluster", partner_key)
                if partner_key not in seen:
                    partners.append((partner_key, partner, data["value"]))
            partners.sort(key=lambda item: -np.inf if np.isnan(item[2]) else item[2], reverse=True)
            for index, (partner_key, partner, value) in enumerate(partners):
                last = index == len(partners) - 1
                arrow = "->" if side == "drop" else "<-"
                branch = "+-- " if last else "|-- "
                lines.append(f"{prefix}{branch}{arrow} {partner}  [{value_format.format(value)}]")
                walk(partner_key, side, prefix + ("    " if last else "|   "),
                     seen | {partner_key})

        for node in sub.graph["focus"]:
            grain_note = f"  {sub.graph['grain']}" if sub.graph["grain"] else ""
            lines.append(f"{sub.nodes[node].get('cluster', node)}  ({metric}{grain_note})")
            if sub.graph["upstream"]:
                lines.append("  sources (orders coming in)")
                walk(node, "source", "  ", {node})
            if sub.graph["downstream"]:
                lines.append("  drops (orders going out)")
                walk(node, "drop", "  ", {node})
        return "\n".join(lines)

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
