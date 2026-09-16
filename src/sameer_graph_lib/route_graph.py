
from __future__ import annotations

import operator
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import networkx as nx

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "Route graph features require pandas and numpy: "
        "pip install 'sameer-graph-lib[route]'"
    ) from exc


SUM_METRICS = [
    "requests", "orders", "accepted_requests", "accepted_orders",
    "completed_orders", "cancelled_orders", "unfulfilled", "supply_count",
]

MEAN_METRICS = [
    "avg_distance", "avg_duration", "avg_price", "avg_wait_time",
    "avg_rating", "accept_rate", "fulfil_rate", "surge_pct",
]

CATEGORICAL = ["segment"]

ID_COLS = ("pickup_cluster", "drop_cluster", "city")

MEAN_WORDS = frozenset({
    "avg", "average", "mean", "median", "rate", "rates", "ratio", "pct",
    "percent", "percentage", "share", "score", "index", "per", "prc",
})

DIMENSION_WORDS = frozenset({
    "hour", "hours", "hr", "hrs", "day", "days", "dow", "weekday", "week",
    "weeks", "month", "months", "quarter", "year", "years", "date", "datetime",
    "slot", "slots", "timeslot", "period", "bucket", "bin", "band", "tier",
    "cohort", "variant", "arm", "segment", "shift", "wave",
})


def _words(name) -> list:
    import re

    return [part for part in re.split(r"[^a-z0-9]+", str(name).lower()) if part]


def looks_like_dimension(name) -> bool:
    return any(word in DIMENSION_WORDS for word in _words(name))

FLAT_GRAIN = "__all__"

ROWS = "__rows"

DERIVED_METRICS = ("length", "ride_time", "speed", "row_count", "count")

DERIVED_OPS = {
    "/": lambda a, b: np.divide(a, b, out=np.full(np.shape(np.asarray(b, dtype=float)),
                                                  np.nan, dtype=float),
                                where=(np.asarray(b, dtype=float) != 0)),
    "*": lambda a, b: np.asarray(a, dtype=float) * np.asarray(b, dtype=float),
    "+": lambda a, b: np.asarray(a, dtype=float) + np.asarray(b, dtype=float),
    "-": lambda a, b: np.asarray(a, dtype=float) - np.asarray(b, dtype=float),
}


def _as_derived(spec):
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


COMPARISONS = {
    ">": operator.gt, "gt": operator.gt,
    ">=": operator.ge, "ge": operator.ge, "min": operator.ge,
    "<": operator.lt, "lt": operator.lt,
    "<=": operator.le, "le": operator.le, "max": operator.le,
    "==": operator.eq, "eq": operator.eq,
    "!=": operator.ne, "ne": operator.ne,
}


def _as_test(spec):
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
            continue
        out.append(column)
    return out


def suggest_ids(df, exclude=()) -> list:
    skip = set(exclude)
    hints = ("cluster", "hex", "id", "zone", "cell", "node", "region")
    return [c for c in df.columns
            if c not in skip and any(h in str(c).lower() for h in hints)]


def _cluster_side_tensor(source, clusters, direction, grain_values):
    merged = RouteTensor(source.schema)
    seen = set()
    for cluster in clusters:
        for _, u, v, data in source.incident(cluster, direction):
            if (u, v) in seen:
                continue
            seen.add((u, v))
            merged.merge(data["tensor"])
    return merged


def _add_rest(sub, source, left_out, side, sign, hop, metric, label, grain_values):
    import collections

    grouped = collections.defaultdict(list)
    for entry in left_out:
        grouped[entry[0]].append(entry)

    for parent_key, entries in grouped.items():
        merged, routes = RouteGraph._merge_left_out(source, entries, sign, grain_values)
        summary = merged.summary(**grain_values)
        value = float(summary.get(metric, np.nan)) if metric != "count" else float(routes)
        clusters = [e[2] for e in entries]
        key = ("rest", side, parent_key)
        sub.add_node(key, cluster=f"{label} ({len(entries)})", level=sign * hop,
                     depth=hop, side=side, value=value, is_focus=False,
                     is_rest=True, metrics=summary, routes=routes,
                     partners=clusters)
        src, dst = (parent_key, key) if sign > 0 else (key, parent_key)
        sub.add_edge(src, dst, value=value, metric=metric, rank_value=value,
                     depth=hop, side=side, routes=routes, is_rest=True,
                     metrics=summary)


def classify_metrics(df, exclude=()) -> tuple[list, list]:
    skip = set(exclude)
    sums, means = [], []
    for column in df.columns:
        if column in skip or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        name = str(column).lower()
        words = set(_words(column))
        if (column in MEAN_METRICS or name.endswith("%")
                or words & MEAN_WORDS):
            means.append(column)
        else:
            sums.append(column)
    return sums, means


def _require_xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Dataset views require xarray: pip install 'sameer-graph-lib[route]'"
        ) from exc
    return xr


def _as_list(value) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, set, frozenset, range, np.ndarray, pd.Index, pd.Series)
    ):
        return [value]
    return list(value)


@dataclass
class RouteSchema:

    grains: dict
    sum_metrics: list = field(default_factory=lambda: list(SUM_METRICS))
    mean_metrics: list = field(default_factory=lambda: list(MEAN_METRICS))
    weight_col: str | None = "requests"
    length_metric: str | None = None
    ride_time_metric: str | None = None
    speed_scale: float = 1.0
    derived_metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.grains:
            raise ValueError("Schema needs at least one grain dimension")
        self.grains = {k: list(v) for k, v in self.grains.items()}
        self._indexers = {k: pd.Index(v) for k, v in self.grains.items()}
        for m in (self.length_metric, self.ride_time_metric):
            if m is not None and m not in self.mean_metrics:
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
                levels = [v.item() if hasattr(v, "item") else v
                          for v in df[col].dropna().unique().tolist()]
                grains[col] = sorted(levels)

        default_weight = cls.__dataclass_fields__["weight_col"].default
        if "weight_col" not in kwargs:
            kwargs["weight_col"] = (default_weight if default_weight in df.columns
                                    else None)
        elif kwargs["weight_col"] is not None and kwargs["weight_col"] not in df.columns:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            raise KeyError(
                f"weight_col={kwargs['weight_col']!r} is not in the frame, so the "
                f"averages would be unweighted. Numeric columns here: {numeric}"
            )

        if metrics is not None:
            chosen = list(dict.fromkeys(metrics))
            if not chosen:
                raise ValueError("metrics= is empty; pass at least one column")
            absent = [m for m in chosen if m not in df.columns]
            if absent:
                numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                raise KeyError(
                    f"Metric column(s) not in the frame: {absent}. "
                    f"Numeric columns here: {numeric}"
                )
            inferred_sums, inferred_means = classify_metrics(df[chosen])
            unusable = [m for m in chosen if m not in inferred_sums + inferred_means]
            if unusable:
                raise TypeError(
                    f"Metric column(s) are not numeric and cannot be measured: {unusable}"
                )
        else:
            skip = set(grain_cols) | set(exclude_cols) | set(CATEGORICAL) | set(ID_COLS)
            skip |= {c for c in df.columns if looks_like_dimension(c)}
            inferred_sums, inferred_means = classify_metrics(df, exclude=skip)

        kwargs.setdefault("sum_metrics", inferred_sums)
        mean = list(kwargs.get("mean_metrics", inferred_means))
        derived = kwargs.get("derived_metrics") or {}
        for key, conventional in (("length_metric", "avg_distance"),
                                  ("ride_time_metric", "avg_duration")):
            metric = kwargs.get(key)
            if metric is None:
                metric = conventional if conventional in df.columns else None
                kwargs[key] = metric
            if metric is not None and metric not in mean:
                mean.append(metric)
        kwargs["mean_metrics"] = mean
        kwargs["derived_metrics"] = derived

        deliberate = {kwargs.get("length_metric"), kwargs.get("ride_time_metric")}
        named = [m for key in ("sum_metrics", "mean_metrics") for m in kwargs.get(key) or []]
        unseen = [m for m in dict.fromkeys(named)
                  if m not in df.columns and m not in deliberate]
        if unseen:
            warnings.warn(
                f"Metric column(s) not in the frame, so they will be NaN: {unseen}",
                stacklevel=2,
            )
        return cls(grains=grains, **kwargs)

    @property
    def grain_names(self):
        return list(self.grains)

    @property
    def shape(self):
        return tuple(len(v) for v in self.grains.values())

    @property
    def has_speed(self) -> bool:
        return self.length_metric is not None and self.ride_time_metric is not None

    @property
    def is_flat(self) -> bool:
        return self.grain_names == [FLAT_GRAIN]

    def ensure_grain(self, df):
        if self.is_flat and FLAT_GRAIN not in df.columns:
            return df.assign(**{FLAT_GRAIN: "all"})
        return df

    @property
    def default_metric(self) -> str:
        if self.sum_metrics:
            return self.sum_metrics[0]
        if self.mean_metrics:
            return self.mean_metrics[0]
        return "row_count"

    @property
    def metric_names(self) -> list[str]:
        names = ["row_count", *self.sum_metrics, *self.mean_metrics]
        if self.has_speed:
            names.append("speed")
        return [*names, *self._derived]

    def apply_derived(self, values: dict) -> dict:
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
            numeric = values.to_numpy(dtype=float)
            safe = np.where(np.isfinite(numeric), numeric, -1.0)
            values = pd.Index(np.rint(safe).astype("int64"))
        return indexer.get_indexer(values)

    def row_weights(self, df):
        if not self.weight_col or self.weight_col not in df:
            return np.ones(len(df))
        w = pd.to_numeric(df[self.weight_col], errors="coerce").fillna(0).clip(lower=0)
        return w.where(w > 0, 1.0).to_numpy(float)

    def compatible(self, other):
        return (self.grains == other.grains
                and self.sum_metrics == other.sum_metrics
                and self.mean_metrics == other.mean_metrics)

    def resolve_metric(self, metric: str) -> str:
        for alias, column in (("length", self.length_metric),
                              ("ride_time", self.ride_time_metric)):
            if metric == alias:
                if column is None:
                    raise KeyError(
                        f"{alias!r} needs a column: pass "
                        f"{'length_metric' if alias == 'length' else 'ride_time_metric'}="
                        " when building the graph"
                    )
                return column
        return metric

    def validate_metric(self, metric: str) -> str:
        resolved = self.resolve_metric(metric)
        if resolved in self.metric_names or resolved in DERIVED_METRICS:
            return resolved
        raise KeyError(
            f"Unknown metric {metric!r}. Available: {', '.join(self.metric_names)}"
        )


class RouteTensor:

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
        return float((self.arrays[ROWS] > 0).mean())

    @property
    def row_total(self) -> float:
        return float(self.arrays[ROWS].sum())

    @property
    def is_empty(self) -> bool:
        return self.row_total == 0

    @property
    def raw(self):
        xr = _require_xarray()
        dims = self.schema.grain_names
        return xr.Dataset({k: (dims, v) for k, v in self.arrays.items()},
                          coords=self.schema.grains)

    @property
    def data(self):
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
        if s.has_speed:
            out["speed"] = self._speed(out[s.length_metric], out[s.ride_time_metric])
        s.apply_derived(out)
        return xr.Dataset({k: (dims, np.asarray(v)) for k, v in out.items()},
                          coords=s.grains)

    def get(self, **grain_values):
        return self.data.sel(**grain_values)

    def selector(self, grain_values):
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
        sel = self.selector(grain_values)
        if sel is None:
            return {k: float(v.sum()) for k, v in self.arrays.items()}
        return {k: float(v[sel].sum()) for k, v in self.arrays.items()}

    def summary(self, **grain_values):
        s = self.schema
        tot = self.totals(**grain_values)
        has = tot[ROWS] > 0
        out = {"row_count": tot[ROWS]}
        for m in s.sum_metrics:
            out[m] = tot[m] if has else np.nan
        for m in s.mean_metrics:
            w = tot[f"{m}__w"]
            out[m] = tot[f"{m}__wsum"] / w if w > 0 else np.nan
        if s.has_speed:
            out["speed"] = float(self._speed(out[s.length_metric], out[s.ride_time_metric]))
        s.apply_derived(out)
        return {k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else v)
                for k, v in out.items()}

    def frame(self, drop_empty: bool = True):
        df = self.data.to_dataframe().reset_index()
        return df[df["row_count"] > 0].reset_index(drop=True) if drop_empty else df

    def profile(self, metric: str = "speed", index=None, columns=None):
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
        lm, rt = np.asarray(lm, dtype=float), np.asarray(ride_time, dtype=float)
        spd = np.divide(lm, rt, out=np.full_like(rt, np.nan), where=(rt > 0) & ~np.isnan(lm))
        return self.schema.speed_scale * spd


class RouteGraph:

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

    SCHEMA_OPTIONS = ("metrics", "sum_metrics", "mean_metrics", "weight_col",
                      "length_metric", "ride_time_metric", "speed_scale",
                      "derived_metrics", "exclude_cols")

    @classmethod
    def from_dataframe(cls, df, schema=None, pickup_col="pickup_cluster",
                       drop_col="drop_cluster", grain_cols=(), **kwargs):
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
        if pickup_col == drop_col:
            raise ValueError(
                f"pickup_col and drop_col are both {pickup_col!r}; every route "
                "would be a self loop. Name the two ends of the route, or use "
                "HexMetricGraph for data with a single cluster column."
            )
        missing = [c for c in (pickup_col, drop_col) if c not in df.columns]
        if missing:
            raise KeyError(f"Route id column(s) not in the frame: {missing}")
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
        data["length"] = data["distance"] = data["weight"] = (
            summ[s.length_metric] if s.length_metric else np.nan)
        data["ride_time"] = summ[s.ride_time_metric] if s.ride_time_metric else np.nan
        data["speed"] = summ.get("speed", np.nan)

    def _validate_cluster(self, cluster):
        if cluster is None or (isinstance(cluster, float) and np.isnan(cluster)):
            raise ValueError(f"Invalid cluster id: {cluster!r}")

    def bucket(self, pickup, drop, **grain_values):
        return self.graph[pickup][drop]["tensor"].get(**grain_values)

    def edge_metrics(self, pickup, drop, **grain_values):
        data, s = self.graph[pickup][drop], self.schema
        if not grain_values:
            return {k: data[k] for k in ("length", "ride_time", "speed")}
        summ = data["tensor"].summary(**grain_values)
        return {"length": summ[s.length_metric] if s.length_metric else np.nan,
                "ride_time": summ[s.ride_time_metric] if s.ride_time_metric else np.nan,
                "speed": summ.get("speed", np.nan)}

    def edge_length(self, pickup, drop, **grain_values):
        return self.edge_metrics(pickup, drop, **grain_values)["length"]

    def edge_ride_time(self, pickup, drop, **grain_values):
        return self.edge_metrics(pickup, drop, **grain_values)["ride_time"]

    def edge_speed(self, pickup, drop, **grain_values):
        return self.edge_metrics(pickup, drop, **grain_values)["speed"]

    def speed_profile(self, pickup, drop):
        return self.bucket(pickup, drop)["speed"].to_pandas()

    def route_summary(self, pickup, drop, **grain_values):
        return self.graph[pickup][drop]["tensor"].summary(**grain_values)

    def node_tensor(self, cluster, direction="out"):
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
        key = self.schema.resolve_metric(by)

        def weight(u, v, d):
            val = d["tensor"].summary(**grain_values)[key] if grain_values else d[by]
            return None if val is None or np.isnan(val) else val

        path = nx.shortest_path(self.graph, source, target, weight=weight)
        total = sum(weight(u, v, self.graph[u][v]) for u, v in zip(path, path[1:]))
        return path, total

    def as_dataset(self):
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

    def metric_names(self) -> list[str]:
        return [*self.schema.metric_names, "count"]
    def resolve_metric(self, metric=None) -> str:
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
        metric = self.resolve_metric(metric)
        return self._value_from_edge(self._edge_data(pickup, drop), metric, grain_values)

    def incident(self, node, direction="out"):
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
        metric = self.resolve_metric(metric)
        return self.partners(node, direction="out", metric=metric, top=top,
                             rank_by=rank_by, **kwargs)

    def top_sources(self, node, top=5, metric=None, rank_by="edge", **kwargs):
        metric = self.resolve_metric(metric)
        return self.partners(node, direction="in", metric=metric, top=top,
                             rank_by=rank_by, **kwargs)

    def partners_frame(self, node, direction="out", metric=None, top=None,
                       extra_metrics=(), rank_by="edge", node_direction="both",
                       **grain_values):
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

    def clusters_summary(self, clusters, direction="out", **grain_values) -> dict:
        return _cluster_side_tensor(self, clusters, direction,
                                    grain_values).summary(**grain_values)

    def node_summary(self, node, direction="out", **grain_values) -> dict:
        return self.node_tensor(node, direction=direction).summary(**grain_values)

    def node_value(self, node, metric=None, direction="out", **grain_values) -> float:
        metric = self.resolve_metric(metric)
        if metric == "count":
            return float(sum(1 for _ in self.incident(node, direction)))
        resolved = self.schema.validate_metric(metric)
        return float(self.node_summary(node, direction=direction, **grain_values)[resolved])

    def nodes_frame(self, metric=None, **grain_values):
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
        lead = (self.schema.default_metric if self.schema.default_metric in wanted
                else wanted[0])
        return frame.sort_values(f"out_{lead}", ascending=False,
                                 na_position="last", ignore_index=True)

    @property
    def weight_col(self):
        return self.schema.weight_col

    def rank_routes(self, metric=None, top=None, ascending=False, **grain_values):
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
        total = RouteTensor(self.schema)
        for _, _, data in self.graph.edges(data=True):
            total.merge(data["tensor"])
        return total.summary(**grain_values)

    def get_graph_stats(self) -> dict:
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
        return [(u, v) for u, v, _ in self.graph.edges(data=True)]

    def has_route(self, pickup, drop) -> bool:
        return self.graph.has_edge(pickup, drop)

    def simple_graph(self, metric=None, **grain_values):
        metric = self.resolve_metric(metric)
        out = nx.DiGraph() if self.graph.is_directed() else nx.Graph()
        out.add_nodes_from(self.graph.nodes)
        for u, v, data in self.graph.edges(data=True):
            value = self._value_from_edge(data, metric, grain_values)
            out.add_edge(u, v, weight=value, **{metric: value, "routes": data.get("count", 1)})
        return out

    def subgraph(self, nodes) -> "RouteGraph":
        keep = set(nodes)
        out = RouteGraph(self.schema, directed=self.graph.is_directed(),
                         allow_self_loops=self.allow_self_loops)
        out.graph = self.graph.subgraph(keep).copy()
        out.route_count = out.graph.number_of_edges()
        return out


    def edge_passes(self, pickup, drop, edge_filter, **grain_values) -> bool:
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
        return [node for node in self.graph.nodes
                if self._node_passes(node, node_filter, direction, grain_values)]

    def keep(self, edge_filter=None, node_filter=None, node_direction="both",
             drop_isolated=True, **grain_values) -> "RouteGraph":
        out = RouteGraph(self.schema, directed=self.graph.is_directed(),
                         allow_self_loops=self.allow_self_loops)
        out.graph = self.graph.copy()
        out.insertion_log = list(self.insertion_log)
        out._apply_filter(edge_filter, node_filter, "keep", node_direction,
                          drop_isolated, grain_values)
        return out

    def cut(self, edge_filter=None, node_filter=None, node_direction="both",
            drop_isolated=True, **grain_values) -> "RouteGraph":
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

    def reach_to(self, target, budget=None, cost=None, max_hops=3, metrics=(),
                 **kwargs):
        from .route_paths import reach_frame

        return reach_frame(self, target, direction="in", cost=cost, budget=budget,
                           max_hops=max_hops, metrics=metrics, **kwargs)

    def reach_from(self, source, budget=None, cost=None, max_hops=3, metrics=(),
                   **kwargs):
        from .route_paths import reach_frame

        return reach_frame(self, source, direction="out", cost=cost, budget=budget,
                           max_hops=max_hops, metrics=metrics, **kwargs)

    def paths(self, source, target, budget=None, cost=None, max_hops=3,
              metrics=(), **kwargs):
        from .route_paths import paths

        return paths(self, source, target, cost=cost, budget=budget,
                     max_hops=max_hops, metrics=metrics, **kwargs)

    def path_total(self, path, metric=None, **grain_values) -> float:
        from .route_paths import path_total

        return path_total(self, path, self.resolve_metric(metric), **grain_values)

    def reach_subgraph(self, start, direction="out", budget=None, cost=None,
                       max_hops=3, **kwargs):
        from .route_paths import reach_subgraph

        return reach_subgraph(self, start, direction=direction, cost=cost,
                              budget=budget, max_hops=max_hops, **kwargs)

    def plot_reach(self, start, **kwargs):
        from .route_viz import plot_reach

        return plot_reach(self, start, **kwargs)

    @staticmethod
    def _level_plan(spec) -> list:
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

    @staticmethod
    def _merge_left_out(source, entries, sign, grain_values):
        merged = RouteTensor(source.schema)
        routes = 0
        for _, parent, partner, _ in entries:
            u, v = (parent, partner) if sign > 0 else (partner, parent)
            merged.merge(source.graph[u][v]["tensor"])
            routes += source.graph[u][v].get("count", 1)
        return merged, routes

    def flow_subgraph(self, focus, upstream=5, downstream=5, metric=None,
                      per_parent=True, min_value=None, include_cross_edges=False,
                      exclude_self_loops=True, edge_filter=None, node_filter=None,
                      node_direction="both", rank_by="edge", mirror=True,
                      rest=False, rest_label="rest", **grain_values):
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
            if cluster in focus_key:
                return focus_key[cluster]
            return (side, cluster) if mirror else cluster

        def expand(direction, plan, sign):
            frontier = [(focus_key[f], f) for f in focus_nodes]
            side = "source" if sign < 0 else "drop"
            for hop, keep in enumerate(plan, start=1):
                if not frontier:
                    break
                focus_set = set(focus_nodes)
                pool, skipped = [], []
                for parent_key, parent in frontier:
                    for partner, value in source.partners(
                            parent, direction=direction, metric=metric,
                            min_value=min_value, rank_by=rank_by,
                            node_direction=node_direction, **grain_values):
                        entry = (parent_key, parent, partner, value)
                        if ((exclude_self_loops and partner == parent)
                                or (hop > 1 and partner in focus_set)):
                            skipped.append(entry)
                        else:
                            pool.append(entry)

                if per_parent:
                    picks, seen = [], {}
                    for entry in pool:
                        seen[entry[0]] = seen.get(entry[0], 0) + 1
                        if keep is None or seen[entry[0]] <= keep:
                            picks.append(entry)
                else:
                    pool.sort(key=lambda item: -np.inf if np.isnan(item[3]) else item[3],
                              reverse=True)
                    picks = pool[:keep] if keep else pool
                left_out = [e for e in pool if e not in picks] + skipped

                next_frontier = []
                for parent_key, parent, partner, value in picks:
                    partner_key = key_for(partner, side)
                    if partner_key not in sub:
                        sub.add_node(partner_key, cluster=partner, level=sign * hop,
                                     depth=hop, side=side, value=0.0, is_focus=False)
                        next_frontier.append((partner_key, partner))
                    if not np.isnan(value) and partner_key != parent_key:
                        node = sub.nodes[partner_key]
                        node["value"] = float(np.nansum([node.get("value", 0.0), value]))
                    src, dst = ((parent_key, partner_key) if sign > 0
                                else (partner_key, parent_key))
                    u, v = (parent, partner) if sign > 0 else (partner, parent)
                    edge_value = (value if rank_by == "edge"
                                  else source.edge_value(u, v, metric, **grain_values))
                    routes = source.graph[u][v].get("count", 1)
                    sub.add_edge(src, dst, value=edge_value, metric=metric,
                                 rank_value=value, depth=hop, side=side,
                                 routes=routes)
                    if src == dst:
                        sub.nodes[src].setdefault("loops", {})[side] = {
                            "value": edge_value, "rank_value": value,
                            "depth": hop, "metric": metric, "routes": routes,
                        }

                if rest:
                    _add_rest(sub, source, left_out, side, sign, hop, metric,
                              rest_label, grain_values)
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
        grain = {k: v for k, v in kwargs.items() if k in self.schema.grain_names}
        options = {k: v for k, v in kwargs.items() if k not in grain}
        return grain, options


    def flow_frame(self, focus, upstream=5, downstream=5, metric=None,
                   extra_metrics=(), **kwargs):
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
        from .route_viz import plot_flow

        return plot_flow(self, focus, **kwargs)

    def plot_partners(self, node, **kwargs):
        from .route_viz import plot_partners

        return plot_partners(self, node, **kwargs)

    def plot_profile(self, pickup=None, drop=None, node=None, **kwargs):
        from .route_viz import plot_profile

        return plot_profile(self, pickup=pickup, drop=drop, node=node, **kwargs)

    def plot_matrix(self, **kwargs):
        from .route_viz import plot_matrix

        return plot_matrix(self, **kwargs)

    def plot_graph(self, **kwargs):
        from .route_viz import plot_route_graph

        return plot_route_graph(self, **kwargs)
