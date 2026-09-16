
from __future__ import annotations

from typing import Iterable, Sequence

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "Hex metric graphs require pandas and numpy: "
        "pip install 'sameer-graph-lib[route]'"
    ) from exc

from ._h3 import cell_to_latlng, get_resolution, grid_distance, is_valid_cell
from .affinity_graph import AffinityGraph
from .route_graph import (FLAT_GRAIN, RouteSchema, RouteTensor,
                          classify_metrics)

class HexMetricGraph:

    def __init__(
        self,
        data=None,
        *,
        hex_col: str = "hexid",
        metrics: Sequence[str] | None = None,
        grain_cols: Sequence[str] = (),
        schema: RouteSchema | None = None,
        value_metric: str | None = None,
        topology: str = "attach",
        ring: int = 1,
        resolution: int | None = None,
        **schema_kwargs,
    ):
        if topology not in ("attach", "adjacent", "none"):
            raise ValueError("topology must be one of: attach, adjacent, none")

        self.hex_col = hex_col
        self.topology = topology
        self.ring = int(ring)
        self.tensors: dict[str, RouteTensor] = {}
        self.row_counts: dict[str, int] = {}

        if data is None:
            raise ValueError("Pass a DataFrame of H3 cells")
        frame, grain_cols = self._prepare(data, hex_col, grain_cols)

        self.grain_cols = tuple(grain_cols)
        if metrics is not None and schema is not None:
            raise TypeError("Pass either schema= or metrics=, not both")
        self.metric_cols = list(metrics) if metrics is not None else None
        self.schema = schema or RouteSchema.from_frame(
            frame, grain_cols=self.grain_cols, metrics=metrics,
            exclude_cols=schema_kwargs.pop("exclude_cols", (hex_col,)),
            **schema_kwargs,
        )
        self.value_metric = value_metric or self.schema.default_metric

        cells = list(dict.fromkeys(frame[hex_col]))
        invalid = [c for c in cells if not is_valid_cell(str(c))]
        if invalid:
            raise ValueError(
                f"{len(invalid)} value(s) in {hex_col!r} are not H3 cells, "
                f"e.g. {invalid[:3]}"
            )
        self.resolution = int(resolution if resolution is not None
                              else get_resolution(str(cells[0])))

        for cell, rows in frame.groupby(hex_col, sort=False):
            self.tensors[cell] = RouteTensor.from_frame(self.schema, rows)
            self.row_counts[cell] = len(rows)

        self.graph = AffinityGraph(hex_resolution=self.resolution)
        self._build_topology(cells)

    @staticmethod
    def _prepare(data, hex_col, grain_cols):
        if hex_col not in data.columns:
            raise KeyError(f"Column {hex_col!r} is not in the frame")
        frame = data.dropna(subset=[hex_col]).copy()
        frame[hex_col] = frame[hex_col].astype(str).str.strip()
        frame = frame[frame[hex_col] != ""]
        if frame.empty:
            raise ValueError(f"No usable H3 cells in {hex_col!r}")

        grain_cols = tuple(grain_cols or ())
        if not grain_cols or grain_cols == (FLAT_GRAIN,):
            frame[FLAT_GRAIN] = "all"
            grain_cols = (FLAT_GRAIN,)
        missing = [c for c in grain_cols if c not in frame.columns]
        if missing:
            raise KeyError(f"Grain column(s) not in the frame: {missing}")
        return frame, grain_cols

    def _insert_cell(self, cell):
        if cell in self.graph.graph:
            return
        if self.topology == "attach":
            self.graph.add_hex(cell)
            return
        self.graph.graph.add_node(cell, count=0, value=0.0)
        if self.topology == "adjacent":
            for other in list(self.graph.graph.nodes):
                if other == cell:
                    continue
                steps = grid_distance(cell, other)
                if 0 < steps <= self.ring:
                    self.graph._add_or_update_edge(
                        cell, other, steps, count_increment=0, kind="adjacent")

    def _refresh(self, cells):
        for cell in cells:
            self.graph.set_hex_metric(
                cell,
                count=self.row_counts.get(cell, 0),
                value=self._safe(self.value(cell, self.value_metric)),
            )

    def _build_topology(self, cells):
        for cell in cells:
            self._insert_cell(cell)
        self.graph.node_add_count = 0
        self.graph.total_value_sum = 0.0
        self._refresh(cells)

    def add_frame(self, data) -> list:
        frame, _ = self._prepare(data, self.hex_col, self.grain_cols)
        invalid = [c for c in frame[self.hex_col].unique() if not is_valid_cell(str(c))]
        if invalid:
            raise ValueError(f"Not H3 cells: {invalid[:3]}")

        touched, added = [], []
        for cell, rows in frame.groupby(self.hex_col, sort=False):
            if cell in self.tensors:
                self.tensors[cell].add_rows(rows)
            else:
                self.tensors[cell] = RouteTensor.from_frame(self.schema, rows)
                added.append(cell)
            self.row_counts[cell] = self.row_counts.get(cell, 0) + len(rows)
            touched.append(cell)

        for cell in added:
            self._insert_cell(cell)
        self._refresh(touched)
        return added

    def add_cell(self, cell, **metric_values) -> "HexMetricGraph":
        row = {self.hex_col: cell}
        row.update(metric_values)
        for grain in self.grain_cols:
            row.setdefault(grain, "all" if grain == FLAT_GRAIN
                           else self.schema.grains[grain][0])
        self.add_frame(pd.DataFrame([row]))
        return self

    @classmethod
    def from_frame(cls, data, **kwargs) -> "HexMetricGraph":
        return cls(data, **kwargs)

    @classmethod
    def per_group(cls, data, group_col: str, **kwargs) -> dict:
        if group_col not in data.columns:
            raise KeyError(f"Column {group_col!r} is not in the frame")
        out = {}
        for key, rows in data.groupby(group_col, sort=False):
            try:
                out[key] = cls(rows, **kwargs)
            except ValueError:
                out[key] = None
        return out

    @staticmethod
    def _safe(value) -> float:
        value = float(value)
        return 0.0 if np.isnan(value) else value

    @property
    def cells(self) -> list:
        return list(self.graph.graph.nodes)

    @property
    def metrics(self) -> list:
        return list(self.schema.metric_names)

    def __len__(self) -> int:
        return self.graph.graph.number_of_nodes()

    def __contains__(self, cell) -> bool:
        return cell in self.graph.graph

    def __repr__(self) -> str:
        return (f"HexMetricGraph(cells={len(self)}, "
                f"edges={self.graph.graph.number_of_edges()}, "
                f"res={self.resolution}, topology={self.topology!r}, "
                f"value_metric={self.value_metric!r})")

    def tensor(self, cell) -> RouteTensor:
        try:
            return self.tensors[cell]
        except KeyError:
            raise KeyError(f"Unknown cell: {cell!r}") from None

    def summary(self, cell, **grain_values) -> dict:
        return self.tensor(cell).summary(**grain_values)

    def value(self, cell, metric=None, **grain_values) -> float:
        metric = metric or self.schema.default_metric
        resolved = self.schema.validate_metric(metric)
        return float(self.summary(cell, **grain_values)[resolved])

    def total_summary(self, **grain_values) -> dict:
        total = RouteTensor(self.schema)
        for tensor in self.tensors.values():
            total.merge(tensor)
        return total.summary(**grain_values)

    def profile(self, cell, metric=None, **kwargs):
        return self.tensor(cell).profile(metric or self.schema.default_metric, **kwargs)

    def frame(self, metrics=None, with_position: bool = True, **grain_values):
        wanted = list(metrics) if metrics else self.metrics
        rows = []
        for cell in self.cells:
            summary = self.summary(cell, **grain_values)
            row = {"hex": cell, "rows": self.row_counts.get(cell, 0),
                   "degree": self.graph.graph.degree(cell)}
            row.update({metric: summary[metric] for metric in wanted})
            if with_position:
                lat, lng = cell_to_latlng(cell)
                row["lat"], row["lng"] = lat, lng
            rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(self.value_metric, ascending=False,
                                 na_position="last", ignore_index=True)

    def top_cells(self, metric=None, top=10, **grain_values):
        metric = metric or self.value_metric
        frame = self.frame(metrics=[self.schema.validate_metric(metric)],
                           with_position=False, **grain_values)
        if frame.empty:
            return frame
        return frame.sort_values(metric, ascending=False, na_position="last",
                                 ignore_index=True).head(top)

    def neighbors(self, cell) -> list:
        if cell not in self.graph.graph:
            raise KeyError(f"Unknown cell: {cell!r}")
        return list(self.graph.graph.neighbors(cell))

    def distance(self, start, end) -> int:
        return grid_distance(start, end)

    def corridor(self, cutoff: float = 0.8, use_values: bool = True) -> list:
        return self.graph.get_appropriate_hexes(cutoff=cutoff, use_values=use_values)

    def corridor_stats(self, cutoff: float = 0.8, use_values: bool = True) -> dict:
        return self.graph.get_appropriate_hexes_with_stats(cutoff=cutoff,
                                                           use_values=use_values)

    def get_graph_stats(self) -> dict:
        stats = self.graph.get_graph_stats()
        stats.update({
            "resolution": self.resolution,
            "topology": self.topology,
            "value_metric": self.value_metric,
            "grains": {k: len(v) for k, v in self.schema.grains.items()},
        })
        return stats

    def describe(self) -> str:
        stats = self.get_graph_stats()
        grains = ", ".join(f"{k} ({v})" for k, v in stats["grains"].items())
        return "\n".join([
            f"cells         : {stats['num_nodes']}",
            f"edges         : {stats['num_edges']} ({self.topology})",
            f"resolution    : {self.resolution}",
            f"rows folded in: {sum(self.row_counts.values()):,}",
            f"grains        : {grains}",
            f"metrics       : {len(self.metrics)}, value metric {self.value_metric!r}",
        ])

    def plot_cells(self, metric=None, **kwargs):
        from .hex_metric_viz import plot_hex_metric

        return plot_hex_metric(self, metric=metric, **kwargs)

    def plot_steps(self, cells=None, **kwargs):
        from .hex_metric_viz import plot_hex_steps

        return plot_hex_steps(self, cells=cells, **kwargs)

    def plot_graph(self, **kwargs):
        return self.graph.visualize_graph(**kwargs)

    def plot_profile(self, cell, metric=None, **kwargs):
        metric = metric or self.schema.default_metric
        return _profile_figure(self.profile(cell, metric=metric), metric, cell, **kwargs)


def _profile_figure(table, metric, cell, figsize=(14, 4.5), cmap="Blues",
                    annotate=True, value_format="{:,.0f}", ax=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib: pip install 'sameer-graph-lib[plot]'"
        ) from exc

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    values = np.asarray(table.to_numpy(), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    mesh = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(values.shape[1]), [str(c) for c in table.columns], fontsize=8)
    ax.set_yticks(range(values.shape[0]),
                  [str(i) for i in getattr(table, "index", [""])], fontsize=8)
    fig.colorbar(mesh, ax=ax, label=metric, shrink=0.85)
    if annotate and values.size <= 24 * 8:
        finite = values[np.isfinite(values)]
        middle = (finite.max() + finite.min()) / 2 if finite.size else 0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(col, row, value_format.format(value), ha="center",
                            va="center", fontsize=6,
                            color="white" if value > middle else "#1F2933")
    ax.set_title(f"{metric}  |  {cell}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
