
from __future__ import annotations

from typing import Sequence

import numpy as np

SOURCE_COLOR = "#4C78A8"
DROP_COLOR = "#4E9F76"
FOCUS_COLOR = "#B3473F"
CROSS_COLOR = "#9AA5AD"

INK = "#1F2933"
MUTED_INK = "#7B8794"
HAIRLINE = "#CBD2D9"


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Route plotting requires matplotlib: pip install 'sameer-graph-lib[plot]'"
        ) from exc
    return plt


def _fade(color: str, depth: int, max_depth: int) -> str:
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(color))
    if max_depth <= 1 or depth <= 1:
        return mcolors.to_hex(rgb)
    t = 0.5 * (depth - 1) / max(max_depth - 1, 1)
    return mcolors.to_hex(rgb + (1.0 - rgb) * t)


def _tint(color: str, amount: float = 0.72) -> str:
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(rgb + (1.0 - rgb) * amount)


def _scale(values, low: float, high: float):
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, (low + high) / 2.0)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        out = np.full(arr.shape, (low + high) / 2.0)
    else:
        out = low + (arr - lo) / (hi - lo) * (high - low)
    return np.where(np.isfinite(out), out, low)


def _short(label, limit=None) -> str:
    text = str(label)
    if limit and len(text) > limit:
        return text[: max(limit - 3, 1)] + "..."
    return text


def _metric_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(m) for m in value]


def _auto_fmt(value: float) -> str:
    size = abs(value)
    if size >= 1000:
        return "{:,.0f}"
    if size >= 100:
        return "{:,.1f}"
    if size >= 1:
        return "{:,.2f}"
    return "{:,.3f}"


def _format(value, fmt=None, metric=None) -> str:
    if isinstance(fmt, dict):
        fmt = fmt.get(metric)
    if value is None:
        return "n/a"
    try:
        if np.isnan(value):
            return "n/a"
    except TypeError:
        return str(value)
    return (fmt or _auto_fmt(float(value))).format(value)


def metric_table(rows, key_width=0, value_width=0, header=None, show_names=True,
                 rule: bool = True) -> str:
    key_width = max([key_width] + [len(k) for k, _ in rows]) if rows else key_width
    value_width = max([value_width] + [len(v) for _, v in rows]) if rows else value_width
    body = [f"{k.ljust(key_width)}  {v.rjust(value_width)}" if show_names else v.rjust(value_width)
            for k, v in rows]
    width = max([len(line) for line in body] + [len(header or "")]) if (body or header) else 0
    lines = []
    if header is not None:
        lines.append(header.center(width))
        if rule and body:
            lines.append("-" * width)
    lines.extend(body)
    return "\n".join(lines)


def _column_widths(all_rows) -> tuple[int, int]:
    key_width = value_width = 0
    for rows in all_rows:
        for key, value in rows:
            key_width = max(key_width, len(key))
            value_width = max(value_width, len(value))
    return key_width, value_width


def _table_rows(values, metrics, value_format, name_len=None):
    return [(_short(metric, name_len), _format(value, value_format, metric))
            for metric, value in zip(metrics, values)]


def flow_layout(sub, level_gap: float = 2.6, spacing: float = 1.0) -> dict:
    levels: dict[int, list] = {}
    for node, data in sub.nodes(data=True):
        levels.setdefault(int(data.get("level", 0)), []).append(node)

    pos: dict = {}
    for level in sorted(levels, key=lambda lv: (abs(lv), lv)):
        nodes = levels[level]

        def barycenter(node):
            neighbours = set(sub.predecessors(node)) | set(sub.successors(node))
            placed = [pos[n][1] for n in neighbours if n in pos]
            return float(np.mean(placed)) if placed else 0.0

        def weight(node):
            value = sub.nodes[node].get("value")
            return -np.inf if value is None or (isinstance(value, float) and np.isnan(value)) else -value

        ordered = nodes if level == 0 else sorted(nodes, key=lambda n: (barycenter(n), weight(n), str(n)))
        count = len(ordered)
        for index, node in enumerate(ordered):
            pos[node] = (level * level_gap, (index - (count - 1) / 2.0) * spacing)
    return pos


def geo_layout(nodes) -> dict:
    from ._h3 import cell_to_latlng, is_valid_cell

    pos = {}
    for node in nodes:
        cell = str(node)
        if not is_valid_cell(cell):
            raise ValueError(f"layout='geo' needs H3 cell ids, got {node!r}")
        lat, lng = cell_to_latlng(cell)
        pos[node] = (lng, lat)
    return pos


def _auto_figsize(sub, node_metrics, edge_metrics, font_size,
                  limits=(12.0, 7.0, 40.0, 34.0)) -> tuple[float, float]:
    counts: dict[int, int] = {}
    for _, data in sub.nodes(data=True):
        level = int(data.get("level", 0))
        counts[level] = counts.get(level, 0) + 1
    levels = max(len(counts), 1)
    busiest = max(counts.values(), default=1)

    char = font_size * 0.62 / 72.0
    node_chars = max((len(m) for m in node_metrics), default=8) + 10
    edge_chars = max((len(m) for m in edge_metrics), default=8) + 10
    column = (node_chars + edge_chars) * char + 1.4
    row = len(node_metrics) * (font_size + 3) / 72.0 + 0.85

    min_w, min_h, max_w, max_h = limits
    if levels == 1:
        min_w = max(6.0, column * 1.5)      # one column needs no room for three
    return (min(max(min_w, levels * column), max_w),
            min(max(min_h, busiest * row + 2.0), max_h))


def plot_flow(
    graph,
    focus,
    upstream=5,
    downstream=5,
    metric: str | None = None,
    node_metrics=None,
    edge_metrics=None,
    size_metric: str | None = None,
    per_parent: bool = True,
    rank_by: str = "edge",
    node_direction: str = "out",
    mirror: bool = True,
    rest: bool = False,
    min_value: float | None = None,
    edge_filter=None,
    node_filter=None,
    include_cross_edges: bool = False,
    self_loops: str = "node",
    exclude_self_loops: bool | None = None,
    layout: str = "layered",
    level_gap: float = 2.6,
    spacing: float = 1.0,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax=None,
    show_edge_values: bool = True,
    show_node_values: bool = True,
    show_metric_names: bool = True,
    value_format=None,
    metric_name_len: int | None = None,
    node_size: tuple[float, float] = (450.0, 2600.0),
    edge_width: tuple[float, float] = (0.9, 6.0),
    source_color: str = SOURCE_COLOR,
    drop_color: str = DROP_COLOR,
    focus_color: str = FOCUS_COLOR,
    cross_color: str = CROSS_COLOR,
    label_len: int | None = None,
    font_size: int = 8,
    table_font_size: int | None = None,
    curve: float = 0.0,
    cross_curve: float | None = None,
    label_cross_edges: bool = False,
    legend: bool = True,
    annotate_levels: bool = True,
    node_metric=None,
    edge_metric=None,
    subgraph=None,
    node_values=None,
    level_label=None,
    legend_labels=None,
    **grain_values,
):
    plt = _require_matplotlib()
    import matplotlib.patches as mpatches
    import networkx as nx

    metric = graph.resolve_metric(metric)
    node_metric_names = _metric_list(node_metrics if node_metrics is not None else node_metric)
    edge_metric_names = _metric_list(edge_metrics if edge_metrics is not None else edge_metric)
    node_metric_names = node_metric_names or [metric]
    edge_metric_names = edge_metric_names or [metric]
    size_metric = size_metric or node_metric_names[0]
    table_font_size = table_font_size or max(font_size - 1, 5)

    sub = subgraph if subgraph is not None else graph.flow_subgraph(
        focus, upstream=upstream, downstream=downstream, metric=metric,
        per_parent=per_parent, rank_by=rank_by, min_value=min_value,
        edge_filter=edge_filter, node_filter=node_filter, mirror=mirror,
        rest=rest, include_cross_edges=include_cross_edges,
        self_loops=self_loops, exclude_self_loops=exclude_self_loops,
        **grain_values,
    )
    if sub.number_of_nodes() == 0:
        raise ValueError("Nothing to plot: the expansion selected no clusters")
    if sub.number_of_edges() == 0 and subgraph is None:
        import warnings

        warnings.warn(
            "No routes survived: the focus cluster is drawn on its own. "
            "Loosen edge_filter/node_filter/min_value, or raise the top-k.",
            stacklevel=2,
        )

    focus_nodes = sub.graph["focus"]
    if layout == "layered":
        pos = flow_layout(sub, level_gap=level_gap, spacing=spacing)
    elif layout == "geo":
        pos = {n: geo_layout([sub.nodes[n].get("cluster", n)])[sub.nodes[n].get("cluster", n)]
               for n in sub.nodes}
    elif layout == "spring":
        pos = nx.spring_layout(sub, seed=7)
    else:
        raise ValueError("layout must be one of: layered, geo, spring")

    if figsize is None:
        figsize = _auto_figsize(sub, node_metric_names, edge_metric_names, table_font_size)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    max_depth = max((abs(int(d.get("level", 0))) for _, d in sub.nodes(data=True)), default=1)

    nodes = list(sub.nodes)

    if node_direction not in ("in", "out", "both"):
        raise ValueError("node_direction must be 'in', 'out' or 'both'")

    def cluster_of(node):
        return sub.nodes[node].get("cluster", node)

    rest_cache: dict = {}

    def node_value(node, name):
        if node_values is not None and name in node_values.get(node, {}):
            return node_values[node][name]
        data = sub.nodes[node]
        if data.get("is_rest"):
            if name == "count":
                return float(data.get("routes", 0))
            if node not in rest_cache:
                rest_cache[node] = graph.clusters_summary(
                    data["partners"], direction=node_direction, **grain_values)
            return float(rest_cache[node].get(name, np.nan))
        if data.get("is_self"):
            if name == "count":
                return float(data.get("routes", 0))
            return float(data.get("metrics", {}).get(name, np.nan))
        if name == "count":
            return float(sub.degree(node))
        return graph.node_value(cluster_of(node), metric=name,
                                direction=node_direction, **grain_values)

    def edge_value(u, v, name):
        carried = sub[u][v].get("metrics")
        if carried is not None:
            return float(carried.get(name, np.nan)) if name != "count" else float(
                sub[u][v].get("routes", 0))
        return graph.edge_value(cluster_of(u), cluster_of(v), name, **grain_values)

    node_table_values = {node: [node_value(node, name) for name in node_metric_names]
                         for node in nodes}
    if size_metric in node_metric_names:
        sizes_from = [node_table_values[node][node_metric_names.index(size_metric)]
                      for node in nodes]
    else:
        sizes_from = [node_value(node, size_metric) for node in nodes]

    sizes = _scale(sizes_from, *node_size)
    borders, fills = [], []
    for node in nodes:
        data = sub.nodes[node]
        if data.get("is_focus"):
            border = focus_color
        elif data.get("is_self"):
            border = focus_color
        else:
            base = source_color if data.get("side") == "source" else drop_color
            border = _fade(base, int(data.get("depth", 1)), max_depth)
        borders.append(border)
        fills.append("white" if data.get("is_rest")
                     else _tint(border, 0.62 if data.get("is_focus") else 0.76))

    nx.draw_networkx_nodes(
        sub, pos, nodelist=nodes, node_color=fills, node_size=list(sizes),
        edgecolors=borders, linewidths=1.4, ax=ax,
    )

    edges = list(sub.edges(data=True))
    if edges:
        widths = _scale([d["value"] for _, _, d in edges], *edge_width)
        width_of = {(u, v): float(w) for (u, v, _), w in zip(edges, widths)}
        flow_edges = [(u, v, d) for u, v, d in edges if d.get("side") != "cross"]
        cross_edges = [(u, v, d) for u, v, d in edges if d.get("side") == "cross"]

        def draw(subset, colors, style, rad):
            if not subset:
                return
            nx.draw_networkx_edges(
                sub, pos, edgelist=[(u, v) for u, v, _ in subset],
                width=[width_of[(u, v)] for u, v, _ in subset],
                edge_color=colors, ax=ax, arrows=True, arrowsize=11,
                arrowstyle="-|>", connectionstyle=f"arc3,rad={rad}", style=style,
                node_size=list(sizes), alpha=0.8, min_source_margin=2, min_target_margin=6,
            )

        def same_level(u, v):
            return sub.nodes[u].get("level") == sub.nodes[v].get("level")

        loops = [(node, side, info) for node in nodes
                 for side, info in (sub.nodes[node].get("loops") or {}).items()]
        flow_edges = [(u, v, d) for u, v, d in flow_edges if u != v]
        straight = [(u, v, d) for u, v, d in flow_edges if not same_level(u, v)]
        sideways = [(u, v, d) for u, v, d in flow_edges if same_level(u, v)]
        for subset, rad in ((straight, curve),
                            (sideways, cross_curve if cross_curve is not None else 0.25)):
            draw(subset,
                 [focus_color if d.get("side") == "self" else
                  _fade(source_color if d.get("side") == "source" else drop_color,
                        int(d.get("depth") or 1), max_depth) for _, _, d in subset],
                 "solid", rad)
        draw(cross_edges, [cross_color] * len(cross_edges), "dashed",
             cross_curve if cross_curve is not None else 0.25)

        loop_labels = {}
        if loops:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            span_x = (max(xs) - min(xs)) or level_gap
            span_y = (max(ys) - min(ys)) or spacing
        for node, side, info in loops:
            colour = _fade(source_color if side == "source" else drop_color,
                           int(info.get("depth") or 1), max_depth)
            wide, tall = 0.11 * span_x, 0.11 * span_y
            x, y = pos[node]
            both = len(sub.nodes[node].get("loops") or {}) > 1
            shift = 0.0 if not both else (-0.62 if side == "source" else 0.62) * wide
            centre_x, centre_y = x + shift, y + tall * 0.75
            ax.add_patch(mpatches.Ellipse(
                (centre_x, centre_y), wide, tall, fill=False,
                lw=width_of.get((node, node), edge_width[0]),
                ec=colour, alpha=0.8, zorder=1))
            tip = 1 if side == "drop" else -1
            ax.annotate("", xy=(centre_x + tip * wide * 0.30, centre_y + tall * 0.40),
                        xytext=(centre_x + tip * wide * 0.12, centre_y + tall * 0.48),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=0,
                                        mutation_scale=13, alpha=0.9), zorder=2)
            lift = 0.10 if side == "source" or not both else 0.24
            loop_labels[(node, side)] = (centre_x,
                                         centre_y + tall * 0.5 + lift * span_y)

        if show_edge_values and loops:
            for node, side, _ in loops:
                rows = _table_rows(
                    [edge_value(node, node, name) for name in edge_metric_names],
                    edge_metric_names, value_format, metric_name_len)
                key_width, value_width = _column_widths([rows])
                x, y = loop_labels[(node, side)]
                ax.text(x, y, metric_table(rows, key_width, value_width,
                                           show_names=show_metric_names
                                           and len(edge_metric_names) > 1),
                        ha="center", va="center", fontsize=table_font_size,
                        family="monospace", color=INK, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.35", facecolor="white",
                                  edgecolor=HAIRLINE, linewidth=0.6, alpha=0.95))

        if show_edge_values:
            # the self node already carries the loop's numbers in its own table
            labelled = [(u, v, d) for u, v, d in
                        (edges if label_cross_edges else flow_edges)
                        if d.get("side") != "self"]
            edge_rows = {
                (u, v): _table_rows(
                    [edge_value(u, v, name) for name in edge_metric_names],
                    edge_metric_names, value_format, metric_name_len,
                )
                for u, v, _ in labelled
            }
            key_width, value_width = _column_widths(edge_rows.values())
            labels = {
                pair: metric_table(rows, key_width, value_width,
                                   show_names=show_metric_names and len(edge_metric_names) > 1)
                for pair, rows in edge_rows.items()
            }
            label_kwargs = dict(
                edge_labels=labels, font_size=table_font_size, font_family="monospace",
                font_color=INK, ax=ax, rotate=False,
                bbox=dict(boxstyle="square,pad=0.35", facecolor="white",
                          edgecolor=HAIRLINE, linewidth=0.6, alpha=0.95),
            )
            try:
                nx.draw_networkx_edge_labels(sub, pos, connectionstyle=f"arc3,rad={curve}",
                                             **label_kwargs)
            except TypeError:
                nx.draw_networkx_edge_labels(sub, pos, **label_kwargs)

    nx.draw_networkx_labels(sub, pos,
                            labels={n: _short(cluster_of(n), label_len) for n in nodes},
                            font_size=font_size, font_weight="bold", font_color=INK, ax=ax)
    if show_node_values:
        node_rows = {
            node: _table_rows(node_table_values[node], node_metric_names,
                              value_format, metric_name_len)
            for node in nodes
        }
        key_width, value_width = _column_widths(node_rows.values())
        for node, size in zip(nodes, sizes):
            radius = float(np.sqrt(max(size, 1.0) / np.pi))
            ax.annotate(
                metric_table(node_rows[node], key_width, value_width,
                             show_names=show_metric_names and len(node_metric_names) > 1),
                xy=pos[node], xycoords="data",
                xytext=(0, -(radius + 7)), textcoords="offset points",
                ha="center", va="top", fontsize=table_font_size, family="monospace",
                color=INK,
                bbox=dict(boxstyle="square,pad=0.35", facecolor="white",
                          edgecolor=HAIRLINE, linewidth=0.6, alpha=0.95),
            )

    if pos:
        # every node on one column leaves the x range to the edge curves, which
        # pins the column off to one side; centre it instead
        xs = [x for x, _ in pos.values()]
        if max(xs) - min(xs) < 1e-9:
            ax.set_xlim(xs[0] - level_gap * 0.8, xs[0] + level_gap * 0.8)

    if annotate_levels and layout == "layered" and pos:
        for level in sorted({int(d.get("level", 0)) for _, d in sub.nodes(data=True)}):
            if level_label is not None:
                text = level_label(level)
            elif level == 0:
                text = "FOCUS"
            elif level < 0:
                text = f"SOURCES  L{abs(level)}"
            else:
                text = f"DROPS  L{level}"
            ax.axvline(level * level_gap, color=HAIRLINE, linewidth=0.7,
                       linestyle=(0, (4, 4)), zorder=0)
            ax.annotate(text, xy=(level * level_gap, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(0, -10),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=font_size, color=MUTED_INK)

    if legend:
        focus_text, source_text, drop_text = legend_labels or (
            "focus cluster", "sources (orders come from)", "drops (orders go to)")
        handles = [
            mpatches.Patch(facecolor=_tint(focus_color), edgecolor=focus_color,
                           linewidth=1.2, label=focus_text),
        ]
        sides = {d.get("side") for _, d in sub.nodes(data=True)}
        if "source" in sides:
            handles.append(mpatches.Patch(facecolor=_tint(source_color),
                                          edgecolor=source_color, linewidth=1.2,
                                          label=source_text))
        if "drop" in sides:
            handles.append(mpatches.Patch(facecolor=_tint(drop_color),
                                          edgecolor=drop_color, linewidth=1.2,
                                          label=drop_text))
        if "self" in sides:
            handles.append(mpatches.Patch(facecolor=_tint(focus_color, 0.86),
                                          edgecolor=focus_color, linewidth=1.2,
                                          label="same cluster to itself"))
        if include_cross_edges:
            handles.append(mpatches.Patch(facecolor=_tint(cross_color),
                                          edgecolor=cross_color, linewidth=1.2,
                                          label="cross route"))
        legend_artist = fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                                   fontsize=font_size, frameon=False,
                                   bbox_to_anchor=(0.5, 0.0), handlelength=1.1,
                                   handleheight=1.1, columnspacing=2.2)
        for text in legend_artist.get_texts():
            text.set_color(MUTED_INK)

    if title is None:
        named = [str(sub.nodes[f].get("cluster", f)) for f in focus_nodes]
        bits = [f"Order flow around {', '.join(named)}",
                f"ranked by {metric}"]
        shown = [m for m in dict.fromkeys(node_metric_names + edge_metric_names) if m != metric]
        if shown:
            bits.append("showing " + ", ".join(shown))
        if sub.graph.get("rank_by") == "node":
            bits[1] = f"top clusters by {metric}"
        if sub.graph["grain"]:
            bits.append(", ".join(f"{k}={v}" for k, v in sub.graph["grain"].items()))
        applied = {**sub.graph.get("edge_filter", {}), **sub.graph.get("node_filter", {})}
        if applied:
            bits.append("filtered on " + ", ".join(applied))
        title = "  |  ".join(bits)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=20)
    ax.margins(x=0.14, y=0.18)
    ax.axis("off")
    fig.tight_layout(rect=(0, 0.04, 1, 1) if legend else None)
    return fig


def plot_reach(
    graph,
    start,
    direction: str = "in",
    budget=None,
    cost: str | None = None,
    max_hops: int = 3,
    min_hops: int = 1,
    edge_filter=None,
    node_filter=None,
    node_metrics=None,
    edge_metrics=None,
    size_metric: str | None = None,
    title: str | None = None,
    **kwargs,
):
    cost = cost or graph.schema.ride_time_metric or graph.schema.default_metric
    grain = {k: v for k, v in kwargs.items() if k in graph.schema.grain_names}

    tree = graph.reach_subgraph(start, direction=direction, budget=budget, cost=cost,
                                max_hops=max_hops, min_hops=min_hops,
                                edge_filter=edge_filter, node_filter=node_filter,
                                **grain)
    if tree.number_of_edges() == 0:
        import warnings

        warnings.warn(
            f"Nothing is within reach of {start!r} under those constraints; "
            "raise budget or max_hops.",
            stacklevel=2,
        )

    shown = _metric_list(node_metrics) or [cost, "hops"]
    size_metric = size_metric or graph.schema.default_metric
    kwargs.setdefault("value_format", {cost: "{:,.1f}", "hops": "{:,.0f}"})
    totals = {
        node: {cost: data.get("cost", 0.0), "hops": float(data.get("hops", 0))}
        for node, data in tree.nodes(data=True)
    }
    inward = tree.graph["direction"] == "in"

    def band(level):
        if level == 0:
            return "START"
        hops = abs(level)
        return f"{hops} HOP" if hops == 1 else f"{hops} HOPS"

    if title is None:
        verb = "reaching" if inward else "reachable from"
        bits = [f"Clusters {verb} {start}"]
        bits.append(f"{cost} within {budget:,.0f}" if budget is not None else f"by {cost}")
        bits.append(f"up to {max_hops} hops")
        if grain:
            bits.append(", ".join(f"{k}={v}" for k, v in grain.items()))
        title = "  |  ".join(bits)

    return plot_flow(
        graph,
        start,
        subgraph=tree,
        metric=cost,
        node_metrics=shown,
        edge_metrics=_metric_list(edge_metrics) or [cost],
        node_values=totals,
        size_metric=size_metric,
        level_label=band,
        legend_labels=(f"{start} (start)", "can reach it", "reachable from it"),
        title=title,
        **kwargs,
    )


def plot_partners(
    graph,
    node,
    top: int = 10,
    metric: str | None = None,
    shared_scale: bool = True,
    figsize: tuple[float, float] = (13, 7),
    value_format: str = "{:,.0f}",
    source_color: str = SOURCE_COLOR,
    drop_color: str = DROP_COLOR,
    label_len: int | None = None,
    title: str | None = None,
    **grain_values,
):
    plt = _require_matplotlib()

    metric = graph.resolve_metric(metric)
    sources = graph.top_sources(node, top=top, metric=metric, **grain_values)
    drops = graph.top_drops(node, top=top, metric=metric, **grain_values)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax, rows, color, heading, padding in (
        (axes[0], sources, source_color, "orders coming from", -6),
        (axes[1], drops, drop_color, "orders going to", 3),
    ):
        if not rows:
            ax.text(0.5, 0.5, "no routes", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(heading, fontsize=11, fontweight="bold")
            ax.axis("off")
            continue
        names = [_short(name, label_len) for name, _ in rows][::-1]
        values = [value for _, value in rows][::-1]
        bars = ax.barh(names, values, color=_tint(color, 0.45), edgecolor=color,
                       linewidth=1.0, height=0.68)
        ax.bar_label(bars, labels=[_format(v, value_format) for v in values],
                     padding=padding, fontsize=8, color=INK)
        ax.set_title(heading, fontsize=11, color=MUTED_INK)
        ax.set_xlabel(metric, color=MUTED_INK)
        ax.grid(True, axis="x", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(HAIRLINE)
        ax.margins(x=0.16)

    axes[0].invert_xaxis()
    axes[0].yaxis.tick_right()
    if shared_scale:
        limit = max([abs(v) for _, v in sources + drops if v == v] or [1.0]) * 1.2
        axes[0].set_xlim(limit, 0)
        axes[1].set_xlim(0, limit)
    if title is None:
        grain_note = ""
        if grain_values:
            grain_note = "  |  " + ", ".join(f"{k}={v}" for k, v in grain_values.items())
        title = f"Top {top} partners of {node}  |  {metric}{grain_note}"
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_profile(
    graph,
    pickup=None,
    drop=None,
    node=None,
    direction: str = "both",
    metric: str | None = None,
    index: str | None = None,
    columns: str | None = None,
    figsize: tuple[float, float] = (14, 5),
    cmap: str = "Blues",
    annotate: bool = True,
    value_format: str = "{:,.0f}",
    title: str | None = None,
    ax=None,
):
    plt = _require_matplotlib()

    metric = graph.resolve_metric(metric)
    if node is not None:
        tensor = graph.node_tensor(node, direction=direction)
        what = f"{node} ({direction})"
    elif pickup is not None and drop is not None:
        tensor = graph.graph[pickup][drop]["tensor"]
        what = f"{pickup} -> {drop}"
    else:
        raise ValueError("Pass either node=... or both pickup=... and drop=...")

    table = tensor.profile(metric=metric, index=index, columns=columns)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    values = np.asarray(table.to_numpy(), dtype=float)
    mesh = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns], fontsize=8)
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index], fontsize=8)
    ax.set_xlabel(table.columns.name or "")
    ax.set_ylabel(table.index.name or "")
    fig.colorbar(mesh, ax=ax, label=metric, shrink=0.85)

    if annotate and values.size <= 24 * 8:
        finite = values[np.isfinite(values)]
        midpoint = (finite.max() + finite.min()) / 2 if finite.size else 0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if not np.isfinite(value):
                    continue
                ax.text(col, row, _format(value, value_format), ha="center", va="center",
                        fontsize=6, color="white" if value > midpoint else INK)

    ax.set_title(title or f"{metric} profile  |  {what}", fontsize=12,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_matrix(
    graph,
    metric: str | None = None,
    nodes=None,
    top: int | None = 15,
    figsize: tuple[float, float] = (12, 10),
    cmap: str = "Blues",
    annotate: bool = True,
    value_format: str = "{:,.0f}",
    label_len: int | None = None,
    title: str | None = None,
    ax=None,
    **grain_values,
):
    plt = _require_matplotlib()

    metric = graph.resolve_metric(metric)
    matrix = graph.flow_matrix(metric=metric, nodes=nodes, top=top, **grain_values)
    if matrix.empty:
        raise ValueError("No routes to plot")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    values = np.asarray(matrix.to_numpy(), dtype=float)
    mesh = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(matrix.shape[1]),
                  [_short(c, label_len) for c in matrix.columns], rotation=90, fontsize=8)
    ax.set_yticks(range(matrix.shape[0]),
                  [_short(i, label_len) for i in matrix.index], fontsize=8)
    ax.set_xlabel("drop cluster")
    ax.set_ylabel("pickup cluster")
    fig.colorbar(mesh, ax=ax, label=metric, shrink=0.85)

    if annotate and values.size <= 400:
        finite = values[np.isfinite(values)]
        midpoint = (finite.max() + finite.min()) / 2 if finite.size else 0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(col, row, _format(value, value_format), ha="center", va="center",
                            fontsize=6, color="white" if value > midpoint else INK)

    ax.set_title(title or f"Pickup x drop {metric}", fontsize=12,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_route_graph(
    graph,
    metric: str | None = None,
    layout: str = "circular",
    top_routes: int | None = None,
    node_direction: str = "out",
    figsize: tuple[float, float] = (14, 10),
    node_size: tuple[float, float] = (300.0, 2200.0),
    edge_width: tuple[float, float] = (0.6, 5.0),
    node_color: str = SOURCE_COLOR,
    edge_cmap: str = "Blues",
    show_labels: bool = True,
    label_len: int | None = None,
    title: str | None = None,
    ax=None,
    **grain_values,
):
    plt = _require_matplotlib()
    import networkx as nx

    metric = graph.resolve_metric(metric)
    simple = graph.simple_graph(metric=metric, **grain_values)
    if top_routes:
        ranked = sorted(simple.edges(data=True),
                        key=lambda e: -np.inf if np.isnan(e[2]["weight"]) else e[2]["weight"],
                        reverse=True)[:top_routes]
        keep = nx.DiGraph() if simple.is_directed() else nx.Graph()
        keep.add_edges_from((u, v, d) for u, v, d in ranked)
        simple = keep

    nodes = list(simple.nodes)
    node_values = [graph.node_value(n, metric=metric, direction=node_direction,
                                    **grain_values)
                   if metric != "count" else float(simple.degree(n)) for n in nodes]

    if layout == "geo":
        pos = geo_layout(nodes)
    elif layout == "circular":
        order = [node for node, _ in sorted(
            zip(nodes, node_values),
            key=lambda item: -np.inf if np.isnan(item[1]) else item[1], reverse=True)]
        angles = np.linspace(0, 2 * np.pi, len(order), endpoint=False) + np.pi / 2
        pos = {node: (float(np.cos(a)), float(np.sin(a))) for node, a in zip(order, angles)}
    elif layout == "spring":
        count = max(simple.number_of_nodes(), 2)
        pos = nx.spring_layout(simple, seed=7, weight=None,
                               k=2.5 / np.sqrt(count), iterations=300)
    elif layout == "shell":
        pos = nx.shell_layout(simple)
    else:
        raise ValueError("layout must be one of: circular, geo, spring, shell")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    sizes = _scale(node_values, *node_size)
    edges = list(simple.edges(data=True))
    widths = _scale([d["weight"] for _, _, d in edges], *edge_width)
    edge_values = [d["weight"] for _, _, d in edges]

    nx.draw_networkx_nodes(simple, pos, nodelist=nodes, node_size=list(sizes),
                           node_color=_tint(node_color, 0.75), edgecolors=node_color,
                           linewidths=1.1, ax=ax)
    drawn = nx.draw_networkx_edges(
        simple, pos, edgelist=[(u, v) for u, v, _ in edges], width=list(widths),
        edge_color=edge_values, edge_cmap=plt.get_cmap(edge_cmap), ax=ax,
        arrows=simple.is_directed(), arrowsize=10, alpha=0.85,
        arrowstyle="-|>", connectionstyle="arc3,rad=0.14", node_size=list(sizes),
    )
    if show_labels:
        nx.draw_networkx_labels(simple, pos,
                                labels={n: _short(n, label_len) for n in nodes},
                                font_size=7, font_color=INK, ax=ax)

    finite = [v for v in edge_values if np.isfinite(v)]
    if finite:
        norm = plt.Normalize(vmin=min(finite), vmax=max(finite))
        mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap(edge_cmap), norm=norm)
        fig.colorbar(mappable, ax=ax, label=metric, shrink=0.8)

    ax.set_title(title or f"Route graph  |  {metric}", fontsize=12,
                 fontweight="bold", color=INK)
    ax.margins(0.1)
    if layout in ("circular", "shell"):
        ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    return fig
