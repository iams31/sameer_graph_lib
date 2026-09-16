
from __future__ import annotations

import numpy as np

from ._h3 import cell_to_boundary, cell_to_latlng, grid_distance

INK = "#1F2933"
MUTED_INK = "#7B8794"
HAIRLINE = "#CBD2D9"


def short_cell(cell, length: int = 4) -> str:
    text = str(cell)
    trimmed = text.rstrip("f")
    return (trimmed or text)[-length:]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib: pip install 'sameer-graph-lib[plot]'"
        ) from exc
    return plt


def plot_hex_metric(
    hex_graph,
    metric=None,
    highlight=None,
    figsize: tuple[float, float] = (11, 9),
    cmap: str = "Blues",
    title: str | None = None,
    show_edges: bool = True,
    show_labels: bool = False,
    label_len: int = 6,
    value_format: str = "{:,.0f}",
    annotate: bool | None = None,
    edge_color: str = "#9AA5AD",
    highlight_color: str = "#B3473F",
    legend: bool = True,
    colorbar: bool = True,
    ax=None,
    **grain_values,
):
    plt = _require_matplotlib()
    from matplotlib.patches import Polygon
    import matplotlib.patches as mpatches

    metric = metric or hex_graph.value_metric
    cells = hex_graph.cells
    if not cells:
        raise ValueError("No cells to plot")

    values = np.array([hex_graph.value(cell, metric, **grain_values) for cell in cells],
                      dtype=float)
    finite = values[np.isfinite(values)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    palette = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=low, vmax=high)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    picked = set(highlight or ())
    if annotate is None:
        annotate = len(cells) <= 60

    lngs, lats = [], []
    for cell, value in zip(cells, values):
        boundary = [(lng, lat) for lat, lng in cell_to_boundary(cell)]
        lngs.extend(point[0] for point in boundary)
        lats.extend(point[1] for point in boundary)
        selected = cell in picked
        ax.add_patch(Polygon(
            boundary, closed=True,
            facecolor=palette(norm(value)) if np.isfinite(value) else "#F0F2F4",
            edgecolor=highlight_color if selected else HAIRLINE,
            linewidth=2.0 if selected else 0.6,
            alpha=0.95, zorder=2 if selected else 1,
        ))

    if show_edges:
        for u, v in hex_graph.graph.graph.edges():
            u_lat, u_lng = cell_to_latlng(u)
            v_lat, v_lng = cell_to_latlng(v)
            ax.plot([u_lng, v_lng], [u_lat, v_lat], color=edge_color,
                    linewidth=0.8, alpha=0.7, zorder=3)

    for cell, value in zip(cells, values):
        lat, lng = cell_to_latlng(cell)
        text = []
        if show_labels:
            text.append(short_cell(cell, label_len))
        if annotate and np.isfinite(value):
            text.append(value_format.format(value))
        if text:
            shade = norm(value) if np.isfinite(value) else 0
            ax.text(lng, lat, "\n".join(text), ha="center", va="center",
                    fontsize=7, zorder=4,
                    color="white" if shade > 0.6 else INK)

    if lngs and lats:
        pad_x = max((max(lngs) - min(lngs)) * 0.06, 1e-4)
        pad_y = max((max(lats) - min(lats)) * 0.06, 1e-4)
        ax.set_xlim(min(lngs) - pad_x, max(lngs) + pad_x)
        ax.set_ylim(min(lats) - pad_y, max(lats) + pad_y)
    ax.set_aspect("equal", adjustable="box")

    if colorbar:
        mappable = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        fig.colorbar(mappable, ax=ax, label=metric, shrink=0.8)
    if legend and picked:
        ax.legend(handles=[mpatches.Patch(facecolor="none", edgecolor=highlight_color,
                                          linewidth=2.0, label="selected")],
                  loc="upper left", fontsize=9, framealpha=0.9)

    if title is None:
        title = f"{metric} per H3 cell"
        if grain_values:
            title += "  |  " + ", ".join(f"{k}={v}" for k, v in grain_values.items())
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel("Longitude", color=MUTED_INK)
    ax.set_ylabel("Latitude", color=MUTED_INK)
    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_hex_steps(
    hex_graph,
    cells=None,
    metric=None,
    ncols: int = 4,
    panel_size: tuple[float, float] = (4.2, 3.8),
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    value_format: str = "{:,.0f}",
    new_color: str = "#B3473F",
    old_color: str = "#4C78A8",
    show_distances: bool = True,
    show_values: bool = True,
    label_len: int = 4,
    font_size: int = 8,
    **grain_values,
):
    plt = _require_matplotlib()
    import networkx as nx

    from .affinity_graph import AffinityGraph

    metric = metric or hex_graph.value_metric
    order = list(hex_graph.tensors)
    wanted = order if cells is None else list(cells)
    unknown = [c for c in wanted if c not in hex_graph]
    if unknown:
        raise KeyError(f"Cell(s) not in the graph: {unknown[:3]}")
    if not wanted:
        raise ValueError("No cells to draw")
    drawn_set = set(wanted)

    values = {cell: hex_graph.value(cell, metric, **grain_values) for cell in order}
    steps = len(drawn_set)
    ncols = max(1, min(ncols, steps))
    nrows = int(np.ceil(steps / ncols))
    figsize = figsize or (panel_size[0] * ncols, panel_size[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    panels = list(axes.ravel())

    replay = AffinityGraph(hex_resolution=hex_graph.resolution)
    panel = 0
    for step, cell in enumerate(order, start=1):
        if hex_graph.topology == "attach":
            replay.add_hex(cell)
        else:
            replay.graph.add_node(cell, count=0, value=0.0)
            for other in list(replay.graph.nodes):
                if other != cell:
                    steps_apart = grid_distance(cell, other)
                    if hex_graph.topology == "adjacent" and 0 < steps_apart <= hex_graph.ring:
                        replay._add_or_update_edge(cell, other, steps_apart,
                                                   count_increment=0, kind="adjacent")

        if cell not in drawn_set:
            continue
        ax = panels[panel]
        panel += 1

        drawn = replay.graph
        pos = {node: cell_to_latlng(node)[::-1] for node in drawn.nodes}
        colors = [new_color if node == cell else old_color for node in drawn.nodes]
        sizes = [420 if node == cell else 300 for node in drawn.nodes]

        nx.draw_networkx_edges(drawn, pos, edge_color=HAIRLINE, width=1.2, ax=ax)
        nx.draw_networkx_nodes(drawn, pos, node_color=colors, node_size=sizes,
                               edgecolors=INK, linewidths=0.8, ax=ax)
        nx.draw_networkx_labels(
            drawn, pos, labels={n: short_cell(n, label_len) for n in drawn.nodes},
            font_size=font_size - 1, font_color="white", font_weight="bold", ax=ax)

        if show_values:
            for node in drawn.nodes:
                lat, lng = cell_to_latlng(node)
                ax.annotate(_format_value(values.get(node), value_format),
                            xy=(lng, lat), xytext=(0, -14),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=font_size - 1, family="monospace", color=INK,
                            bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                                      edgecolor=HAIRLINE, linewidth=0.5, alpha=0.9))

        if show_distances and drawn.number_of_edges():
            labels = {(u, v): str(int(d.get("distance", d.get("weight", 0)) or 0))
                      for u, v, d in drawn.edges(data=True)}
            nx.draw_networkx_edge_labels(drawn, pos, edge_labels=labels,
                                         font_size=max(font_size - 3, 5),
                                         font_color=MUTED_INK, ax=ax, rotate=False)

        log = replay.insertion_log[-1] if replay.insertion_log else {}
        note = {"first_node": "first cell", "second_node": "joined the first",
                "increment": "already present"}.get(log.get("action"), "")
        if log.get("action") == "insert":
            note = f"nearest d={log.get('nearest_dist')}"
            if log.get("edges_rerouted"):
                note += f", rerouted {len(log['edges_rerouted'])}"
        ax.set_title(f"{step}. + {short_cell(cell, label_len)}"
                     f"   {_format_value(values.get(cell), value_format)}"
                     + (f"\n{note}" if note else ""),
                     fontsize=font_size + 1, color=INK)
        ax.margins(0.22)
        ax.axis("off")

    for extra in panels[steps:]:
        extra.axis("off")

    if title is None:
        title = f"Adding cells one at a time  |  {metric}"
        if grain_values:
            title += "  |  " + ", ".join(f"{k}={v}" for k, v in grain_values.items())
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _format_value(value, fmt="{:,.0f}") -> str:
    if value is None:
        return "n/a"
    try:
        if np.isnan(value):
            return "n/a"
    except TypeError:
        return str(value)
    return fmt.format(value)
