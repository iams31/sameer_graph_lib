"""Build and save a QC graph image from a comma-separated H3 array."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

from sameer_graph_lib import HexGraph
from sameer_graph_lib._h3 import get_resolution


H3_CSV = (
    "88618c4f29fffff,88618c4f23fffff,88618c4f03fffff,88618c4d49fffff,"
    "88618c4d09fffff,88618c4c69fffff,88618c4c29fffff,88618c4d55fffff,"
    "88618c4f3dfffff,88618c4f1dfffff,88618c4d43fffff,88618c4c63fffff,"
    "88618c4c23fffff,88618c4f57fffff,88618c4f37fffff,88618c4f17fffff,"
    "88618c4d5dfffff,88618c4d1dfffff,88618c4f31fffff,88618c4f11fffff,"
    "88618c4d57fffff,88618c4f2bfffff,88618c4d51fffff,88618c4f0bfffff,"
    "88618c4f25fffff,88618c4d4bfffff,88618c4f05fffff,88618c4d0bfffff,"
    "88618c4c6bfffff,88618c4c2bfffff,88618c4c65fffff,88618c4c45fffff,"
    "88618c4c25fffff,88618c4c05fffff,88618c4f39fffff,88618c4f15fffff,"
    "88618c4d5bfffff,88618c4f35fffff,88618c4f19fffff,88618c4f33fffff,"
    "88618c4d59fffff,88618c4c01fffff,88618c4f13fffff,88618c4c21fffff,"
    "88618c4d19fffff,88618c4f2dfffff,88618c4d53fffff,88618c4c61fffff,"
    "88618c4f27fffff,88618c4f07fffff,88618c4d4dfffff,88618c4c6dfffff,"
    "88618c4e27fffff,88618c4c4dfffff,88618c4c2dfffff,88618c4c0dfffff,"
    "88618c4f21fffff,88618c4f01fffff,88618c4c67fffff,88618c4c47fffff,"
    "88618c4c27fffff,88618c4d41fffff,88618c4c07fffff,88618c4f1bfffff,"
    "88618c4f3bfffff"
)


def build_graph() -> tuple[HexGraph, list[str]]:
    hexes = [cell.strip() for cell in H3_CSV.split(",") if cell.strip()]
    resolution = get_resolution(hexes[0])

    graph = HexGraph(hex_resolution=resolution)
    for cell in hexes:
        graph.add_node(cell)

    return graph, hexes


def format_edge_list(edges: list[tuple[str, str, float]]) -> str:
    return "; ".join(f"{u}->{v}({weight})" for u, v, weight in edges)


def format_node_list(nodes: list[str]) -> str:
    return "; ".join(nodes)


def write_insertion_logs(hexes: list[str], output_dir: Path) -> None:
    graph = HexGraph(hex_resolution=get_resolution(hexes[0]))
    csv_path = output_dir / "h3_array_insertion_steps.csv"
    md_path = output_dir / "h3_array_insertion_steps.md"

    rows = []
    for step, cell in enumerate(hexes, start=1):
        graph.add_node(cell)
        log = graph.insertion_log[-1]
        row = {
            "step": step,
            "hex": cell,
            "action": log.get("action", ""),
            "nearest_nodes": format_node_list(log.get("nearest_nodes", [])),
            "nearest_dist": log.get("nearest_dist", ""),
            "connected_to": format_node_list(log.get("connected_to", [])),
            "edges_added": format_edge_list(log.get("edges_added", [])),
            "edges_removed": format_edge_list(log.get("edges_removed", [])),
            "edges_rerouted": format_edge_list(log.get("edges_rerouted", [])),
            "new_count": log.get("new_count", ""),
        }
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# H3 Array Insertion Steps\n\n")
        handle.write("| Step | Hex | Action | Nearest | Distance | Added | Removed | Rerouted |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| {step} | `{hex}` | {action} | {nearest_nodes} | {nearest_dist} | "
                "{edges_added} | {edges_removed} | {edges_rerouted} |\n".format(**row)
            )


def save_step_images(hexes: list[str], output_dir: Path, batch_size: int = 8) -> None:
    graph = HexGraph(hex_resolution=get_resolution(hexes[0]))
    step_records = []

    for step, cell in enumerate(hexes, start=1):
        graph.add_node(cell)
        snapshot = graph.graph.copy()
        step_records.append((step, cell, graph.insertion_log[-1], snapshot))

    for batch_start in range(0, len(step_records), batch_size):
        batch = step_records[batch_start : batch_start + batch_size]
        fig, axes = plt.subplots(1, len(batch), figsize=(4.8 * len(batch), 5.5))
        if len(batch) == 1:
            axes = [axes]

        for ax, (step, cell, log, snapshot) in zip(axes, batch):
            pos = {
                node: graph._get_geo_pos()[node]
                for node in snapshot.nodes
                if node in graph.graph
            }
            node_colors = ["#e67e22" if node == cell else "#3498db" for node in snapshot.nodes]
            node_labels = {
                node: f"{idx + 1}\n{str(node)[-4:]}"
                for idx, node in enumerate(snapshot.nodes)
            }
            edge_labels = {
                (u, v): f"{data.get('weight', data.get('distance', ''))}"
                for u, v, data in snapshot.edges(data=True)
            }

            nx.draw_networkx_edges(snapshot, pos, edge_color="#2c3e50", width=1.6, ax=ax)
            nx.draw_networkx_nodes(
                snapshot,
                pos,
                node_color=node_colors,
                node_size=420,
                ax=ax,
                edgecolors="black",
                linewidths=1.0,
            )
            nx.draw_networkx_labels(snapshot, pos, labels=node_labels, font_size=7, ax=ax)
            nx.draw_networkx_edge_labels(
                snapshot,
                pos,
                edge_labels=edge_labels,
                font_size=6,
                font_color="red",
                ax=ax,
            )

            title_lines = [f"Step {step}", cell]
            if log["action"] == "insert":
                title_lines.append(f"nearest d={log['nearest_dist']} count={len(log['nearest_nodes'])}")
                if log.get("edges_removed"):
                    title_lines.append(f"removed={len(log['edges_removed'])}")
                if log.get("edges_rerouted"):
                    title_lines.append(f"rerouted={len(log['edges_rerouted'])}")
            elif log["action"] == "second_node":
                title_lines.append(f"connect d={log['distance']}")
            elif log["action"] == "increment":
                title_lines.append(f"exists count={log['new_count']}")
            else:
                title_lines.append(log["action"])

            ax.set_title("\n".join(title_lines), fontsize=8, fontweight="bold")
            ax.axis("off")

        first_step = batch[0][0]
        last_step = batch[-1][0]
        fig.suptitle(
            f"H3 node insertion steps {first_step}-{last_step}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        output_path = output_dir / f"h3_array_steps_{first_step:02d}_{last_step:02d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "h3_array_graph.png"
    cell_output_path = output_dir / "h3_array_cells.png"
    basemap_output_path = output_dir / "h3_array_cells_basemap.png"
    hull_output_path = output_dir / "h3_array_convex_hull.geojson"

    graph, hexes = build_graph()
    selected = graph.get_appropriate_hexes(cutoff=0.8)
    stats = graph.get_appropriate_hexes_with_stats(cutoff=0.8)

    title = (
        "H3 Array Graph QC "
        f"({graph.get_graph_stats()['num_nodes']} nodes, "
        f"80% cluster: {stats['num_hexes']} hexes)"
    )
    fig = graph.visualize_graph(
        title=title,
        highlight_hexes=selected,
        figsize=(16, 11),
        show_edge_weights=False,
        show_labels=False,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    cell_fig = graph.plot_h3_cells(
        title="H3 Array Cell Footprint",
        highlight_hexes=selected,
        figsize=(12, 10),
        show_labels=False,
    )
    cell_fig.savefig(cell_output_path, dpi=180, bbox_inches="tight")
    plt.close(cell_fig)

    hull_geojson = {
        "type": "Feature",
        "properties": {"name": "h3_array_center_convex_hull"},
        "geometry": graph.convex_hull_geojson(),
    }
    hull_output_path.write_text(json.dumps(hull_geojson, indent=2), encoding="utf-8")

    try:
        basemap_fig = graph.plot_h3_cells_map(
            title="H3 Array Cell Footprint on Basemap",
            highlight_hexes=selected,
            figsize=(14, 14),
            basemap=True,
        )
        basemap_fig.savefig(basemap_output_path, dpi=180, bbox_inches="tight")
        plt.close(basemap_fig)
        print(f"Saved basemap footprint: {basemap_output_path}")
    except ImportError:
        print("Skipped basemap footprint. Install with: uv run --extra geo --extra plot ...")

    write_insertion_logs(hexes, output_dir)
    save_step_images(hexes, output_dir)

    print(f"Input hexes: {len(hexes)}")
    print(f"Graph stats: {graph.get_graph_stats()}")
    print(f"80% cluster stats: {stats}")
    print(f"Saved image: {output_path}")
    print(f"Saved cell footprint: {cell_output_path}")
    print(f"Saved convex hull GeoJSON: {hull_output_path}")
    print(f"Saved insertion CSV: {output_dir / 'h3_array_insertion_steps.csv'}")
    print(f"Saved insertion Markdown: {output_dir / 'h3_array_insertion_steps.md'}")
    print(f"Saved step images: {output_dir / 'h3_array_steps_*.png'}")


if __name__ == "__main__":
    main()
