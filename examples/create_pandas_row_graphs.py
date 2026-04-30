"""Create one HexGraph per pandas DataFrame row.

This example is for data shaped like:

    route_id | hexes
    -------- | -----------------------------------------
    r1       | ["88618c4f29fffff", "88618c4f23fffff"]
    r2       | ["88618c4f29fffff", "88618c4d49fffff"]

Each row gets its own independent graph. The graphs are stored in a new
``graph`` column so you can inspect stats, save plots, or run corridor
selection per row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from sameer_graph_lib import HexGraph
from sameer_graph_lib._h3 import get_resolution


def normalize_hexes(value: Iterable[str] | str) -> list[str]:
    """Return a clean H3 list from either a list-like value or CSV string."""
    if isinstance(value, str):
        raw_hexes = value.split(",")
    else:
        raw_hexes = value

    return [str(cell).strip() for cell in raw_hexes if str(cell).strip()]


def create_graph_for_row(hexes: Iterable[str] | str) -> HexGraph | None:
    """Build one independent HexGraph for a single DataFrame row."""
    clean_hexes = normalize_hexes(hexes)
    if not clean_hexes:
        return None

    graph = HexGraph(hex_resolution=get_resolution(clean_hexes[0]))
    graph.add_hex_array(clean_hexes)
    return graph


def save_graph_image(graph: HexGraph, output_path: Path, title: str) -> None:
    """Save a QC graph image for one row."""
    import matplotlib.pyplot as plt

    fig = graph.visualize_graph(
        title=title,
        show_labels=False,
        show_edge_weights=False,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.DataFrame(
        {
            "route_id": ["route_1", "route_2"],
            "hexes": [
                [
                    "88618c4f29fffff",
                    "88618c4f23fffff",
                    "88618c4f03fffff",
                    "88618c4d49fffff",
                ],
                [
                    "88618c4f29fffff",
                    "88618c4d49fffff",
                    "88618c4d09fffff",
                    "88618c4c69fffff",
                ],
            ],
        }
    )

    df["graph"] = df["hexes"].apply(create_graph_for_row)
    df["graph_stats"] = df["graph"].apply(lambda graph: graph.get_graph_stats() if graph else None)

    output_dir = Path(__file__).resolve().parent
    for row_index, row in df.iterrows():
        graph = row["graph"]
        if graph is None:
            continue

        route_id = row["route_id"]
        selected = graph.get_appropriate_hexes(cutoff=0.8)
        stats = graph.get_appropriate_hexes_with_stats(cutoff=0.8)

        print(f"{route_id} graph stats: {row['graph_stats']}")
        print(f"{route_id} 80% selected hexes: {selected}")
        print(f"{route_id} 80% selected stats: {stats}")

        save_graph_image(
            graph,
            output_dir / f"pandas_row_graph_{row_index}.png",
            title=f"Graph for {route_id}",
        )


if __name__ == "__main__":
    main()
