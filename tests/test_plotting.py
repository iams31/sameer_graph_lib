import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import h3
import matplotlib.pyplot as plt

from sameer_graph_lib import HexGraph


def test_visualize_graph_returns_matplotlib_figure():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    hexes = list(h3.grid_disk(center, 1))[:3]
    graph = HexGraph(hex_resolution=8)
    graph.add_hex_array(hexes)

    fig = graph.visualize_graph(highlight_hexes=hexes[:2])

    assert fig.__class__.__name__ == "Figure"
    plt.close(fig)


def test_visualize_step_by_step_returns_matplotlib_figure():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    hexes = list(h3.grid_disk(center, 1))[:3]
    graph = HexGraph(hex_resolution=8)

    fig = graph.visualize_step_by_step(hexes, labels=["A"])

    assert fig.__class__.__name__ == "Figure"
    plt.close(fig)

