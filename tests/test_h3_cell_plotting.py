import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import h3
import matplotlib.pyplot as plt

from sameer_graph_lib import HexGraph, plot_h3_cells


def test_plot_h3_cells_accepts_single_cell():
    cell = h3.latlng_to_cell(12.935, 77.624, 8)

    fig = plot_h3_cells(cell, show_labels=True, label_full_hex=True)

    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes[0].patches) == 1
    plt.close(fig)


def test_plot_h3_cells_accepts_array_and_highlight():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    cells = list(h3.grid_disk(center, 1))

    fig = plot_h3_cells(cells, selected_cells=cells[0], show_centers=True)

    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes[0].patches) == len(cells)
    plt.close(fig)


def test_graph_plot_h3_cells_defaults_to_graph_nodes():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    cells = list(h3.grid_disk(center, 1))[:3]
    graph = HexGraph(hex_resolution=8)
    graph.add_hex_array(cells)

    fig = graph.plot_h3_cells(show_labels=False)

    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes[0].patches) == graph.graph.number_of_nodes()
    plt.close(fig)

