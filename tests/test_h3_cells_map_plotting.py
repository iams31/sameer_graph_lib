import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("geopandas")
pytest.importorskip("contextily")

import h3
import matplotlib.pyplot as plt

from sameer_graph_lib import HexGraph, cells_to_geodataframe, plot_h3_cells_map


def test_cells_to_geodataframe_builds_h3_polygons():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    cells = list(h3.grid_disk(center, 1))

    gdf = cells_to_geodataframe(cells, selected_cells=cells[0])

    assert len(gdf) == len(cells)
    assert gdf.crs.to_string() == "EPSG:4326"
    assert gdf["selected"].sum() == 1


def test_plot_h3_cells_map_without_basemap_returns_figure():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    cells = list(h3.grid_disk(center, 1))

    fig = plot_h3_cells_map(cells, selected_cells=cells[0], basemap=False)

    assert fig.__class__.__name__ == "Figure"
    plt.close(fig)


def test_graph_plot_h3_cells_map_without_basemap_returns_figure():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    cells = list(h3.grid_disk(center, 1))[:3]
    graph = HexGraph(hex_resolution=8)
    graph.add_hex_array(cells)

    fig = graph.plot_h3_cells_map(basemap=False)

    assert fig.__class__.__name__ == "Figure"
    plt.close(fig)

