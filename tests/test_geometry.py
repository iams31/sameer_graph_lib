import pytest

pytest.importorskip("shapely")

import h3

from sameer_graph_lib import HexGraph, getLatLng, h3_convex_hull, makingHull


def test_get_latlng_and_h3_convex_hull():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    cells = list(h3.grid_disk(center, 1))

    latlng = getLatLng(cells)
    hull = h3_convex_hull(cells)

    assert len(latlng) == len(cells)
    assert hull.geom_type in {"Polygon", "LineString", "Point"}
    assert not hull.is_empty


def test_making_hull_accepts_lat_lng_points():
    points = [(12.935, 77.624), (12.936, 77.625), (12.934, 77.626)]

    hull = makingHull(points)

    assert hull.geom_type == "Polygon"
    assert not hull.is_empty


def test_graph_convex_hull_methods():
    center = h3.latlng_to_cell(12.935, 77.624, 8)
    cells = list(h3.grid_disk(center, 1))
    graph = HexGraph(hex_resolution=8)
    graph.add_hex_array(cells)

    hull = graph.convex_hull()
    geojson = graph.convex_hull_geojson()

    assert hull.geom_type == "Polygon"
    assert geojson["type"] == "Polygon"

