import h3

from sameer_graph_lib._h3 import grid_distance, latlng_to_cell


def test_grid_distance_uses_native_h3_grid_distance():
    origin = latlng_to_cell(12.9716, 77.5946, 9)
    neighbor = sorted(cell for cell in h3.grid_disk(origin, 1) if cell != origin)[0]

    assert grid_distance(origin, origin) == h3.grid_distance(origin, origin)
    assert grid_distance(origin, neighbor) == h3.grid_distance(origin, neighbor)

