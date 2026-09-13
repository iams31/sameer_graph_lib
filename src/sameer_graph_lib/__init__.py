"""Public API for sameer_graph_lib."""

from .affinity_graph import AffinityGraph
from .corridor_extractor import CorridorExtractor
from .geometry import getLatLng, get_latlng, h3_convex_hull, h3_convex_hull_geojson, makingHull, making_hull
from .hex_graph import HexGraph
from .motion import (
    compute_accelerations,
    compute_accelerations_from_timestamps,
    compute_speeds,
    compute_speeds_from_timestamps,
    detect_sudden_braking_from_timestamps,
    dominant_frequency_hz,
    get_max_speed,
    max_accelaration,
    resample_polyline,
    smooth_speeds_median,
)
from .plotting import cells_to_geodataframe, plot_h3_cells, plot_h3_cells_map
from .plylinedecoding import decode_polyline, get_hexes_from_polyline
from .spatial_ingestor import SpatialIngestor
from .topology_analyzer import TopologyAnalyzer

# The route stack and the DataFrame plotter need pandas/numpy (and xarray or
# matplotlib for some views), so they are imported on first use instead of at
# package import time. ``from sameer_graph_lib import RouteExplorer`` works the
# same way it would for an eager import.
_LAZY = {
    "CATEGORICAL": "route_graph",
    "MEAN_METRICS": "route_graph",
    "RouteGraph": "route_graph",
    "RouteSchema": "route_graph",
    "RouteTensor": "route_graph",
    "SUM_METRICS": "route_graph",
    "RouteExplorer": "route_explorer",
    "flow_layout": "route_viz",
    "geo_layout": "route_viz",
    "plot_flow": "route_viz",
    "plot_matrix": "route_viz",
    "plot_partners": "route_viz",
    "plot_profile": "route_viz",
    "plot_route_graph": "route_viz",
    "PALETTE": "plotter",
    "compare_distributions": "plotter",
    "compare_groups": "plotter",
    "describe_distribution": "plotter",
    "ecdf_points": "plotter",
    "group_stats": "plotter",
    "kde_curve": "plotter",
    "plot_columns": "plotter",
    "plot_correlation": "plotter",
    "plot_distribution": "plotter",
    "plot_grid": "plotter",
    "plot_multi_axis": "plotter",
    "plot_overlay": "plotter",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "AffinityGraph",
    "CATEGORICAL",
    "CorridorExtractor",
    "HexGraph",
    "MEAN_METRICS",
    "PALETTE",
    "RouteExplorer",
    "RouteGraph",
    "RouteSchema",
    "RouteTensor",
    "SUM_METRICS",
    "SpatialIngestor",
    "TopologyAnalyzer",
    "cells_to_geodataframe",
    "compare_distributions",
    "compare_groups",
    "compute_accelerations",
    "compute_accelerations_from_timestamps",
    "compute_speeds",
    "compute_speeds_from_timestamps",
    "decode_polyline",
    "describe_distribution",
    "detect_sudden_braking_from_timestamps",
    "dominant_frequency_hz",
    "ecdf_points",
    "flow_layout",
    "geo_layout",
    "getLatLng",
    "get_hexes_from_polyline",
    "get_latlng",
    "get_max_speed",
    "group_stats",
    "h3_convex_hull",
    "h3_convex_hull_geojson",
    "kde_curve",
    "makingHull",
    "making_hull",
    "max_accelaration",
    "plot_columns",
    "plot_correlation",
    "plot_distribution",
    "plot_flow",
    "plot_grid",
    "plot_h3_cells",
    "plot_h3_cells_map",
    "plot_matrix",
    "plot_multi_axis",
    "plot_overlay",
    "plot_partners",
    "plot_profile",
    "plot_route_graph",
    "resample_polyline",
    "smooth_speeds_median",
]

__version__ = "0.3.0"
