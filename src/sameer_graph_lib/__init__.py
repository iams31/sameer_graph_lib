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

__all__ = [
    "AffinityGraph",
    "CorridorExtractor",
    "HexGraph",
    "cells_to_geodataframe",
    "compute_accelerations",
    "compute_accelerations_from_timestamps",
    "compute_speeds",
    "compute_speeds_from_timestamps",
    "decode_polyline",
    "detect_sudden_braking_from_timestamps",
    "dominant_frequency_hz",
    "getLatLng",
    "get_hexes_from_polyline",
    "get_latlng",
    "get_max_speed",
    "h3_convex_hull",
    "h3_convex_hull_geojson",
    "makingHull",
    "making_hull",
    "max_accelaration",
    "plot_h3_cells",
    "plot_h3_cells_map",
    "resample_polyline",
    "smooth_speeds_median",
    "SpatialIngestor",
    "TopologyAnalyzer",
]

__version__ = "0.1.0"
