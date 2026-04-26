"""Small compatibility layer for h3-py 3.x and 4.x APIs."""

from __future__ import annotations

import math
from typing import List, Tuple

import h3

LatLng = Tuple[float, float]

EARTH_RADIUS_KM = 6371.0088


def is_valid_cell(cell: str) -> bool:
    fn = getattr(h3, "is_valid_cell", None) or getattr(h3, "h3_is_valid", None)
    return bool(fn(cell)) if fn else False


def latlng_to_cell(lat: float, lng: float, resolution: int) -> str:
    fn = getattr(h3, "latlng_to_cell", None)
    if fn:
        return fn(lat, lng, resolution)
    return h3.geo_to_h3(lat, lng, resolution)


def cell_to_latlng(cell: str) -> LatLng:
    fn = getattr(h3, "cell_to_latlng", None)
    if fn:
        lat, lng = fn(cell)
    else:
        lat, lng = h3.h3_to_geo(cell)
    return float(lat), float(lng)


def cell_to_boundary(cell: str) -> List[LatLng]:
    fn = getattr(h3, "cell_to_boundary", None)
    if fn:
        return [(float(lat), float(lng)) for lat, lng in fn(cell)]

    legacy_fn = getattr(h3, "h3_to_geo_boundary", None)
    if legacy_fn:
        return [(float(lat), float(lng)) for lat, lng in legacy_fn(cell, geo_json=False)]

    raise RuntimeError("Installed h3 package does not expose cell boundary helpers.")


def get_resolution(cell: str) -> int:
    fn = getattr(h3, "get_resolution", None) or getattr(h3, "h3_get_resolution", None)
    return int(fn(cell))


def grid_path_cells(start: str, end: str) -> List[str]:
    fn = getattr(h3, "grid_path_cells", None) or getattr(h3, "h3_line", None)
    if not fn:
        raise RuntimeError("Installed h3 package does not expose grid path helpers.")
    return list(fn(start, end))


def average_hexagon_edge_length(resolution: int, unit: str = "km") -> float:
    fn = getattr(h3, "average_hexagon_edge_length", None)
    if fn:
        return float(fn(resolution, unit=unit))

    fn = getattr(h3, "edge_length", None)
    if fn:
        try:
            return float(fn(resolution, unit=unit))
        except TypeError:
            return float(fn(res=resolution, unit=unit))

    raise RuntimeError("Installed h3 package does not expose edge length helpers.")


def cell_area(cell: str, unit: str = "km^2") -> float:
    fn = getattr(h3, "cell_area", None)
    if not fn:
        raise RuntimeError("Installed h3 package does not expose cell_area.")
    return float(fn(cell, unit=unit))


def haversine_km(a: LatLng, b: LatLng) -> float:
    lat1, lng1 = map(math.radians, a)
    lat2, lng2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    sin_dlat = math.sin(dlat / 2.0)
    sin_dlng = math.sin(dlng / 2.0)
    h = sin_dlat**2 + math.cos(lat1) * math.cos(lat2) * sin_dlng**2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def approximate_grid_distance(start: str, end: str) -> int:
    if start == end:
        return 0

    resolution = get_resolution(start)
    edge_km = average_hexagon_edge_length(resolution, unit="km")
    center_spacing_km = max(edge_km * math.sqrt(3), 1e-9)
    distance_km = haversine_km(cell_to_latlng(start), cell_to_latlng(end))
    return max(1, int(round(distance_km / center_spacing_km)))


def grid_distance(start: str, end: str) -> int:
    """Return native H3 grid distance between two cells."""
    if start == end:
        return 0

    fn = getattr(h3, "grid_distance", None)
    if fn:
        try:
            return int(fn(start, end))
        except Exception:
            pass

    legacy_fn = getattr(h3, "h3_distance", None)
    if legacy_fn:
        try:
            return int(legacy_fn(start, end))
        except Exception:
            pass

    return approximate_grid_distance(start, end)
