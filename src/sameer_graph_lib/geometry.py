"""Geometry helpers for H3 cell collections."""

from __future__ import annotations

from collections.abc import Iterable

from ._h3 import cell_to_latlng, is_valid_cell
from .plotting import normalize_cells


def get_latlng(cells: str | Iterable[str]) -> list[tuple[float, float]]:
    """Return H3 cell centers as ``(lat, lng)`` tuples."""
    hexes = normalize_cells(cells)
    invalid = [cell for cell in hexes if not is_valid_cell(cell)]
    if invalid:
        raise ValueError(f"Invalid H3 cell(s): {', '.join(invalid[:3])}")
    return [cell_to_latlng(cell) for cell in hexes]


def get_lnglat(cells: str | Iterable[str]) -> list[tuple[float, float]]:
    """Return H3 cell centers as Shapely-friendly ``(lng, lat)`` tuples."""
    return [(lng, lat) for lat, lng in get_latlng(cells)]


def making_hull(points: Iterable[tuple[float, float]]):
    """Return the convex hull for input points.

    The expected coordinate order is ``(lat, lng)`` to match your current
    helper. The returned Shapely geometry uses standard ``(lng, lat)`` order.
    """
    try:
        from shapely.geometry import MultiPoint, Point
    except ImportError as exc:
        raise ImportError(
            "Convex hull helpers require the geo extra: "
            "pip install 'sameer-graph-lib[geo]'"
        ) from exc

    coords = list(points)
    if not coords:
        return None

    lnglat = [(lng, lat) for lat, lng in coords]
    if len(lnglat) == 1:
        return Point(lnglat[0])

    return MultiPoint(lnglat).convex_hull


def h3_convex_hull(cells: str | Iterable[str]):
    """Return a Shapely convex hull around H3 cell centers."""
    return making_hull(get_latlng(cells))


def h3_convex_hull_geojson(cells: str | Iterable[str]) -> dict | None:
    """Return the H3 center convex hull as a GeoJSON-like mapping."""
    hull = h3_convex_hull(cells)
    if hull is None:
        return None

    try:
        from shapely.geometry import mapping
    except ImportError as exc:
        raise ImportError(
            "Convex hull helpers require the geo extra: "
            "pip install 'sameer-graph-lib[geo]'"
        ) from exc

    return mapping(hull)


# Backwards-compatible aliases matching the user's current helper names.
getLatLng = get_latlng
makingHull = making_hull

