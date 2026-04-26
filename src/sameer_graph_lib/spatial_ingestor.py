"""Input normalization utilities for H3 route data."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from ._h3 import (
    cell_to_latlng,
    get_resolution,
    grid_distance,
    grid_path_cells,
    is_valid_cell,
    latlng_to_cell,
)

Coordinate = Tuple[float, float]


class SpatialIngestor:
    """Convert raw spatial inputs into contiguous H3 chains."""

    def __init__(self, resolution: int = 9, strict: bool = True) -> None:
        self.resolution = int(resolution)
        self.strict = strict

    def ingest_h3_array(self, hex_list: Iterable[str]) -> List[str]:
        """Validate and gap-fill an explicit list of H3 cells."""
        return self.normalize_h3_chain(list(hex_list))

    def ingest_latlng_sequence(
        self,
        coords: Sequence[Coordinate],
        resolution: int | None = None,
    ) -> List[str]:
        """Convert a lat/lng sequence into a contiguous H3 route."""
        res = self.resolution if resolution is None else int(resolution)
        cells = [latlng_to_cell(float(lat), float(lng), res) for lat, lng in coords]
        return self.normalize_h3_chain(cells)

    def ingest_encoded_polyline(
        self,
        polyline_str: str,
        resolution: int | None = None,
        precision: int = 5,
    ) -> List[str]:
        """Decode a Google encoded polyline and convert it to H3 cells."""
        return self.ingest_latlng_sequence(
            decode_polyline(polyline_str, precision=precision),
            resolution=resolution,
        )

    def normalize_h3_chain(self, hexes: Sequence[str]) -> List[str]:
        """Remove consecutive duplicates and fill gaps between adjacent samples."""
        cells = self._dedupe_consecutive([str(h) for h in hexes if h])
        if not cells:
            return []

        self._validate_cells(cells)
        if len(cells) == 1:
            return cells

        normalized = [cells[0]]
        for cell in cells[1:]:
            if cell == normalized[-1]:
                continue

            bridge = self._bridge_cells(normalized[-1], cell)
            if not bridge:
                normalized.append(cell)
            elif bridge[0] == normalized[-1]:
                normalized.extend(bridge[1:])
            else:
                normalized.extend(bridge)

        return self._dedupe_consecutive(normalized)

    def _validate_cells(self, cells: Sequence[str]) -> None:
        invalid = [cell for cell in cells if not is_valid_cell(cell)]
        if invalid and self.strict:
            sample = ", ".join(invalid[:3])
            raise ValueError(f"Invalid H3 cell(s): {sample}")

        resolutions = {get_resolution(cell) for cell in cells if is_valid_cell(cell)}
        if len(resolutions) > 1 and self.strict:
            raise ValueError("All H3 cells in one route must use the same resolution.")

    def _bridge_cells(self, start: str, end: str) -> List[str]:
        try:
            path = grid_path_cells(start, end)
            return path if path else [start, end]
        except Exception:
            return self._fallback_bridge_cells(start, end)

    def _fallback_bridge_cells(self, start: str, end: str) -> List[str]:
        """Approximate a bridge when h3 cannot produce an exact grid path."""
        if start == end:
            return [start]

        resolution = get_resolution(start)
        steps = max(1, grid_distance(start, end))
        lat1, lng1 = cell_to_latlng(start)
        lat2, lng2 = cell_to_latlng(end)

        sampled = []
        for idx in range(steps + 1):
            ratio = idx / steps
            lat = lat1 + (lat2 - lat1) * ratio
            lng = lng1 + (lng2 - lng1) * ratio
            sampled.append(latlng_to_cell(lat, lng, resolution))

        sampled[0] = start
        sampled[-1] = end
        sampled = self._dedupe_consecutive(sampled)

        expanded = [sampled[0]]
        for cell in sampled[1:]:
            if cell == expanded[-1]:
                continue
            try:
                bridge = grid_path_cells(expanded[-1], cell)
                expanded.extend(bridge[1:] if bridge and bridge[0] == expanded[-1] else bridge)
            except Exception:
                expanded.append(cell)

        return self._dedupe_consecutive(expanded)

    @staticmethod
    def _dedupe_consecutive(items: Sequence[str]) -> List[str]:
        deduped: List[str] = []
        for item in items:
            if not deduped or deduped[-1] != item:
                deduped.append(item)
        return deduped


def decode_polyline(polyline_str: str, precision: int = 5) -> List[Coordinate]:
    """Decode a Google encoded polyline string without extra dependencies."""
    coordinates: List[Coordinate] = []
    index = 0
    lat = 0
    lng = 0
    factor = 10**precision

    while index < len(polyline_str):
        lat_delta, index = _decode_polyline_value(polyline_str, index)
        lng_delta, index = _decode_polyline_value(polyline_str, index)
        lat += lat_delta
        lng += lng_delta
        coordinates.append((lat / factor, lng / factor))

    return coordinates


def _decode_polyline_value(polyline_str: str, index: int) -> tuple[int, int]:
    result = 0
    shift = 0

    while True:
        if index >= len(polyline_str):
            raise ValueError("Invalid encoded polyline: truncated value.")
        byte = ord(polyline_str[index]) - 63
        index += 1
        result |= (byte & 0x1F) << shift
        shift += 5
        if byte < 0x20:
            break

    value = ~(result >> 1) if result & 1 else result >> 1
    return value, index

