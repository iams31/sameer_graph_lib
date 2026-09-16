
from __future__ import annotations
def _require_polyline():
    try:
        import polyline 
    except ImportError as exc:
        raise ImportError(
            "Motion helpers require polyline: pip install 'sameer-graph-lib[motion]'"
        ) from exc
    return polyline
def _require_h3():
    try:
        import h3 
    except ImportError as exc:
        raise ImportError(
            "Motion helpers require polyline: pip install 'sameer-graph-lib[motion]'"
        ) from exc
    return h3

def decode_polyline(
    polyline_str: str,
    precision: int = 5,
    geojson: bool = False,
) -> list[tuple[float, float]]:
    polyline=_require_polyline()
    return polyline.decode(polyline_str, precision=precision, geojson=geojson)


def get_hexes_from_polyline(
    polyline_str: str,
    precision: int = 5,
    res: int = 9,
) -> list[str]:
    h3=_require_h3()
    coordinates = decode_polyline(polyline_str, precision=precision, geojson=False)
    return [h3.latlng_to_cell(lat, lng, res) for lat, lng in coordinates]
