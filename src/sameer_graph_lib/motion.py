"""Motion analysis for polylines sampled at a fixed time interval.

All inputs use the convention ``polyline = [(lat, lng), (lat, lng), ...]`` and
``interval_seconds`` is the time between consecutive points. Distances are
computed with the haversine formula on a sphere of radius 6_371_000 m.
"""

from __future__ import annotations
from .plylinedecoding import decode_polyline
from collections.abc import Sequence
Point = tuple[float, float]
TimedPoint = tuple[float, float, float]
def _require_pyproj():
    try:
        import pyproj
    except ImportError as exc:
        raise ImportError(
            "Motion helpers require pyproj: pip install 'sameer-graph-lib[motion]'"
        ) from exc
    return pyproj
def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Motion helpers require numpy: pip install 'sameer-graph-lib[motion]'"
        ) from exc
    return np
def _haversine_m(p1, p2,geojson=True) -> float:
    pyproj=_require_pyproj()
    geodesic = pyproj.Geod(ellps='WGS84')
    if geojson:
        u,v, distance_m = geodesic.inv(p1[0], p1[1], p2[0], p2[1])
    else:
        u,v, distance_m = geodesic.inv(p1[1], p1[0], p2[1], p2[0])
    return distance_m/1000.0
def get_max_speed(polyline, interval_seconds: float, smooth: bool = True,geojson=True,precision=5,max_bucket=5) -> float:
    decoded=decode_polyline(polyline_str=polyline, precision=precision, geojson=geojson)
    if len(decoded) < 2:
        return {}
    speed_array=[]
    max_speed={}
    if len(decoded) <= max_bucket + 1:
        for j in range(len(decoded)-1):
            dist=_haversine_m(decoded[j], decoded[j+1],geojson=geojson)
            speed=dist*3600/interval_seconds
            speed_array.append(speed)
        max_speed[1]=max(speed_array)
        max_speed['all']=max(speed_array)
        return max_speed
    for i in range(1,max_bucket+1):
        speed_array1=[]
        for j in range(len(decoded)-1-i):
            dist=_haversine_m(decoded[j], decoded[j+i],geojson=geojson)
            speed=dist*3600/(interval_seconds*i)
            speed_array1.append(speed)
            speed_array.append(speed)
        max_speed[i]=max(speed_array1)
    max_speed['all']=max(speed_array)
    return max_speed

def max_accelaration(polyline, interval_seconds: float, smooth: bool = True,geojson=True,precision=5) -> float:
    accelaration_array=[]
    decoded=decode_polyline(polyline_str=polyline, precision=precision, geojson=geojson)
    if len(decoded) < 3:
        return {}
    for i in range(len(decoded)-2):
        dist1=_haversine_m(decoded[i], decoded[i+1],geojson=geojson)
        dist2=_haversine_m(decoded[i+1], decoded[i+2],geojson=geojson)
        speed1=dist1*3600/interval_seconds
        speed2=dist2*3600/interval_seconds
        accelaration=(speed2-speed1)/interval_seconds
        accelaration_array.append(accelaration)
    return {'max':max(accelaration_array),'min':min(accelaration_array)}


# ---------------------------------------------------------------------------
# New helpers below: m/s and m/s^2 outputs. They use _geodesic_m (true
# meters) so they stay independent of _haversine_m's km return value.
# ---------------------------------------------------------------------------


def _geodesic_m(p1, p2, geojson: bool = True) -> float:
    """Geodesic distance in meters between two points."""
    pyproj = _require_pyproj()
    geodesic = pyproj.Geod(ellps="WGS84")
    if geojson:
        _, _, d = geodesic.inv(p1[0], p1[1], p2[0], p2[1])
    else:
        _, _, d = geodesic.inv(p1[1], p1[0], p2[1], p2[0])
    return d


def _decode_or_passthrough(polyline, geojson: bool, precision: int) -> list:
    """Accept either an encoded polyline string or a sequence of points."""
    if isinstance(polyline, str):
        return decode_polyline(polyline_str=polyline, precision=precision, geojson=geojson)
    return list(polyline)


def _validate_timed(points: Sequence[TimedPoint], min_points: int) -> None:
    if len(points) < min_points:
        raise ValueError(
            f"timestamped polyline needs at least {min_points} points, "
            f"got {len(points)}"
        )
    for i in range(len(points) - 1):
        if points[i + 1][2] <= points[i][2]:
            raise ValueError(
                f"timestamps must be strictly increasing; "
                f"got t[{i}]={points[i][2]} >= t[{i+1}]={points[i + 1][2]}"
            )


def compute_speeds(
    polyline,
    interval_seconds: float,
    geojson: bool = True,
    precision: int = 5,
) -> list[float]:
    """Return instantaneous speeds (m/s) between consecutive points.

    Length is ``n - 1``. ``polyline`` may be an encoded string or a sequence
    of points already in the order set by ``geojson``.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    decoded = _decode_or_passthrough(polyline, geojson, precision)
    if len(decoded) < 2:
        raise ValueError(f"polyline needs at least 2 points, got {len(decoded)}")
    return [
        _geodesic_m(decoded[i], decoded[i + 1], geojson=geojson) / interval_seconds
        for i in range(len(decoded) - 1)
    ]


def compute_accelerations(
    polyline,
    interval_seconds: float,
    geojson: bool = True,
    precision: int = 5,
) -> list[float]:
    """Return accelerations (m/s^2) between consecutive speed samples.

    Length is ``n - 2``. Positive values are acceleration, negative values
    are deceleration.
    """
    speeds = compute_speeds(polyline, interval_seconds, geojson=geojson, precision=precision)
    if len(speeds) < 2:
        raise ValueError("polyline needs at least 3 points to compute acceleration")
    return [
        (speeds[i + 1] - speeds[i]) / interval_seconds
        for i in range(len(speeds) - 1)
    ]


def smooth_speeds_median(
    polyline,
    interval_seconds: float,
    window_size: int = 7,
    geojson: bool = True,
    precision: int = 5,
) -> list[float]:
    """Return speeds (m/s) with outliers nullified via sliding-window median.

    Each sample is replaced by the median of itself and its
    ``window_size - 1`` neighbours (edge-padded at the boundaries). A median
    is robust: a single bad GPS reading is replaced by its neighbourhood's
    middle value, instead of being smeared across surrounding samples
    (which is what FFT or polynomial smoothers would do).

    Default ``window_size=7`` handles route-shape polylines where outliers
    can come in clusters of 2–3 consecutive samples; for clean GPS traces
    where outliers are isolated, ``window_size=3`` or ``5`` is enough.
    """
    np = _require_numpy()
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    speeds = compute_speeds(polyline, interval_seconds, geojson=geojson, precision=precision)
    if len(speeds) < window_size:
        return speeds

    from numpy.lib.stride_tricks import sliding_window_view

    half = window_size // 2
    arr = np.asarray(speeds, dtype=float)
    padded = np.pad(arr, half, mode="edge")
    windows = sliding_window_view(padded, window_size)
    return np.median(windows, axis=1).tolist()


def dominant_frequency_hz(
    polyline,
    interval_seconds: float,
    geojson: bool = True,
    precision: int = 5,
) -> float:
    """Return the dominant non-DC frequency (Hz) of the speed signal."""
    np = _require_numpy()
    speeds = compute_speeds(polyline, interval_seconds, geojson=geojson, precision=precision)
    if len(speeds) < 2:
        return 0.0

    spectrum = np.abs(np.fft.rfft(speeds))
    freqs = np.fft.rfftfreq(len(speeds), d=interval_seconds)
    spectrum[0] = 0
    if spectrum.max() == 0:
        return 0.0
    return float(freqs[int(spectrum.argmax())])


# ---------------------------------------------------------------------------
# Timestamped polylines: each point is (lat, lng, t_seconds)
# ---------------------------------------------------------------------------


def compute_speeds_from_timestamps(points: Sequence[TimedPoint]) -> list[float]:
    """Return speeds (m/s) using each segment's actual delta-t."""
    _validate_timed(points, min_points=2)
    return [
        _geodesic_m(points[i], points[i + 1], geojson=False)
        / (points[i + 1][2] - points[i][2])
        for i in range(len(points) - 1)
    ]


def compute_accelerations_from_timestamps(points: Sequence[TimedPoint]) -> list[float]:
    """Return accelerations (m/s^2) for a non-uniformly-sampled trace.

    For each interior point ``i+1`` we take ``(v[i+1] - v[i]) / dt`` where
    ``dt = (segment_i_duration + segment_{i+1}_duration) / 2``.
    """
    _validate_timed(points, min_points=3)
    speeds = compute_speeds_from_timestamps(points)
    out = []
    for i in range(len(speeds) - 1):
        dt_i = points[i + 1][2] - points[i][2]
        dt_next = points[i + 2][2] - points[i + 1][2]
        out.append((speeds[i + 1] - speeds[i]) / ((dt_i + dt_next) / 2))
    return out


def detect_sudden_braking_from_timestamps(
    points: Sequence[TimedPoint],
    threshold_mps2: float = -3.0,
) -> list[dict]:
    """Return sudden-braking events for a timestamped trace.

    Each event is ``{index, time_s, deceleration_mps2}`` where ``index``
    is the index in ``points`` of the boundary sample.
    """
    if threshold_mps2 >= 0:
        raise ValueError("threshold_mps2 must be negative (it is a deceleration)")
    accels = compute_accelerations_from_timestamps(points)
    return [
        {
            "index": i + 1,
            "time_s": points[i + 1][2],
            "deceleration_mps2": a,
        }
        for i, a in enumerate(accels)
        if a <= threshold_mps2
    ]


def resample_polyline(
    points: Sequence[TimedPoint],
    interval_seconds: float,
) -> list[Point]:
    """Resample a timestamped polyline to a uniform time grid (linear interp).

    Returns ``[(lat, lng), ...]``. Pass to :func:`compute_speeds` with
    ``geojson=False`` to use FFT smoothing on real GPS logs.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    _validate_timed(points, min_points=2)

    t_start = points[0][2]
    t_end = points[-1][2]
    n_steps = int((t_end - t_start) / interval_seconds) + 1

    out: list[Point] = []
    j = 0
    for k in range(n_steps):
        t = t_start + k * interval_seconds
        while j < len(points) - 2 and points[j + 1][2] < t:
            j += 1
        t1, t2 = points[j][2], points[j + 1][2]
        alpha = 0.0 if t2 == t1 else (t - t1) / (t2 - t1)
        lat = points[j][0] + alpha * (points[j + 1][0] - points[j][0])
        lng = points[j][1] + alpha * (points[j + 1][1] - points[j][1])
        out.append((lat, lng))
    return out
