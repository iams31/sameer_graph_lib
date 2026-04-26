"""Geospatial plotting helpers for H3 cells."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

from ._h3 import cell_to_boundary, cell_to_latlng, is_valid_cell


def normalize_cells(cells: str | Iterable[str]) -> list[str]:
    if isinstance(cells, str):
        if "," in cells:
            return [cell.strip() for cell in cells.split(",") if cell.strip()]
        return [cells]
    return [str(cell).strip() for cell in cells if str(cell).strip()]


def plot_h3_cells(
    cells: str | Iterable[str],
    *,
    title: str = "H3 Cells",
    ax=None,
    figsize: tuple[float, float] = (10, 8),
    face_color: str = "#3498db",
    edge_color: str = "#1f2d3d",
    selected_cells: str | Iterable[str] | None = None,
    selected_face_color: str = "#2ecc71",
    alpha: float = 0.45,
    linewidth: float = 1.2,
    show_labels: bool = True,
    label_full_hex: bool = False,
    show_centers: bool = False,
    equal_aspect: bool = True,
):
    """Plot one H3 cell or an iterable/comma string of H3 cells as polygons.

    Returns the Matplotlib ``Figure`` object. The axes use longitude on X and
    latitude on Y, so the plot can be inspected as a geospatial footprint.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    hexes = normalize_cells(cells)
    invalid = [cell for cell in hexes if not is_valid_cell(cell)]
    if invalid:
        raise ValueError(f"Invalid H3 cell(s): {', '.join(invalid[:3])}")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    selected = set(normalize_cells(selected_cells) if selected_cells is not None else [])
    all_lngs: list[float] = []
    all_lats: list[float] = []

    for cell in hexes:
        boundary = cell_to_boundary(cell)
        polygon_points = [(lng, lat) for lat, lng in boundary]
        all_lngs.extend(lng for lng, _ in polygon_points)
        all_lats.extend(lat for _, lat in polygon_points)

        patch = Polygon(
            polygon_points,
            closed=True,
            facecolor=selected_face_color if cell in selected else face_color,
            edgecolor=edge_color,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(patch)

        center_lat, center_lng = cell_to_latlng(cell)
        if show_centers:
            ax.scatter([center_lng], [center_lat], s=18, c=edge_color, zorder=3)

        if show_labels:
            label = cell if label_full_hex else cell[-6:]
            ax.text(
                center_lng,
                center_lat,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="#111111",
                zorder=4,
            )

    if all_lngs and all_lats:
        lng_margin = max((max(all_lngs) - min(all_lngs)) * 0.08, 0.001)
        lat_margin = max((max(all_lats) - min(all_lats)) * 0.08, 0.001)
        ax.set_xlim(min(all_lngs) - lng_margin, max(all_lngs) + lng_margin)
        ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    return fig


def cells_to_geodataframe(
    cells: str | Iterable[str],
    *,
    selected_cells: str | Iterable[str] | None = None,
):
    """Convert one or more H3 cells to a GeoPandas dataframe of polygons."""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise ImportError(
            "GeoPandas plotting requires the geo extra: "
            "pip install 'sameer-graph-lib[geo]'"
        ) from exc

    hexes = normalize_cells(cells)
    invalid = [cell for cell in hexes if not is_valid_cell(cell)]
    if invalid:
        raise ValueError(f"Invalid H3 cell(s): {', '.join(invalid[:3])}")

    selected = set(normalize_cells(selected_cells) if selected_cells is not None else [])
    rows = []
    for cell in hexes:
        boundary = cell_to_boundary(cell)
        polygon = Polygon([(lng, lat) for lat, lng in boundary])
        center_lat, center_lng = cell_to_latlng(cell)
        rows.append(
            {
                "h3_cell": cell,
                "selected": cell in selected,
                "center_lat": center_lat,
                "center_lng": center_lng,
                "geometry": polygon,
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def plot_h3_cells_map(
    cells: str | Iterable[str],
    *,
    selected_cells: str | Iterable[str] | None = None,
    column: str | None = "selected",
    title: str = "H3 Cells Map",
    ax=None,
    figsize: tuple[float, float] = (16, 16),
    alpha: float = 0.35,
    edge_color: str = "black",
    linewidth: float = 1.0,
    basemap: bool = True,
    basemap_source=None,
    legend: bool = True,
    hide_axes: bool = True,
):
    """Plot H3 cells with GeoPandas and an optional Contextily basemap.

    This mirrors the GeoPandas workflow:
    convert H3 polygons to ``EPSG:3857``, plot them, then add a web basemap.
    Returns the Matplotlib ``Figure`` object.
    """
    try:
        import contextily as cx
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Basemap plotting requires the geo extra: "
            "pip install 'sameer-graph-lib[geo]'"
        ) from exc

    gdf = cells_to_geodataframe(cells, selected_cells=selected_cells).to_crs(epsg=3857)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if hide_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plot_kwargs = {
        "ax": ax,
        "alpha": alpha,
        "edgecolor": edge_color,
        "linewidth": linewidth,
        "legend": legend,
    }
    if column and column in gdf.columns:
        plot_kwargs.update({"column": column, "categorical": True})
        if legend:
            plot_kwargs["legend_kwds"] = {"loc": "upper left"}
    else:
        plot_kwargs["color"] = "#3498db"

    gdf.plot(**plot_kwargs)

    if basemap:
        source = basemap_source or cx.providers.CartoDB.Positron
        cx.add_basemap(ax, crs=gdf.crs, source=source)

    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
