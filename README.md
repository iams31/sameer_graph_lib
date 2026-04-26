# sameer-graph-lib

`sameer-graph-lib` is an editable Python library for building H3-based route affinity graphs with NetworkX.

It turns H3 arrays, latitude/longitude sequences, and encoded polylines into connected hex chains, inserts them into a weighted graph, extracts high-affinity corridors, and decomposes the graph into a main trunk plus minor branches.

## Editable install

```powershell
python -m pip install -e ".[dev,plot]"
```

Because the install is editable, changes you make inside `src/sameer_graph_lib` are picked up immediately by Python without reinstalling.

If your machine uses `uv`, run commands through the managed environment:

```powershell
uv run --extra dev --extra plot python -c "import sameer_graph_lib; print(sameer_graph_lib.__version__)"
```

## Quick start

```python
from sameer_graph_lib import HexGraph

graph = HexGraph(hex_resolution=9)

route = graph.add_latlng_sequence([
    (12.9716, 77.5946),
    (12.9760, 77.5990),
])

print(graph.get_graph_stats())
selected = graph.get_appropriate_hexes(cutoff=0.8)
fig = graph.visualize_graph(title="80% compact cluster", highlight_hexes=selected)
print(graph.decompose_branches())
```

## Main APIs

- `SpatialIngestor`: converts H3 arrays, lat/lng sequences, and encoded polylines into contiguous H3 chains.
- `AffinityGraph`: NetworkX wrapper for array-based insertion, nearest attachment, affinity scoring, editing, and JSON persistence.
- `CorridorExtractor`: uses exact all-node Dijkstra selection to extract the most compact cluster covering a target percentage of graph traversal volume.
- `TopologyAnalyzer`: separates the main branch from residual minor branches.
- `HexGraph`: backwards-compatible convenience class for your original code style.

Graph creation follows the original per-node procedure: H3 arrays are normalized, then each hex is inserted with `add_node`/`add_hex`. Lat/lng sequences and encoded polylines are first converted into H3 arrays at the requested resolution, then inserted the same way.

For QC, use:

```python
fig = graph.visualize_graph(highlight_hexes=selected)
fig.savefig("graph_qc.png", dpi=150, bbox_inches="tight")

fig = graph.visualize_step_by_step(route[:5], labels=["A", "B", "C", "D", "E"])
fig.savefig("insertion_steps.png", dpi=150, bbox_inches="tight")
```

To plot actual H3 hex boundaries as geospatial polygons:

```python
from sameer_graph_lib import plot_h3_cells, plot_h3_cells_map

fig = plot_h3_cells("88618c4f29fffff", label_full_hex=True)
fig.savefig("single_h3_cell.png", dpi=150, bbox_inches="tight")

fig = graph.plot_h3_cells(highlight_hexes=selected, show_labels=False)
fig.savefig("h3_cell_footprint.png", dpi=150, bbox_inches="tight")

fig = plot_h3_cells_map(route, selected_cells=selected)
fig.savefig("h3_cell_basemap.png", dpi=150, bbox_inches="tight")
```

`plot_h3_cells_map` uses GeoPandas + Contextily. Install it with:

```powershell
python -m pip install -e ".[plot,geo]"
uv run --extra plot --extra geo python -c "from sameer_graph_lib import plot_h3_cells_map"
```

To get H3 centers and a convex hull:

```python
from sameer_graph_lib import getLatLng, h3_convex_hull

points = getLatLng(route)          # [(lat, lng), ...]
hull = h3_convex_hull(route)       # Shapely geometry in (lng, lat)
graph_hull = graph.convex_hull()   # Same, using graph nodes
```

## Useful commands

```powershell
python -m pytest
python -m build
uv run --extra dev pytest -q
uv run --extra dev python -m build
```

Build artifacts will appear in `dist/` after `python -m build`.
