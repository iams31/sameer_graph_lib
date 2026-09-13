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

## Install

From PyPI after publication:

```powershell
pip install sameer-graph-lib
```

With optional plotting and geospatial extras:

```powershell
pip install "sameer-graph-lib[plot,geo]"
```

For the route flow graph and the DataFrame plotter (pandas, numpy, xarray, matplotlib):

```powershell
pip install "sameer-graph-lib[analysis]"
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
- `RouteSchema` / `RouteTensor` / `RouteGraph`: grain-aware pickup -> drop graph where each edge carries a metric cube (see [Route flow graphs](#route-flow-graphs-pickup---drop)).
- `RouteExplorer`: the one-liner front door to the route graph - build from a DataFrame, rank partners, and plot the flow around a cluster.
- `plotter`: stateless DataFrame plotting - multi-column, multi-axis, multi-group, plus distribution checks.

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

## Route flow graphs (pickup -> drop)

The second half of the library works on tabular route data rather than H3
chains: one row per `pickup_cluster x drop_cluster x week_period x hour`, with
whatever metric columns you export. Every edge stores a **tensor** - a dense
cube over the grain dimensions - so sums stay exact, averages stay correctly
weighted, and two tensors can be merged without re-reading the source rows.

The metric columns are **yours**: the schema reads whatever numeric columns the
frame has. Names that look like averages (`avg_`, `_rate`, `pct`, `ratio`, ...)
are weighted-averaged, the rest are summed, and grain/id/categorical columns are
never treated as metrics. Pass `sum_metrics=` / `mean_metrics=` to override the
split, and `metric=` anywhere to choose the variable; with nothing named, calls
fall back to `schema.default_metric` (the first sum metric).

Install the extra:

```powershell
python -m pip install -e ".[analysis]"      # pandas, numpy, xarray, matplotlib
```

### Quick start

```python
from sameer_graph_lib import RouteExplorer

ex = RouteExplorer(df, metric="orders")   # metric= is optional

ex.sources("A1", top=5)     # where A1 orders come from
ex.drops("A1", top=5)       # where A1 orders go
ex.route("A1", "B1")        # every metric for one route
ex.summary("A1")            # every metric for one cluster
print(ex.describe())
```

`where` and `using` return a new view that shares the same graph, so you can
narrow the question without rebuilding anything:

```python
peak = ex.where(week_period="weekday", hour=[8, 9, 10]).using("speed")
peak.drops("A1", top=5)
```

### Plotting the flow around a cluster

`plot` puts the focus clusters in the middle, the clusters that feed them on
the left, and the clusters they feed on the right. The per-level top-k is the
main control: pass an int for one level, or a list for one number per level.

```python
# top 5 sources, then the top 3 sources of each of those, then the top 2;
# and on the other side, the top 4 drops and the top 2 of each.
fig = ex.plot("A1", upstream=[5, 3, 2], downstream=[4, 2])

# several focus clusters, plus the routes between the selected set
ex.plot(["A1", "C1"], upstream=3, downstream=3, include_cross_edges=True)

# rank by one variable, show several: every node and edge carries a table
ex.plot("A1", metric="orders",
        node_metrics=["orders", "accepted_orders", "requests", "speed"],
        edge_metrics=["orders", "speed", "avg_distance"],
        hour=[8, 9, 10])
```

`node_metrics` and `edge_metrics` take any number of variables. They are drawn
as a small aligned table under each node and on each edge, with the metric
names on the left and the values right-aligned; every table in one figure
shares its column widths, and the figure sizes itself so they do not collide.
Numbers are formatted per value by default (62,709 and 0.273 both read
correctly); pass `value_format="{:,.1f}"` for one format, or a dict such as
`value_format={"speed": "{:.3f}"}` per metric.

Other options: `size_metric=` (which variable drives node size),
`show_metric_names=False` (values only), `per_parent=False` (rank a whole level
globally instead of per parent), `min_value=`, `layout="layered" | "geo" |
"spring"` (`geo` places H3 cluster ids at their real coordinates),
`show_edge_values`, `label_cross_edges`, `curve`, `figsize`, and the colours.

The same expansion is available without plotting:

```python
ex.flow("A1", upstream=[5, 3], downstream=4)       # tidy DataFrame
ex.subgraph("A1", upstream=2, downstream=2)        # nx.DiGraph, levels on nodes
print(ex.tree("A1", upstream=[5, 3], downstream=4))
```

```text
A1  (orders)
  sources (orders coming in)
  |-- <- C1  [8,523.4]
  |   |-- <- D3  [8,802.4]
  |   +-- <- D2  [3,646.5]
  +-- <- C3  [7,041.1]
  drops (orders going out)
  |-- -> B4  [4,574.9]
  +-- -> B5  [4,446.4]
```

### Other route views

| Call | What you get |
| --- | --- |
| `ex.plot_partners("A1", top=10)` | top sources and top drops as back-to-back bars on one shared scale |
| `ex.plot_profile(cluster="A1")` | week_period x hour heatmap for a cluster or a route |
| `ex.plot_matrix(top=15)` | pickup x drop heatmap |
| `ex.plot_graph(top_routes=30)` | the whole graph as a ranked ring, sized and coloured by a metric |
| `ex.clusters_frame()` | per cluster inbound vs outbound volume and net balance |
| `ex.top_routes(20)` | routes ranked by any metric |
| `ex.matrix()` / `ex.frame()` | pivot table / long per-bucket frame |

The lower-level classes are public too: `RouteSchema` (grains and metric
columns), `RouteTensor` (one route's metric cube, with `summary`, `totals`,
`profile`, `merge`) and `RouteGraph` (the NetworkX graph, with `partners`,
`node_tensor`, `shortest_route`, `flow_subgraph`, `as_dataset`).

## DataFrame plotting

`sameer_graph_lib.plotter` is a stateless plotting layer for any DataFrame -
each function takes a frame and returns a figure.

```python
from sameer_graph_lib import plotter

# n columns, one panel each, test vs control overlaid in every panel
plotter.plot_columns(df, ["orders", "speed", "aor"], x="hour", group="variant",
                     agg="mean", kind="line", mode="grid")

# columns with different scales on one chart, one y-axis each
plotter.plot_multi_axis(df, ["orders", "speed"], x="hour", agg="mean")

# distribution of a column, with count/mean/median/sd/skew in the corner
plotter.plot_distribution(df, "speed", kind="hist", bins=40)

# the same distribution per group, one colour per line, on one chart
plotter.compare_distributions(df, "speed", group="variant", kind="kde", with_box=True)

# A/B readout: aggregate per group, optionally relative to a baseline
plotter.compare_groups(df, ["orders", "speed"], group="variant", agg="mean",
                       normalize_to="control")

plotter.describe_distribution(df, "speed", group="variant")   # stats table
plotter.plot_correlation(df, ["orders", "speed", "distance"])
```

`mode` is `grid` (one panel per column), `overlay` (one shared axes) or `twin`
(one y-axis per column). `kind` is `line`, `step`, `area`, `bar`, `barh`,
`scatter`, `hist`, `kde`, `box`, `violin` or `ecdf`. `group` splits every
series by a categorical column and gives each level its own colour. KDE and
ECDF are computed with numpy, so scipy is not required.

The explorer exposes the same helpers over its own long frame:

```python
ex.plot_columns(["orders", "speed"], x="hour", group="week_period", agg="mean")
ex.compare_distributions("speed", "week_period")
```

## Useful commands

```powershell
python -m pytest
python -m build
uv run --extra dev pytest -q
uv run --extra dev python -m build
uv run --extra dev python -m twine check dist/*
```

Build artifacts will appear in `dist/` after `python -m build`.

## Publish To PyPI

1. Build the package:

```powershell
uv run --extra dev python -m build
```

2. Validate the package metadata:

```powershell
uv run --extra dev python -m twine check dist/*
```

3. Upload to PyPI:

```powershell
uv run --extra dev python -m twine upload dist/*
```

After upload, users can install it with:

```powershell
pip install sameer-graph-lib
```

## Publish From GitHub

This repo also includes a Trusted Publishing workflow in
[.github/workflows/publish.yml](C:/Users/rrran/Desktop/sameer_graph_lib/.github/workflows/publish.yml:1).

To finish that setup:

1. Create the project on PyPI, or reserve the name `sameer-graph-lib`.
2. On PyPI, open the project settings and add a Trusted Publisher for:
   `owner`: `iams31`
   `repository`: `sameer_graph_lib`
   `workflow`: `publish.yml`
   `environment`: `pypi`
3. Create a GitHub Release, or run the workflow manually from the Actions tab.

After that, GitHub Actions can publish without storing a long-lived PyPI token.

Official references:

- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- Packaging guide upload flow: https://packaging.python.org/tutorials/packaging-projects/
