# H3 Array Graph Example

This example builds a `HexGraph` from the provided comma-separated H3 cells and saves a QC image.

Run:

```powershell
uv run --extra plot python examples\create_h3_array_graph.py
```

Output:

- `h3_array_graph.png`
- `h3_array_cells.png`
- `h3_array_cells_basemap.png` when run with the `geo` extra
- `h3_array_convex_hull.geojson`
- `h3_array_insertion_steps.csv`
- `h3_array_insertion_steps.md`
- `h3_array_steps_01_08.png`, `h3_array_steps_09_16.png`, ...

The step logs include the full H3 hex name, insertion action, nearest nodes,
distance, edges added, edges removed, and edges rerouted.

For the basemap image, run:

```powershell
uv run --extra plot --extra geo python examples\create_h3_array_graph.py
```

## Pandas row graph example

To create one separate `HexGraph` for each DataFrame row that contains a list
of H3 cells, run:

```powershell
python -m pip install pandas
uv run --extra plot python examples\create_pandas_row_graphs.py
```

This saves `pandas_row_graph_0.png`, `pandas_row_graph_1.png`, ... and stores
each row's independent graph in a `graph` column.

## Route flow example (pickup -> drop graph)

`route_flow_example.ipynb` is a notebook, with its outputs saved, that walks
through the whole route stack on a synthetic frame shaped like a real export
(`pickup_cluster`, `drop_cluster`, `week_period`, `hour` plus metric columns):

1. build the graph from a DataFrame in one call, and see which columns the
   schema decided to sum and which to average
2. rank where a cluster's orders come from and where they go
3. plot that flow with a separate top-k per level, several variables at a time
   as tables on the nodes and edges, choosing whether the top-k measures the
   biggest routes (`rank_by="edge"`) or the biggest clusters (`rank_by="node"`)
4. filter the graph on edge or node metrics, either into a new graph (`keep`)
   or by pruning in place (`cut`), including inside the plot itself
5. reachability: which clusters can reach a given one inside a travel-time
   budget and a hop cap, the routes they take, and the same as a plot
6. the supporting views: partners, week_period x hour profile, pickup x drop
   matrix, the whole graph as a ranked ring
7. the stateless DataFrame plotter: multi-column panels, multi-axis charts,
   distribution checks and group comparisons
8. `HexMetricGraph`: a graph built from a single H3 cell column, with the
   metrics on the nodes and the edges from the H3 array logic, when the data
   has no pickup/drop pairs at all

Open it with Jupyter, or just read it on GitHub - every cell already shows its
output:

```powershell
uv run --extra analysis jupyter lab examples
oute_flow_example.ipynb
```

## A cluster that feeds itself

`self_loop_flow.py` covers rows whose pickup and drop are the same cluster - a
trip that starts and ends in the same place, often the largest single route the
cluster has.

```powershell
uv run --extra analysis python examples\self_loop_flow.py
```

It prints the arithmetic and saves two images:

- `self_loop_counted.png` - the default. The loop is not drawn, but it costs no
  place in the top x and `rest=True` carries its volume, so the drawn routes
  plus the rest come back to the cluster's real total (`12,000 + 18,840 =
  30,840` in, `2,100 + 18,000 = 20,100` out)
- `self_loop_drawn.png` - `exclude_self_loops=False`, where the loop is a ring
  on the cluster it loops on with its own metric table, and does take a place in
  the top x

- `self_loop_one_side.png` - the two sides judge the loop separately, each
  against its own top x, so a loop that is large beside a cluster's drops but
  small beside its sources is drawn downstream only

The script also shows what that costs: at `upstream=3` with the loop drawn you
see two outside clusters rather than three, so raise the top x by one if you
want the same number of partners.

## Hex metric graph images

Built from a single H3 column with `HexMetricGraph` (see the README section
*Hex metric graphs*). Regenerate any of them with one call, for example:

```python
hg.plot_steps(metric="orders").savefig("hex_metric_steps.png", dpi=140,
                                       bbox_inches="tight")
```

- `hex_metric_steps.png` - each cell joining the graph in turn, in red, with its
  value underneath and the grid distance on every edge
- `hex_metric_steps_added.png` - the same for a second batch of cells added to
  an existing graph with `add_frame`
- `hex_metric_cells.png` - the finished cells shaded by `orders`, with the 80%
  corridor outlined
- `hex_metric_cells_peak.png` - the same metric for the 08-09 window only
