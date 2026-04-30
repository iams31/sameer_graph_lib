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
