import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
h3 = pytest.importorskip("h3")

from sameer_graph_lib import HexMetricGraph

CENTRE = h3.latlng_to_cell(12.935, 77.624, 8)
CELLS = sorted(h3.grid_disk(CENTRE, 1))


def make_frame(grains=True, cells=CELLS):
    """One hex column plus metric columns - no pickup, no drop, no edges."""
    rows = []
    for index, cell in enumerate(cells, start=1):
        buckets = ([(w, h) for w in ("weekday", "weekend") for h in (8, 9, 18)]
                   if grains else [(None, None)])
        for week_period, hour in buckets:
            row = {
                "hexid": cell,
                "orders": 100.0 * index,
                "requests": 200.0 * index,
                "unfulfilled": float(index),
                "avg_distance": 2.0 + index,
                "avg_duration": 10.0 + index,
            }
            if grains:
                row["week_period"], row["hour"] = week_period, hour
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def hex_graph():
    return HexMetricGraph(make_frame(), hex_col="hexid",
                          grain_cols=("week_period", "hour"))


def test_one_hex_column_is_enough(hex_graph):
    assert len(hex_graph) == len(CELLS)
    assert set(hex_graph.cells) == set(CELLS)
    assert hex_graph.graph.graph.number_of_edges() > 0      # edges came from H3
    assert "orders" in hex_graph.schema.sum_metrics
    assert "avg_distance" in hex_graph.schema.mean_metrics
    assert "hexid" not in hex_graph.metrics                 # the id is not a metric


def test_metrics_land_on_the_cells(hex_graph):
    frame = make_frame()
    for cell in hex_graph.cells:
        rows = frame[frame["hexid"] == cell]
        assert hex_graph.value(cell, "orders") == pytest.approx(rows["orders"].sum())
    weights = frame["requests"]
    expected = float((frame["avg_distance"] * weights).sum() / weights.sum())
    assert hex_graph.total_summary()["avg_distance"] == pytest.approx(expected)
    assert hex_graph.total_summary()["orders"] == pytest.approx(frame["orders"].sum())


def test_grain_filtering_works_per_cell(hex_graph):
    cell = hex_graph.cells[0]
    everything = hex_graph.value(cell, "orders")
    morning = hex_graph.value(cell, "orders", hour=[8, 9])
    assert morning < everything
    assert hex_graph.summary(cell, hour=8, week_period="weekday")["row_count"] == 1
    with pytest.raises(KeyError):
        hex_graph.value(cell, "orders", hour=99)


def test_a_frame_with_no_grain_still_works():
    flat = HexMetricGraph(make_frame(grains=False), hex_col="hexid")
    assert flat.schema.grain_names == ["__all__"]
    assert len(flat) == len(CELLS)
    assert flat.value(flat.cells[0], "orders") > 0


def test_frames_and_rankings(hex_graph):
    frame = hex_graph.frame()
    assert len(frame) == len(CELLS)
    assert {"hex", "rows", "degree", "orders", "lat", "lng"} <= set(frame.columns)
    assert frame["orders"].is_monotonic_decreasing            # sorted by value metric

    top = hex_graph.top_cells("orders", top=3)
    assert len(top) == 3
    assert list(top["orders"]) == sorted(top["orders"], reverse=True)
    assert top.iloc[0]["orders"] == pytest.approx(max(
        hex_graph.value(cell, "orders") for cell in hex_graph.cells))


def test_topology_choices():
    frame = make_frame()
    attached = HexMetricGraph(frame, hex_col="hexid", grain_cols=("week_period", "hour"))
    adjacent = HexMetricGraph(frame, hex_col="hexid", grain_cols=("week_period", "hour"),
                              topology="adjacent")
    bare = HexMetricGraph(frame, hex_col="hexid", grain_cols=("week_period", "hour"),
                          topology="none")
    assert attached.graph.graph.number_of_edges() > 0
    assert adjacent.graph.graph.number_of_edges() > 0
    assert bare.graph.graph.number_of_edges() == 0
    assert len(bare) == len(CELLS)                            # cells still stored
    for u, v in adjacent.graph.graph.edges():
        assert adjacent.distance(u, v) <= 1                   # only real neighbours
    with pytest.raises(ValueError):
        HexMetricGraph(frame, hex_col="hexid", topology="magic")


def test_node_values_feed_the_existing_selection(hex_graph):
    # the cell metric is written onto the AffinityGraph node, so the library's
    # own corridor logic can weight by it
    for cell in hex_graph.cells:
        assert hex_graph.graph.graph.nodes[cell]["value"] == pytest.approx(
            hex_graph.value(cell, hex_graph.value_metric))
    corridor = hex_graph.corridor(0.8)
    assert 0 < len(corridor) <= len(hex_graph)
    stats = hex_graph.corridor_stats(0.8)
    assert stats["value_coverage_pct"] >= 80
    assert set(corridor) <= set(hex_graph.cells)


def test_neighbours_and_distance(hex_graph):
    cell = hex_graph.cells[0]
    assert all(n in hex_graph for n in hex_graph.neighbors(cell))
    assert hex_graph.distance(cell, cell) == 0
    assert hex_graph.distance(CELLS[0], CELLS[1]) >= 1
    with pytest.raises(KeyError):
        hex_graph.neighbors("nope")


def test_one_graph_per_group():
    frame = make_frame()
    frame["city"] = ["north" if i % 2 else "south" for i in range(len(frame))]
    graphs = HexMetricGraph.per_group(frame, "city", hex_col="hexid",
                                      grain_cols=("week_period", "hour"))
    assert set(graphs) == {"north", "south"}
    assert all(isinstance(g, HexMetricGraph) for g in graphs.values())
    total = sum(g.total_summary()["orders"] for g in graphs.values())
    assert total == pytest.approx(frame["orders"].sum())
    with pytest.raises(KeyError):
        HexMetricGraph.per_group(frame, "missing", hex_col="hexid")


def test_bad_input_is_rejected():
    frame = make_frame()
    with pytest.raises(KeyError):
        HexMetricGraph(frame, hex_col="not_a_column")
    with pytest.raises(ValueError):
        HexMetricGraph(pd.DataFrame({"hexid": ["not-an-h3-cell"], "orders": [1.0]}),
                       hex_col="hexid")
    with pytest.raises(ValueError):
        HexMetricGraph(pd.DataFrame({"hexid": [None], "orders": [1.0]}), hex_col="hexid")
    with pytest.raises(KeyError):
        HexMetricGraph(frame, hex_col="hexid", grain_cols=("nope",))
    with pytest.raises(ValueError):
        HexMetricGraph(None)


def test_describe_and_stats(hex_graph):
    stats = hex_graph.get_graph_stats()
    assert stats["num_nodes"] == len(CELLS)
    assert stats["resolution"] == 8
    assert stats["value_metric"] == "orders"
    text = hex_graph.describe()
    assert "cells" in text and "resolution" in text


def test_plots(hex_graph):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fig in (hex_graph.plot_cells(),
                hex_graph.plot_cells(metric="requests", highlight=hex_graph.corridor(0.8)),
                hex_graph.plot_cells(hour=[8, 9], show_labels=True, show_edges=False),
                hex_graph.plot_graph(show_labels=False),
                hex_graph.plot_profile(hex_graph.cells[0])):
        assert fig.__class__.__name__ == "Figure"
        plt.close(fig)

    with pytest.raises(KeyError):
        hex_graph.plot_cells(metric="nope")


def test_metrics_can_be_chosen():
    frame = make_frame()
    frame["note"] = "not a number"
    everything = HexMetricGraph(frame, hex_col="hexid",
                                grain_cols=("week_period", "hour"))
    assert {"orders", "requests", "unfulfilled"} <= set(everything.schema.sum_metrics)

    picked = HexMetricGraph(frame, hex_col="hexid", grain_cols=("week_period", "hour"),
                            metrics=["orders", "avg_distance"])
    assert picked.schema.sum_metrics == ["orders"]
    assert "avg_distance" in picked.schema.mean_metrics     # classified as an average
    assert "requests" not in picked.metrics                 # not asked for
    assert "unfulfilled" not in picked.metrics
    assert picked.value_metric == "orders"
    # the numbers still match the frame
    cell = picked.cells[0]
    rows = frame[frame["hexid"] == cell]
    assert picked.value(cell, "orders") == pytest.approx(rows["orders"].sum())
    with pytest.raises(KeyError):
        picked.value(cell, "requests")                      # deliberately absent


def test_metric_selection_rejects_bad_input():
    frame = make_frame()
    frame["note"] = "text"
    with pytest.raises(KeyError):
        HexMetricGraph(frame, hex_col="hexid", metrics=["nope"])
    with pytest.raises(TypeError):
        HexMetricGraph(frame, hex_col="hexid", metrics=["note"])
    with pytest.raises(ValueError):
        HexMetricGraph(frame, hex_col="hexid", metrics=[])


def test_explicit_sum_and_mean_split_wins():
    graph = HexMetricGraph(make_frame(), hex_col="hexid",
                           grain_cols=("week_period", "hour"),
                           metrics=["orders", "requests"],
                           sum_metrics=["orders"], mean_metrics=["requests"])
    assert graph.schema.sum_metrics == ["orders"]
    assert "requests" in graph.schema.mean_metrics


def test_add_frame_grows_the_graph():
    first, rest = CELLS[:3], CELLS[3:]
    graph = HexMetricGraph(make_frame(cells=first), hex_col="hexid",
                           grain_cols=("week_period", "hour"))
    assert len(graph) == len(first)
    before_edges = graph.graph.graph.number_of_edges()

    added = graph.add_frame(make_frame(cells=rest))
    assert sorted(added) == sorted(rest)
    assert len(graph) == len(CELLS)
    assert graph.graph.graph.number_of_edges() > before_edges   # attached, not orphaned
    for cell in added:
        assert graph.graph.graph.degree(cell) > 0
        assert graph.graph.graph.nodes[cell]["value"] == pytest.approx(
            graph.value(cell, graph.value_metric))


def test_add_frame_accumulates_into_a_known_cell():
    cell = CELLS[0]
    graph = HexMetricGraph(make_frame(cells=[cell]), hex_col="hexid",
                           grain_cols=("week_period", "hour"))
    before = graph.value(cell, "orders")
    rows = graph.row_counts[cell]

    added = graph.add_frame(make_frame(cells=[cell]))
    assert added == []                                      # nothing new to attach
    assert len(graph) == 1
    assert graph.value(cell, "orders") == pytest.approx(before * 2)
    assert graph.row_counts[cell] == rows * 2


def test_add_cell_takes_values_directly():
    graph = HexMetricGraph(make_frame(cells=CELLS[:3]), hex_col="hexid",
                           grain_cols=("week_period", "hour"))
    fresh = CELLS[-1]
    assert fresh not in graph
    graph.add_cell(fresh, orders=250.0, requests=500.0)
    assert fresh in graph
    assert graph.value(fresh, "orders") == pytest.approx(250.0)
    assert graph.graph.graph.degree(fresh) > 0


def test_add_frame_works_without_grains():
    graph = HexMetricGraph(make_frame(grains=False, cells=CELLS[:2]), hex_col="hexid")
    graph.add_frame(make_frame(grains=False, cells=CELLS[2:4]))
    assert len(graph) == 4


def test_short_cell_labels_are_distinguishable():
    from sameer_graph_lib.hex_metric_viz import short_cell

    labels = {short_cell(cell) for cell in CELLS}
    assert len(labels) == len(CELLS)                        # the trailing f runs are gone
    assert all(not label.endswith("fff") for label in labels)
    assert short_cell("ffffffff") == "ffff"                 # degenerate id still works


def test_plot_steps(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = HexMetricGraph(make_frame(cells=CELLS[:4]), hex_col="hexid",
                           grain_cols=("week_period", "hour"))
    fig = graph.plot_steps(metric="orders", ncols=2)
    assert len(fig.axes) >= len(CELLS[:4])
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles[0].startswith("1. +") and "first cell" in titles[0]
    path = tmp_path / "steps.png"
    fig.savefig(path, dpi=80, bbox_inches="tight")
    assert path.exists() and path.stat().st_size > 0
    plt.close(fig)

    added = graph.add_frame(make_frame(cells=CELLS[4:6]))
    fig = graph.plot_steps(cells=added)                     # only the new ones
    assert len([ax for ax in fig.axes if ax.get_title()]) == len(added)
    plt.close(fig)

    with pytest.raises(KeyError):
        graph.plot_steps(cells=["not-in-the-graph"])
