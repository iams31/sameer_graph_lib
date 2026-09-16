import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sameer_graph_lib import RouteExplorer, RouteGraph
from sameer_graph_lib import route_viz


def make_frame():
    edges = (
        [("A1", f"B{i}") for i in range(1, 5)]
        + [(f"C{i}", "A1") for i in range(1, 5)]
        + [(f"D{i}", "C1") for i in range(1, 3)]
        + [("A1", "A2"), ("A2", "A1")]
    )
    rows = []
    for scale, (pickup, drop) in enumerate(edges, start=1):
        for week_period in ("weekday", "weekend"):
            for hour in (8, 9, 18):
                rows.append({
                    "pickup_cluster": pickup,
                    "drop_cluster": drop,
                    "week_period": week_period,
                    "hour": hour,
                    "orders": 25.0 * scale,
                    "requests": 50.0 * scale,
                    "accepted_orders": 10.0 * scale,
                    "avg_distance": 2.0 + scale,
                    "avg_duration": 12.0 + scale,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def graph():
    return RouteGraph.from_dataframe(make_frame(), grain_cols=("week_period", "hour"))


def close(fig):
    assert fig.__class__.__name__ == "Figure"
    plt.close(fig)


def test_flow_layout_places_levels_left_to_right(graph):
    sub = graph.flow_subgraph("A1", upstream=[2, 1], downstream=2)
    pos = route_viz.flow_layout(sub)
    assert pos["A1"][0] == 0
    for node, data in sub.nodes(data=True):
        if data["side"] == "source":
            assert pos[node][0] < 0
        elif data["side"] == "drop":
            assert pos[node][0] > 0
    # every node in a level shares one x
    xs = {data["level"]: set() for _, data in sub.nodes(data=True)}
    for node, data in sub.nodes(data=True):
        xs[data["level"]].add(pos[node][0])
    assert all(len(values) == 1 for values in xs.values())


def test_plot_flow_variants(graph):
    close(graph.plot_flow("A1"))
    close(graph.plot_flow("A1", upstream=[3, 2], downstream=[2, 1]))
    close(graph.plot_flow(["A1", "C1"], upstream=2, downstream=2, include_cross_edges=True))
    close(graph.plot_flow("A1", upstream=0, downstream=3))
    close(graph.plot_flow("A1", layout="spring"))
    close(graph.plot_flow("A1", per_parent=False, upstream=[4, 2]))


def test_plot_flow_can_show_a_different_variable(graph):
    fig = graph.plot_flow("A1", metric="orders", edge_metric="speed",
                          node_metric="accepted_orders", value_format="{:,.2f}",
                          hour=[8, 9])
    title = fig.axes[0].get_title()
    assert "ranked by orders" in title
    assert "showing accepted_orders" in title
    assert "hour=[8, 9]" in title
    plt.close(fig)


def test_plot_flow_rejects_impossible_requests(graph):
    with pytest.raises(ValueError):
        graph.plot_flow("A1", layout="hyperbolic")
    with pytest.raises(ValueError):
        route_viz.geo_layout(["A1"])          # not H3 cells
    with pytest.raises(KeyError):
        graph.plot_flow("A1", metric="nope")


def test_other_route_plots(graph):
    close(graph.plot_partners("A1", top=3))
    close(graph.plot_partners("D1", top=3))           # a node with no sources
    close(graph.plot_profile(pickup="A1", drop="B1"))
    close(graph.plot_profile(node="A1", metric="speed"))
    close(graph.plot_matrix(top=5))
    close(graph.plot_graph(top_routes=6))
    close(graph.plot_graph(layout="circular"))
    with pytest.raises(ValueError):
        graph.plot_profile()


def test_explorer_plots_inherit_metric_and_grain(tmp_path):
    explorer = RouteExplorer(make_frame(), grain_cols=("week_period", "hour")).where(hour=[8, 9]).using("speed")
    fig = explorer.plot("A1", upstream=2, downstream=2)
    assert "speed" in fig.axes[0].get_title()
    path = explorer.save(fig, tmp_path / "flow.png")
    assert (tmp_path / "flow.png").exists() and path.endswith("flow.png")

    close(explorer.plot_partners("A1", top=3))
    close(explorer.plot_matrix(top=4))
    close(explorer.plot_profile(cluster="A1"))


def test_explorer_dataframe_plots():
    pytest.importorskip("xarray")
    explorer = RouteExplorer(make_frame(), grain_cols=("week_period", "hour"))
    close(explorer.plot_columns(["orders", "speed"], x="hour",
                                group="week_period", agg="mean"))
    close(explorer.plot_distribution("speed"))
    close(explorer.compare_distributions("speed", "week_period"))


def test_scale_handles_flat_and_missing_values():
    assert route_viz._scale([], 1, 2).size == 0
    flat = route_viz._scale([5, 5, 5], 1, 3)
    assert set(np.round(flat, 6)) == {2.0}
    with_nan = route_viz._scale([np.nan, 1, 3], 1, 3)
    assert with_nan[0] == 1 and with_nan[-1] == 3
    assert route_viz._format(np.nan) == "n/a"
    assert route_viz._short("abcdefgh", 5) == "ab..."


def test_metric_table_aligns_columns():
    rows = [("orders", "62,709"), ("speed", "0.273")]
    table = route_viz.metric_table(rows, key_width=12, value_width=6, header="A1")
    lines = table.splitlines()
    assert lines[0].strip() == "A1"
    assert set(lines[1]) == {"-"}
    assert len({len(line) for line in lines[1:]}) == 1        # body lines line up
    assert lines[-1].startswith("speed")
    assert lines[-1].endswith("0.273")

    values_only = route_viz.metric_table(rows, value_width=6, show_names=False)
    assert "orders" not in values_only


def test_value_formatting_adapts_to_magnitude():
    assert route_viz._format(62709.4) == "62,709"
    assert route_viz._format(0.2731) == "0.273"
    assert route_viz._format(7.3812) == "7.38"
    assert route_viz._format(np.nan) == "n/a"
    assert route_viz._format(0.2731, "{:.1f}") == "0.3"
    assert route_viz._format(0.2731, {"speed": "{:.4f}"}, "speed") == "0.2731"
    assert route_viz._format(0.2731, {"other": "{:.4f}"}, "speed") == "0.273"


def test_node_and_edge_tables_carry_every_metric(graph):
    metrics = ["orders", "accepted_orders", "speed"]
    fig = graph.plot_flow("A1", upstream=2, downstream=2,
                          node_metrics=metrics, edge_metrics=["orders", "speed"])
    ax = fig.axes[0]
    tables = [child.get_text() for child in ax.texts]
    node_tables = [text for text in tables if "accepted_orders" in text]
    assert node_tables, "expected a table under every node"
    for text in node_tables:
        for metric in metrics:
            assert metric in text
        assert len(text.splitlines()) == len(metrics)
    edge_tables = [text for text in tables
                   if "orders" in text and "accepted_orders" not in text]
    assert edge_tables and all("speed" in text for text in edge_tables)
    plt.close(fig)


def test_single_metric_table_omits_the_name(graph):
    fig = graph.plot_flow("A1", upstream=1, downstream=1, node_metrics="orders")
    tables = [child.get_text() for child in fig.axes[0].texts]
    assert not any("orders" in text for text in tables)   # values only
    plt.close(fig)


def test_singular_aliases_still_work(graph):
    fig = graph.plot_flow("A1", upstream=1, downstream=1,
                          node_metric="accepted_orders", edge_metric="speed")
    assert "showing" in fig.axes[0].get_title()
    plt.close(fig)


def test_cross_edges_are_unlabelled_by_default(graph):
    # C1 -> A1 is a real route, but with downstream off it is not part of the
    # expansion, so it comes back as a cross edge between two selected nodes
    options = dict(upstream=1, downstream=0, include_cross_edges=True)
    sub = graph.flow_subgraph(["A1", "C1"], **options)
    crossings = [d["side"] for _, _, d in sub.edges(data=True)].count("cross")
    assert crossings >= 1

    plain = graph.plot_flow(["A1", "C1"], **options)
    labelled = graph.plot_flow(["A1", "C1"], label_cross_edges=True, **options)
    assert len(labelled.axes[0].texts) == len(plain.axes[0].texts) + crossings
    plt.close(plain)
    plt.close(labelled)


def test_figure_grows_with_the_tables(graph):
    sub = graph.flow_subgraph("A1", upstream=3, downstream=3)
    unclamped = (0.0, 0.0, 100.0, 100.0)
    one = route_viz._auto_figsize(sub, ["orders"], ["orders"], 8, unclamped)
    many = route_viz._auto_figsize(
        sub, ["orders", "accepted_orders", "requests", "speed"],
        ["orders", "speed"], 8, unclamped,
    )
    assert many[0] > one[0] and many[1] > one[1]

    deep = graph.flow_subgraph("A1", upstream=[5, 3], downstream=3)   # reaches D1/D2
    assert {d["level"] for _, d in deep.nodes(data=True)} > {d["level"] for _, d in sub.nodes(data=True)}
    assert (route_viz._auto_figsize(deep, ["orders"], ["orders"], 8, unclamped)[0]
            > one[0])                                   # an extra level widens it

    assert graph.plot_flow("A1", figsize=(9, 6)).get_size_inches().tolist() == [9, 6]
    plt.close("all")


def test_plot_accepts_filters(graph):
    cutoff = float(np.median([graph.edge_value(u, v, "speed") for u, v in graph.routes()]))
    fig = graph.plot_flow("A1", upstream=3, downstream=3,
                          edge_filter={"speed": (">", cutoff)})
    assert "filtered on speed" in fig.axes[0].get_title()
    plt.close(fig)

    fig = graph.plot_flow("A1", upstream=3, downstream=3, node_filter={"orders": 100})
    assert "filtered on orders" in fig.axes[0].get_title()
    plt.close(fig)


def test_plot_warns_when_a_filter_leaves_nothing(graph):
    with pytest.warns(UserWarning, match="No routes survived"):
        fig = graph.plot_flow("A1", upstream=2, downstream=2,
                              edge_filter={"orders": 10 ** 9})
    assert len(fig.axes[0].collections) >= 1          # the focus is still drawn
    plt.close(fig)


def test_explorer_plot_forwards_filters():
    explorer = RouteExplorer(make_frame(), grain_cols=("week_period", "hour"))
    cutoff = float(np.median([explorer.value(u, v, "speed") for u, v in explorer.routes]))
    close(explorer.plot("A1", upstream=2, downstream=2,
                        edge_filter={"speed": (">", cutoff)},
                        node_metrics=["orders", "speed"]))
    close(explorer.keep(edge_filter={"speed": (">", cutoff)}).plot("A1"))


def test_node_tables_show_what_the_cluster_sends(graph):
    """The node number is the cluster as a pickup: the orders it sends out."""
    fig = graph.plot_flow("A1", upstream=2, downstream=2, node_metrics=["orders"])
    tables = [t.get_text() for t in fig.axes[0].texts]
    expected = f"{graph.node_value('A1', 'orders', 'out'):,.0f}"
    assert any(expected in text for text in tables)
    # and not the inbound or the combined figure, which differ here
    assert graph.node_value("A1", "orders", "out") != graph.node_value("A1", "orders", "both")
    plt.close(fig)


def test_node_direction_is_selectable(graph):
    for direction in ("in", "out", "both"):
        fig = graph.plot_flow("A1", upstream=2, downstream=2,
                              node_metrics=["orders"], node_direction=direction)
        wanted = f"{graph.node_value('A1', 'orders', direction):,.0f}"
        assert any(wanted in t.get_text() for t in fig.axes[0].texts), direction
        plt.close(fig)
    with pytest.raises(ValueError, match="node_direction"):
        graph.plot_flow("A1", node_direction="sideways")


def test_the_ring_view_measures_the_same_way(graph):
    for direction in ("in", "out", "both"):
        close(graph.plot_graph(node_direction=direction))
