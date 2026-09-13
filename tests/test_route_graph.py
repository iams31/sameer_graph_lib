import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from sameer_graph_lib import RouteExplorer, RouteGraph, RouteSchema, RouteTensor


def make_frame():
    """A1 has sources C1..C3 and drops B1..B3; C1 in turn is fed by D1/D2."""
    rng = np.random.default_rng(7)
    edges = (
        [("A1", f"B{i}") for i in range(1, 4)]
        + [(f"C{i}", "A1") for i in range(1, 4)]
        + [(f"D{i}", "C1") for i in range(1, 3)]
    )
    rows = []
    for scale, (pickup, drop) in enumerate(edges, start=1):
        for week_period in ("weekday", "weekend"):
            for hour in (8, 9, 18):
                pings = 100.0 * scale
                rows.append({
                    "pickup_cluster": pickup,
                    "drop_cluster": drop,
                    "week_period": week_period,
                    "hour": hour,
                    "city": "blr",
                    "segment": "core",
                    "orders": pings * 0.5,
                    "requests": pings,
                    "accepted_orders": pings * 0.3,
                    "supply_count": float(scale),
                    "avg_distance": 2.0 + scale,
                    "avg_duration": 10.0 + scale,
                    "accept_rate": float(rng.uniform(0.3, 0.9)),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def graph():
    return RouteGraph.from_dataframe(make_frame())


def test_schema_infers_grains_and_metrics():
    schema = RouteSchema.from_frame(make_frame())
    assert schema.grain_names == ["week_period", "hour"]
    assert len(schema.grains["hour"]) == 24          # hours always span the full day
    assert schema.shape == (2, 24)
    assert "orders" in schema.sum_metrics
    assert schema.length_metric in schema.mean_metrics
    assert "speed" in schema.metric_names


def test_metrics_are_inferred_from_the_frame():
    frame = make_frame().rename(columns={"orders": "trips", "avg_distance": "avg_km"})
    schema = RouteSchema.from_frame(frame, length_metric="avg_km",
                                    ride_time_metric="avg_duration",
                                    weight_col="requests")
    # numeric columns become metrics; avg_/rate names are averaged, rest summed
    assert "trips" in schema.sum_metrics and "requests" in schema.sum_metrics
    assert "avg_km" in schema.mean_metrics and "accept_rate" in schema.mean_metrics
    # grains, ids and categoricals are never metrics
    for column in ("hour", "week_period", "pickup_cluster", "drop_cluster", "segment"):
        assert column not in schema.sum_metrics + schema.mean_metrics
    assert schema.default_metric == schema.sum_metrics[0]

    graph = RouteGraph.from_dataframe(frame, schema=schema)
    assert graph.resolve_metric() == schema.default_metric
    assert graph.rank_routes(top=1).columns.tolist()[2] == schema.default_metric


def test_explicit_metric_lists_win_over_inference():
    schema = RouteSchema.from_frame(make_frame(), sum_metrics=["orders"],
                                    mean_metrics=["avg_distance", "avg_duration"])
    assert schema.sum_metrics == ["orders"]
    assert schema.default_metric == "orders"


def test_schema_keeps_length_metrics_even_when_absent():
    frame = make_frame().drop(columns=["avg_distance"])
    schema = RouteSchema.from_frame(frame)
    tensor = RouteTensor.from_frame(schema, frame)
    assert np.isnan(tensor.summary()["speed"])       # no distance column, no speed


def test_tensor_sums_and_weighted_means_are_exact():
    frame = make_frame()
    schema = RouteSchema.from_frame(frame)
    tensor = RouteTensor.from_frame(schema, frame)
    summary = tensor.summary()

    weights = frame["requests"].clip(lower=0).where(lambda s: s > 0, 1.0)
    expected_mean = float((frame["avg_distance"] * weights).sum() / weights.sum())
    assert summary["orders"] == pytest.approx(frame["orders"].sum())
    assert summary["avg_distance"] == pytest.approx(expected_mean)
    assert summary["speed"] == pytest.approx(summary["avg_distance"] / summary["avg_duration"])
    assert tensor.row_total == len(frame)


def test_tensor_merge_matches_single_pass():
    frame = make_frame()
    schema = RouteSchema.from_frame(frame)
    merged = RouteTensor.from_frame(schema, frame.iloc[:40]).merge(
        RouteTensor.from_frame(schema, frame.iloc[40:])
    )
    whole = RouteTensor.from_frame(schema, frame)
    for metric in ("orders", "avg_distance", "speed"):
        assert merged.summary()[metric] == pytest.approx(whole.summary()[metric])


def test_grain_selection_narrows_the_totals(graph):
    everything = graph.route_summary("A1", "B1")
    morning = graph.route_summary("A1", "B1", hour=[8, 9])
    assert morning["orders"] < everything["orders"]
    assert graph.route_summary("A1", "B1", hour=8, week_period="weekday")["row_count"] == 1
    with pytest.raises(KeyError):
        graph.route_summary("A1", "B1", hour=99)
    with pytest.raises(KeyError):
        graph.route_summary("A1", "B1", nonsense=1)


def test_edge_and_node_queries(graph):
    assert graph.edge_length("A1", "B1") == pytest.approx(3.0)
    assert graph.edge_speed("A1", "B1") == pytest.approx(3.0 / 11.0)
    assert graph.edge_value("A1", "B1", "orders") > 0
    assert graph.node_value("A1", "orders", "out") == pytest.approx(
        sum(graph.edge_value("A1", drop, "orders") for drop in ("B1", "B2", "B3"))
    )
    with pytest.raises(KeyError):
        graph.edge_value("A1", "B1", "not_a_metric")


def test_partner_ranking_is_ordered(graph):
    drops = graph.top_drops("A1", top=2)
    assert [name for name, _ in drops] == ["B3", "B2"]        # scale grows with edge order
    sources = graph.top_sources("A1", top=2)
    assert [name for name, _ in sources] == ["C3", "C2"]
    frame = graph.partners_frame("A1", direction="in", top=3)
    assert list(frame["rank"]) == [1, 2, 3]
    assert frame["share"].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        graph.partners("A1", direction="sideways")


def test_flow_subgraph_levels_and_top_k(graph):
    sub = graph.flow_subgraph("A1", upstream=[2, 1], downstream=2)
    levels = {node: data["level"] for node, data in sub.nodes(data=True)}
    assert levels["A1"] == 0
    assert sorted(n for n, lv in levels.items() if lv == -1) == ["C2", "C3"]
    assert sorted(n for n, lv in levels.items() if lv == 1) == ["B2", "B3"]
    assert sub.graph["metric"] == "orders"

    deep = graph.flow_subgraph("A1", upstream=[3, 2], downstream=0)
    assert {n for n, d in deep.nodes(data=True) if d["level"] == -2} == {"D1", "D2"}
    assert not [n for n, d in deep.nodes(data=True) if d["level"] > 0]


def test_flow_level_plan_variants(graph):
    assert RouteGraph._level_plan(3) == [3]
    assert RouteGraph._level_plan([5, 3, 2]) == [5, 3, 2]
    assert RouteGraph._level_plan(None) == []
    assert RouteGraph._level_plan(0) == []
    assert RouteGraph._level_plan([4, 0, 9]) == [4]
    assert graph.flow_subgraph("A1", upstream=None, downstream=1).number_of_nodes() == 2


def test_flow_frame_and_tree(graph):
    frame = graph.flow_frame("A1", upstream=2, downstream=2, extra_metrics=["speed"])
    assert set(frame["side"]) == {"source", "drop"}
    assert "speed" in frame.columns
    assert len(frame) == 4

    tree = graph.flow_tree("A1", upstream=2, downstream=1)
    assert "sources (orders coming in)" in tree
    assert "drops (orders going out)" in tree
    assert "C3" in tree


def test_flow_supports_multiple_focus_and_grain(graph):
    sub = graph.flow_subgraph(["A1", "C1"], upstream=1, downstream=1)
    assert sub.graph["focus"] == ["A1", "C1"]
    peak = graph.flow_frame("A1", upstream=2, downstream=0, hour=[8, 9])
    full = graph.flow_frame("A1", upstream=2, downstream=0)
    assert peak["orders"].sum() < full["orders"].sum()
    with pytest.raises(KeyError):
        graph.flow_subgraph("NOPE")


def test_graph_level_tables(graph):
    nodes = graph.nodes_frame()
    assert set(nodes.columns) >= {"cluster", "out_orders", "in_orders", "net"}
    assert len(nodes) == graph.graph.number_of_nodes()

    routes = graph.rank_routes(top=3)
    assert len(routes) == 3
    assert routes["orders"].is_monotonic_decreasing

    matrix = graph.flow_matrix()
    assert matrix.loc["A1", "B1"] == pytest.approx(graph.edge_value("A1", "B1"))
    assert graph.total_summary()["orders"] > 0
    assert graph.get_graph_stats()["num_edges"] == len(graph.routes())
    assert graph.simple_graph().number_of_edges() == graph.graph.number_of_edges()
    assert graph.subgraph(["A1", "B1"]).graph.number_of_edges() == 1


def test_shortest_route_uses_length(graph):
    path, total = graph.shortest_route("D1", "B1")
    assert path[0] == "D1" and path[-1] == "B1"
    assert total == pytest.approx(
        sum(graph.edge_length(u, v) for u, v in zip(path, path[1:]))
    )


def test_xarray_views_when_available(graph):
    pytest.importorskip("xarray")
    dataset = graph.as_dataset()
    assert dataset.sizes["route"] == graph.graph.number_of_edges()
    assert graph.speed_profile("A1", "B1").shape == (2, 24)
    long = graph.to_frame()
    assert {"week_period", "hour", "row_count"} <= set(long.columns)


def test_explorer_reads_like_a_sentence():
    explorer = RouteExplorer(make_frame())
    assert len(explorer) == explorer.graph.graph.number_of_nodes()
    assert "A1" in explorer and "ZZ" not in explorer
    assert explorer.drops("A1", top=1)["partner"].iloc[0] == "B3"
    assert explorer.sources("A1", top=1)["partner"].iloc[0] == "C3"
    assert "clusters" in explorer.describe()
    assert explorer.route("A1", "B1")["orders"] > 0
    assert set(explorer.top_routes(2).columns) >= {"pickup_cluster", "drop_cluster"}


def test_explorer_views_are_sticky_but_isolated():
    explorer = RouteExplorer(make_frame())
    peak = explorer.where(hour=[8, 9]).using("speed")
    assert peak.metric == "speed"
    assert peak.grain == {"hour": [8, 9]}
    assert explorer.metric == "orders"          # original untouched
    assert explorer.grain == {}
    assert peak.graph is explorer.graph               # no rebuild
    assert peak.all_grains().grain == {}
    assert peak.drops("A1", top=1)[peak.metric].iloc[0] > 0
    with pytest.raises(KeyError):
        explorer.where(nope=1)
