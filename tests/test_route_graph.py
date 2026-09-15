import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from sameer_graph_lib import RouteExplorer, RouteGraph, RouteSchema, RouteTensor
from sameer_graph_lib.route_graph import looks_like_dimension


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
    return RouteGraph.from_dataframe(make_frame(), grain_cols=("week_period", "hour"))


def test_schema_infers_grains_and_metrics():
    schema = RouteSchema.from_frame(make_frame(), grain_cols=("week_period", "hour"))
    assert schema.grain_names == ["week_period", "hour"]
    assert len(schema.grains["hour"]) == 24          # hours always span the full day
    assert schema.shape == (2, 24)
    assert "orders" in schema.sum_metrics
    assert schema.length_metric in schema.mean_metrics
    assert "speed" in schema.metric_names


def test_metrics_are_inferred_from_the_frame():
    frame = make_frame().rename(columns={"orders": "trips", "avg_distance": "avg_km"})
    schema = RouteSchema.from_frame(frame, grain_cols=("week_period", "hour"), length_metric="avg_km",
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
    schema = RouteSchema.from_frame(make_frame(), grain_cols=("week_period", "hour"), sum_metrics=["orders"],
                                    mean_metrics=["avg_distance", "avg_duration"])
    assert schema.sum_metrics == ["orders"]
    assert schema.default_metric == "orders"


def test_schema_keeps_length_metrics_even_when_absent():
    frame = make_frame().drop(columns=["avg_distance"])
    schema = RouteSchema.from_frame(frame, grain_cols=("week_period", "hour"))
    tensor = RouteTensor.from_frame(schema, frame)
    assert np.isnan(tensor.summary()["speed"])       # no distance column, no speed


def test_tensor_sums_and_weighted_means_are_exact():
    frame = make_frame()
    schema = RouteSchema.from_frame(frame, grain_cols=("week_period", "hour"))
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
    schema = RouteSchema.from_frame(frame, grain_cols=("week_period", "hour"))
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
    explorer = RouteExplorer(make_frame(), grain_cols=("week_period", "hour"))
    assert len(explorer) == explorer.graph.graph.number_of_nodes()
    assert "A1" in explorer and "ZZ" not in explorer
    assert explorer.drops("A1", top=1)["partner"].iloc[0] == "B3"
    assert explorer.sources("A1", top=1)["partner"].iloc[0] == "C3"
    assert "clusters" in explorer.describe()
    assert explorer.route("A1", "B1")["orders"] > 0
    assert set(explorer.top_routes(2).columns) >= {"pickup_cluster", "drop_cluster"}


def test_explorer_views_are_sticky_but_isolated():
    explorer = RouteExplorer(make_frame(), grain_cols=("week_period", "hour"))
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


def test_filter_tests_accept_numbers_operators_and_ranges(graph):
    routes = graph.matching_routes(edge_filter={"orders": 400})
    assert routes == graph.matching_routes(edge_filter={"orders": (">=", 400)})
    assert set(graph.matching_routes(edge_filter={"orders": (">", 400)})) <= set(routes)

    values = {(u, v): graph.edge_value(u, v, "orders") for u, v in graph.routes()}
    banded = graph.matching_routes(edge_filter={"orders": (300, 600)})
    assert all(300 <= values[route] <= 600 for route in banded)
    assert set(banded) == {r for r, value in values.items() if 300 <= value <= 600}

    # several metrics are combined with AND
    both = graph.matching_routes(edge_filter={"orders": 300, "speed": (">", 0.2)})
    assert all(values[r] >= 300 and graph.edge_value(*r, "speed") > 0.2 for r in both)


def test_keep_returns_a_new_graph_and_leaves_the_original_alone(graph):
    before = graph.graph.number_of_edges()
    cutoff = float(np.median([graph.edge_value(u, v, "speed") for u, v in graph.routes()]))
    fast = graph.keep(edge_filter={"speed": (">", cutoff)})
    assert fast is not graph
    assert graph.graph.number_of_edges() == before          # untouched
    assert fast.graph.number_of_edges() < before
    assert all(fast.edge_value(u, v, "speed") > cutoff for u, v in fast.routes())
    # tensors are shared, so the numbers agree with the source graph
    for u, v in fast.routes():
        assert fast.edge_value(u, v, "orders") == graph.edge_value(u, v, "orders")


def test_cut_is_in_place_and_the_mirror_of_keep():
    frame = make_frame()
    kept = RouteGraph.from_dataframe(frame, grain_cols=("week_period", "hour")).keep(edge_filter={"orders": (">=", 400)})
    cut = RouteGraph.from_dataframe(frame, grain_cols=("week_period", "hour"))
    result = cut.cut(edge_filter={"orders": ("<", 400)})
    assert result is cut                                     # chainable, in place
    assert sorted(cut.routes()) == sorted(kept.routes())


def test_node_filter_drops_the_cluster_and_its_routes(graph):
    threshold = 2000
    busy = graph.keep(node_filter={"orders": threshold})
    for cluster in busy.graph.nodes:
        assert graph.node_value(cluster, "orders", "both") >= threshold
    for u, v in busy.routes():
        assert u in busy.graph and v in busy.graph
    assert set(busy.graph.nodes) == set(graph.matching_clusters({"orders": threshold}))


def test_filters_respect_the_grain(graph):
    # the same test against a narrower window sees smaller totals, so fewer pass
    busiest = max(graph.edge_value(u, v, "orders") for u, v in graph.routes())
    everything = graph.matching_routes(edge_filter={"orders": busiest})
    morning = graph.matching_routes(edge_filter={"orders": busiest}, hour=[8, 9])
    assert len(everything) == 1
    assert morning == []
    assert graph.edge_value(*everything[0], "orders", hour=[8, 9]) < busiest


def test_isolated_clusters_go_unless_asked_to_stay(graph):
    tight = {"orders": graph.rank_routes(top=1)["orders"].iloc[0]}
    pruned = graph.keep(edge_filter=tight)
    assert all(pruned.graph.degree(n) > 0 for n in pruned.graph.nodes)
    kept = graph.keep(edge_filter=tight, drop_isolated=False)
    assert kept.graph.number_of_nodes() == graph.graph.number_of_nodes()


def test_filter_rejects_nonsense(graph):
    with pytest.raises(ValueError):
        graph.keep()                                          # no filter at all
    with pytest.raises(KeyError):
        graph.keep(edge_filter={"not_a_metric": 1})
    with pytest.raises(ValueError):
        graph.keep(edge_filter={"orders": ("~", 1)})          # unknown operator
    with pytest.raises(ValueError):
        graph.keep(edge_filter={"orders": (1, 2, 3)})         # malformed test


def test_nan_never_passes_a_filter():
    frame = make_frame().drop(columns=["avg_distance"])       # speed becomes NaN
    graph = RouteGraph.from_dataframe(frame, grain_cols=("week_period", "hour"))
    assert np.isnan(graph.edge_value("A1", "B1", "speed"))
    assert graph.matching_routes(edge_filter={"speed": (">", 0)}) == []
    assert graph.matching_routes(edge_filter={"speed": ("<", 999)}) == []


def test_filter_applies_before_the_expansion_walks(graph):
    cutoff = float(np.median([graph.edge_value(u, v, "speed") for u, v in graph.routes()]))
    slow = graph.keep(edge_filter={"speed": ("<=", cutoff)}).routes()
    filtered = graph.flow_subgraph("A1", upstream=3, downstream=3,
                                   edge_filter={"speed": (">", cutoff)})
    for u, v in filtered.edges():
        if filtered[u][v]["side"] != "cross":
            assert (u, v) not in slow
    # the top-k is then chosen from the survivors, not from everything
    plain = graph.flow_subgraph("A1", upstream=3, downstream=3)
    assert set(filtered.nodes) != set(plain.nodes)
    assert filtered.graph["edge_filter"] == {"speed": (">", cutoff)}


def test_filtering_away_the_focus_says_so(graph):
    with pytest.raises(ValueError, match="removed the focus"):
        graph.flow_subgraph("A1", node_filter={"orders": 10 ** 9})


def test_explorer_keep_and_cut(graph):
    explorer = RouteExplorer(make_frame(), grain_cols=("week_period", "hour"))
    cutoff = float(np.median([explorer.value(u, v, "speed") for u, v in explorer.routes]))
    fast = explorer.keep(edge_filter={"speed": (">", cutoff)})
    assert fast.graph is not explorer.graph
    assert len(fast.routes) < len(explorer.routes)
    assert fast.metric == explorer.metric                     # the view keeps its defaults

    peak = explorer.where(hour=[8, 9]).keep(edge_filter={"orders": 1000})
    assert peak.grain == {"hour": [8, 9]}
    assert len(peak.routes) <= len(explorer.routes)

    trimmed = RouteExplorer(make_frame(), grain_cols=("week_period", "hour"))
    assert trimmed.cut(edge_filter={"orders": ("<", 400)}) is trimmed
    assert len(trimmed.routes) < len(explorer.routes)


def make_lopsided_frame():
    """B1 barely feeds A1 but is huge overall; B2 feeds A1 hard but is small."""
    legs = {("B1", "A1"): 10, ("B1", "Z9"): 900, ("B2", "A1"): 300,
            ("B2", "Z8"): 5, ("B3", "A1"): 200, ("B3", "Z7"): 100}
    return pd.DataFrame([
        {"pickup_cluster": s, "drop_cluster": d, "orders": float(v),
         "requests": float(v * 2), "avg_distance": 5.0, "avg_duration": 20.0}
        for (s, d), v in legs.items()
    ])


def test_grains_are_optional_and_can_be_anything():
    frame = make_frame()
    flat = RouteGraph.from_dataframe(frame)
    assert flat.schema.is_flat                                # one bucket
    assert flat.schema.grain_names == ["__all__"]
    assert flat.edge_value("A1", "B1", "orders") == pytest.approx(
        frame[(frame.pickup_cluster == "A1") & (frame.drop_cluster == "B1")]["orders"].sum())

    # a cohort column is a perfectly good grain
    frame = frame.assign(variant=["test" if i % 2 else "control" for i in range(len(frame))])
    cohort = RouteGraph.from_dataframe(frame, grain_cols=("variant",))
    assert cohort.schema.grains == {"variant": ["control", "test"]}
    split = sum(cohort.edge_value("A1", "B1", "orders", variant=arm)
                for arm in ("test", "control"))
    assert split == pytest.approx(cohort.edge_value("A1", "B1", "orders"))


def test_columns_named_for_a_bucket_do_not_become_metrics():
    # summing `hour` is meaningless, so it stays out even when it is not a grain
    graph = RouteGraph.from_dataframe(make_frame())
    assert "hour" not in graph.schema.sum_metrics + graph.schema.mean_metrics
    assert graph.schema.default_metric == "orders"
    for name in ("hour", "day_of_week", "time_slot", "week_period", "cohort"):
        assert looks_like_dimension(name), name


def test_integer_counts_are_still_metrics():
    """Few distinct values does not make a column a dimension.

    A count like `cancelled` (0-11) or `riders` (1-30) is an integer with a
    handful of levels, exactly like an hour column - but it is a measurement,
    and dropping it would lose real data silently.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame([{
        "pickup_cluster": "A1", "drop_cluster": "B1",
        "orders": float(rng.integers(50, 400)),
        "cancelled": int(rng.integers(0, 12)),
        "riders": int(rng.integers(1, 30)),
        "avg_distance": 4.0, "avg_duration": 16.0,
    } for _ in range(60)])
    graph = RouteGraph.from_dataframe(frame)
    assert {"cancelled", "riders"} <= set(graph.schema.sum_metrics)
    assert graph.edge_value("A1", "B1", "cancelled") == pytest.approx(
        frame["cancelled"].sum())
    for name in ("cancelled", "riders", "orders", "supply_count"):
        assert not looks_like_dimension(name), name


def test_top_k_can_rank_on_the_edge_or_on_the_cluster():
    graph = RouteGraph.from_dataframe(make_lopsided_frame())
    by_route = [name for name, _ in graph.top_sources("A1", top=2)]
    by_cluster = [name for name, _ in graph.top_sources("A1", top=2, rank_by="node")]
    assert by_route == ["B2", "B3"]            # biggest routes into A1
    assert by_cluster == ["B1", "B2"]          # biggest clusters that feed A1
    assert graph.top_sources("A1", top=1, rank_by="node")[0][1] == pytest.approx(
        graph.node_value("B1", "orders", "both"))
    with pytest.raises(ValueError):
        graph.partners("A1", rank_by="sideways")


def test_the_expansion_honours_the_ranking_basis():
    graph = RouteGraph.from_dataframe(make_lopsided_frame())
    by_route = graph.flow_subgraph("A1", upstream=2, downstream=0)
    by_cluster = graph.flow_subgraph("A1", upstream=2, downstream=0, rank_by="node")
    assert sorted(n for n in by_route if n != "A1") == ["B2", "B3"]
    assert sorted(n for n in by_cluster if n != "A1") == ["B1", "B2"]
    assert by_cluster.graph["rank_by"] == "node"
    # the edge still carries the route value, with the ranking value alongside
    edge = by_cluster["B1"]["A1"]
    assert edge["value"] == pytest.approx(graph.edge_value("B1", "A1", "orders"))
    assert edge["rank_value"] == pytest.approx(graph.node_value("B1", "orders", "both"))


def test_partners_frame_records_the_basis():
    graph = RouteGraph.from_dataframe(make_lopsided_frame())
    frame = graph.partners_frame("A1", direction="in", rank_by="node")
    assert set(frame["ranked_by"]) == {"node"}
    assert list(frame["partner"])[:2] == ["B1", "B2"]


def test_edge_columns_can_be_chosen():
    graph = RouteGraph.from_dataframe(make_frame(), grain_cols=("week_period", "hour"))
    everything = graph.edges_frame()
    assert {"orders", "requests", "speed", "avg_distance"} <= set(everything.columns)

    chosen = graph.edges_frame(metrics=["orders", "speed"])
    assert list(chosen.columns) == ["pickup_cluster", "drop_cluster", "count",
                                    "coverage", "orders", "speed"]
    windowed = graph.edges_frame(metrics=["orders"], hour=[8, 9])
    assert (windowed["orders"] < chosen["orders"]).all()      # a narrower window
    with pytest.raises(KeyError):
        graph.edges_frame(metrics=["nope"])


def make_weighted_frame():
    """B feeds A lightly at 2km, C feeds A heavily at 4km; A sends out at 9km."""
    return pd.DataFrame([
        {"pickup_cluster": "B", "drop_cluster": "A", "orders": 10.0,
         "requests": 100.0, "avg_distance": 2.0, "avg_duration": 10.0},
        {"pickup_cluster": "C", "drop_cluster": "A", "orders": 20.0,
         "requests": 300.0, "avg_distance": 4.0, "avg_duration": 10.0},
        {"pickup_cluster": "A", "drop_cluster": "D", "orders": 30.0,
         "requests": 100.0, "avg_distance": 9.0, "avg_duration": 10.0},
    ])


def test_node_split_holds_in_and_out_per_metric():
    graph = RouteGraph.from_dataframe(make_weighted_frame(), weight_col="requests")
    split = graph.node_split("A", metrics=["orders", "requests", "avg_distance"])
    assert split["orders"] == {"in": 30.0, "out": 30.0}          # 10 + 20, and 30
    assert split["requests"] == {"in": 400.0, "out": 100.0}
    # the inbound average is weighted by requests, not a mean of the averages
    assert split["avg_distance"]["in"] == pytest.approx((2 * 100 + 4 * 300) / 400)
    assert split["avg_distance"]["in"] != pytest.approx((2 + 4) / 2)
    assert split["avg_distance"]["out"] == pytest.approx(9.0)
    assert "diff" not in split["orders"]                          # not asked for


def test_the_weight_column_is_the_users_choice():
    frame = make_weighted_frame()
    by_requests = RouteGraph.from_dataframe(frame, weight_col="requests")
    by_orders = RouteGraph.from_dataframe(frame, weight_col="orders")
    assert by_requests.weight_col == "requests"
    assert by_orders.weight_col == "orders"
    assert by_requests.node_split("A", metrics=["avg_distance"])["avg_distance"]["in"] == (
        pytest.approx((2 * 100 + 4 * 300) / 400))
    assert by_orders.node_split("A", metrics=["avg_distance"])["avg_distance"]["in"] == (
        pytest.approx((2 * 10 + 4 * 20) / 30))
    # sums do not care which column weights the averages
    assert by_requests.node_split("A", metrics=["orders"])["orders"] == (
        by_orders.node_split("A", metrics=["orders"])["orders"])


def test_differences_are_opt_in_per_metric():
    graph = RouteGraph.from_dataframe(make_weighted_frame(), weight_col="requests")
    chosen = graph.node_split("A", metrics=["orders", "requests", "avg_distance"],
                              diff=["requests"])
    assert "diff" in chosen["requests"] and "diff" not in chosen["orders"]
    assert chosen["requests"]["diff"] == pytest.approx(100.0 - 400.0)

    everything = graph.node_split("A", metrics=["orders", "requests"], diff=True)
    assert all("diff" in entry for entry in everything.values())

    flipped = graph.node_split("A", metrics=["requests"], diff=True,
                               diff_order="in-out")["requests"]
    assert flipped["diff"] == pytest.approx(400.0 - 100.0)
    with pytest.raises(ValueError):
        graph.node_split("A", metrics=["orders"], diff=True, diff_order="sideways")


def test_node_metrics_frame_spreads_it_across_the_graph():
    graph = RouteGraph.from_dataframe(make_weighted_frame(), weight_col="requests")
    frame = graph.node_metrics_frame(metrics=["orders", "avg_distance"],
                                     diff=["orders"])
    assert set(frame.columns) == {"cluster", "in_routes", "out_routes",
                                  "in_orders", "out_orders", "diff_orders",
                                  "in_avg_distance", "out_avg_distance"}
    row = frame[frame["cluster"] == "A"].iloc[0]
    assert (row["in_orders"], row["out_orders"], row["diff_orders"]) == (30.0, 30.0, 0.0)
    # a side with no routes is NaN, which is not the same as zero
    source = frame[frame["cluster"] == "B"].iloc[0]
    assert np.isnan(source["in_orders"]) and source["out_orders"] == 10.0
    assert source["in_routes"] == 0

    just_one = graph.node_metrics_frame(metrics=["orders"], nodes=["A"])
    assert list(just_one["cluster"]) == ["A"]
    with pytest.raises(KeyError):
        graph.node_metrics_frame(nodes=["nope"])
    with pytest.raises(KeyError):
        graph.node_metrics_frame(metrics=["not_a_metric"])


def test_node_split_respects_the_grain():
    frame = make_frame()
    graph = RouteGraph.from_dataframe(frame, grain_cols=("week_period", "hour"))
    everything = graph.node_split("A1", metrics=["orders"])["orders"]
    morning = graph.node_split("A1", metrics=["orders"], hour=[8, 9])["orders"]
    assert morning["in"] < everything["in"]
    assert morning["out"] < everything["out"]


def test_explorer_exposes_the_same():
    explorer = RouteExplorer(make_weighted_frame(), weight_col="requests")
    assert explorer.weight_col == "requests"
    assert explorer.node_split("A", metrics=["orders"])["orders"]["in"] == 30.0
    frame = explorer.node_metrics_frame(metrics=["orders"], diff=True)
    assert "diff_orders" in frame.columns


def test_nothing_new_is_required():
    """Every argument added for the node view is an optional keyword."""
    graph = RouteGraph.from_dataframe(make_weighted_frame())

    bare = graph.node_split("A")                       # cluster only
    assert set(bare) == set(graph.schema.metric_names)
    assert all(set(entry) == {"in", "out"} for entry in bare.values())

    frame = graph.node_metrics_frame()                 # no arguments at all
    assert len(frame) == graph.graph.number_of_nodes()
    assert {"in_orders", "out_orders"} <= set(frame.columns)
    assert not any(c.startswith("diff_") for c in frame.columns)
    assert frame["out_orders"].iloc[0] == frame["out_orders"].max()   # sorted sensibly

    # and they cannot be passed positionally by mistake
    with pytest.raises(TypeError):
        graph.node_split("A", ["orders"])
    with pytest.raises(TypeError):
        graph.node_metrics_frame(["orders"])


def make_mirror_frame():
    """Exp is the observation node; B2 and A2 sit on both of its sides."""
    legs = [("B2", "Exp"), ("A1", "B2"), ("A2", "B2"), ("A3", "B2"),
            ("Exp", "A1"), ("Exp", "B1"), ("Exp", "B2"), ("B1", "A2"), ("B1", "C1")]
    return pd.DataFrame([
        {"pickup_cluster": u, "drop_cluster": v, "orders": 10.0 * (i + 1),
         "requests": 20.0, "avg_distance": 3.0, "avg_duration": 12.0}
        for i, (u, v) in enumerate(legs)
    ])


def test_a_cluster_on_both_sides_is_split():
    graph = RouteGraph.from_dataframe(make_mirror_frame())
    sub = graph.flow_subgraph("Exp", upstream=[3, 3], downstream=[3, 3])

    clusters = {}
    for node, data in sub.nodes(data=True):
        clusters.setdefault(data["cluster"], []).append(data["side"])
    # B2 feeds Exp and Exp feeds B2, so it appears once on each side
    assert sorted(clusters["B2"]) == ["drop", "source"]
    assert sorted(clusters["A2"]) == ["drop", "source"]
    # a cluster on one side only keeps its plain id as the node key
    assert "A3" in sub and sub.nodes["A3"]["cluster"] == "A3"
    assert ("source", "B2") in sub and ("drop", "B2") in sub
    # and every node carries the real cluster id for lookups
    assert all("cluster" in d for _, d in sub.nodes(data=True))


def test_splitting_removes_the_backwards_arrows():
    graph = RouteGraph.from_dataframe(make_mirror_frame())
    options = dict(upstream=[3, 3], downstream=[3, 3])

    merged = graph.flow_subgraph("Exp", mirror=False, **options)
    backwards = [(u, v) for u, v in merged.edges()
                 if merged.nodes[u]["level"] > merged.nodes[v]["level"]]
    assert backwards, "the old behaviour drew routes pointing back across the figure"

    split = graph.flow_subgraph("Exp", mirror=True, **options)
    assert not [(u, v) for u, v in split.edges()
                if split.nodes[u]["level"] > split.nodes[v]["level"]]


def test_a_cluster_reached_twice_on_one_side_is_drawn_once():
    graph = RouteGraph.from_dataframe(make_mirror_frame())
    sub = graph.flow_subgraph("Exp", upstream=0, downstream=[3, 3])
    drops = [d["cluster"] for _, d in sub.nodes(data=True) if d["side"] == "drop"]
    assert len(drops) == len(set(drops))          # no repeats within a side


def test_the_frames_report_cluster_ids_not_node_keys():
    graph = RouteGraph.from_dataframe(make_mirror_frame())
    frame = graph.flow_frame("Exp", upstream=[3, 3], downstream=[3, 3])
    for column in ("pickup_cluster", "drop_cluster"):
        assert all(isinstance(v, str) for v in frame[column])
    assert "B2" in set(frame["pickup_cluster"]) | set(frame["drop_cluster"])
    tree = graph.flow_tree("Exp", upstream=2, downstream=2)
    assert "('source'" not in tree and "('drop'" not in tree


def test_derived_metrics_are_computed_after_aggregation():
    frame = pd.DataFrame([
        {"pickup_cluster": "A", "drop_cluster": "B", "day": "mon",
         "orders": 10.0, "requests": 100.0, "avg_distance": 2.0, "avg_duration": 10.0},
        {"pickup_cluster": "A", "drop_cluster": "B", "day": "tue",
         "orders": 90.0, "requests": 300.0, "avg_distance": 4.0, "avg_duration": 10.0},
    ])
    graph = RouteGraph.from_dataframe(
        frame, grain_cols=("day",), weight_col="requests",
        derived_metrics={"fill_rate": ("orders", "/", "requests"),
                         "lost": ("requests", "-", "orders")})

    assert "fill_rate" in graph.schema.metric_names
    # per bucket it is that bucket's ratio
    assert graph.edge_value("A", "B", "fill_rate", day="mon") == pytest.approx(0.1)
    assert graph.edge_value("A", "B", "fill_rate", day="tue") == pytest.approx(0.3)
    # collapsed it is the ratio of the totals, not the mean of the ratios
    assert graph.edge_value("A", "B", "fill_rate") == pytest.approx(100 / 400)
    assert graph.edge_value("A", "B", "fill_rate") != pytest.approx((0.1 + 0.3) / 2)
    assert graph.edge_value("A", "B", "lost") == pytest.approx(300.0)


def test_a_derived_metric_behaves_like_any_other():
    frame = make_weighted_frame()
    graph = RouteGraph.from_dataframe(
        frame, derived_metrics={"per_request": ("orders", "/", "requests")})
    assert graph.node_split("A", metrics=["per_request"])["per_request"]["in"] == (
        pytest.approx(graph.node_value("A", "orders", "in")
                      / graph.node_value("A", "requests", "in")))
    assert not graph.rank_routes(metric="per_request").empty
    assert graph.matching_routes(edge_filter={"per_request": (">", 0.05)})
    assert "per_request" in graph.edges_frame(metrics=["per_request"]).columns


def test_a_derived_metric_can_be_a_function():
    frame = make_weighted_frame()
    graph = RouteGraph.from_dataframe(frame, derived_metrics={
        "gap": lambda v: v["requests"] - v["orders"] * 2})
    edge = graph.route_summary("B", "A")
    assert edge["gap"] == pytest.approx(edge["requests"] - edge["orders"] * 2)


def test_bad_derived_specs_are_rejected():
    frame = make_weighted_frame()
    with pytest.raises(ValueError, match="left, op, right"):
        RouteGraph.from_dataframe(frame, derived_metrics={"x": ("orders", "/")})
    with pytest.raises(ValueError, match="Unknown operator"):
        RouteGraph.from_dataframe(frame, derived_metrics={"x": ("orders", "^", "requests")})
    with pytest.raises(ValueError, match="already used by a column"):
        RouteGraph.from_dataframe(frame, derived_metrics={"orders": ("orders", "/", "requests")})
    with pytest.raises(KeyError, match="not one of the schema"):
        RouteGraph.from_dataframe(frame, derived_metrics={"x": ("nope", "/", "requests")})


def test_dividing_by_zero_gives_nan_not_an_error():
    frame = pd.DataFrame([{"pickup_cluster": "A", "drop_cluster": "B",
                           "orders": 5.0, "requests": 0.0,
                           "avg_distance": 1.0, "avg_duration": 1.0}])
    graph = RouteGraph.from_dataframe(
        frame, weight_col=None, derived_metrics={"rate": ("orders", "/", "requests")})
    assert np.isnan(graph.edge_value("A", "B", "rate"))
