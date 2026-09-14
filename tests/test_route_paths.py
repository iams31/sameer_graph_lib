import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from sameer_graph_lib import RouteExplorer, RouteGraph

# A network with hand-chosen leg times (minutes), so every expected answer
# below can be worked out on paper:
#
#   E1 --35--> D1 --40--> C1 --50--> A1 --25--> B1 --20--> B2
#              D2 --90-->  ^
#                    C2 --30--> A1
#                    X1 --200--> C2
LEGS = {
    ("E1", "D1"): 35, ("D1", "C1"): 40, ("D2", "C1"): 90, ("C1", "A1"): 50,
    ("C2", "A1"): 30, ("X1", "C2"): 200, ("A1", "B1"): 25, ("B1", "B2"): 20,
}


def make_frame(volume=100.0):
    rows = []
    for index, ((pickup, drop), minutes) in enumerate(LEGS.items(), start=1):
        for week_period in ("weekday", "weekend"):
            for hour in (8, 18):
                rows.append({
                    "pickup_cluster": pickup,
                    "drop_cluster": drop,
                    "week_period": week_period,
                    "hour": hour,
                    "orders": volume * index,
                    "requests": volume * index * 2,
                    "avg_distance": minutes / 3.0,
                    "avg_duration": float(minutes),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def graph():
    return RouteGraph.from_dataframe(make_frame(), grain_cols=("week_period", "hour"))


def test_leg_times_survive_the_tensor(graph):
    for (pickup, drop), minutes in LEGS.items():
        assert graph.edge_value(pickup, drop, "avg_duration") == pytest.approx(minutes)


def test_reach_to_respects_a_time_budget(graph):
    within_two_hours = graph.reach_to("A1", budget=120, max_hops=3)
    found = dict(zip(within_two_hours["cluster"], within_two_hours["avg_duration"]))
    assert found == pytest.approx({"C2": 30.0, "C1": 50.0, "D1": 90.0})
    # D2 needs 90 + 50 and E1 needs 35 + 40 + 50, both over budget
    assert "D2" not in found and "E1" not in found

    within_one_hour = graph.reach_to("A1", budget=60)
    assert sorted(within_one_hour["cluster"]) == ["C1", "C2"]


def test_reach_records_the_route_and_orders_it(graph):
    frame = graph.reach_to("A1", budget=120)
    assert list(frame["avg_duration"]) == sorted(frame["avg_duration"])   # cheapest first
    row = frame[frame["cluster"] == "D1"].iloc[0]
    assert row["path"] == "D1 -> C1 -> A1"                                # travel order
    assert row["hops"] == 2


def test_reach_from_walks_the_other_way(graph):
    frame = graph.reach_from("A1", budget=60)
    assert dict(zip(frame["cluster"], frame["avg_duration"])) == pytest.approx(
        {"B1": 25.0, "B2": 45.0})
    assert frame[frame["cluster"] == "B2"].iloc[0]["path"] == "A1 -> B1 -> B2"
    # nothing flows back the other way
    assert "C1" not in list(frame["cluster"])


def test_hop_cap_is_independent_of_the_cost_cap(graph):
    generous = graph.reach_to("A1", budget=200, max_hops=3)
    assert "E1" in list(generous["cluster"])          # 125 minutes, but 3 hops
    tighter = graph.reach_to("A1", budget=200, max_hops=2)
    assert "E1" not in list(tighter["cluster"])       # still cheap enough, too far
    assert "D2" in list(tighter["cluster"])           # 140 minutes in 2 hops


def test_min_hops_excludes_direct_neighbours(graph):
    indirect = graph.reach_to("A1", budget=200, min_hops=2)
    assert sorted(indirect["cluster"]) == ["D1", "D2", "E1"]
    assert all(indirect["hops"] >= 2)


def test_no_budget_means_only_the_hop_cap_applies(graph):
    everything = graph.reach_to("A1", max_hops=4)
    assert set(everything["cluster"]) == {"C1", "C2", "D1", "D2", "E1", "X1"}


def test_cost_can_be_any_metric(graph):
    # distances are a third of the minutes, so the same trip costs a third
    by_distance = graph.reach_to("A1", budget=40, cost="avg_distance")
    assert set(by_distance["cluster"]) == {"C1", "C2", "D1"}
    assert by_distance[by_distance["cluster"] == "D1"].iloc[0]["avg_distance"] == pytest.approx(30.0)
    # D2 costs (90 + 50) / 3 = 46.7, just outside
    assert graph.path_total(["D2", "C1", "A1"], "avg_distance") == pytest.approx(140 / 3)
    assert "D2" in set(graph.reach_to("A1", budget=47, cost="avg_distance")["cluster"])


def test_metrics_are_totalled_along_the_path(graph):
    frame = graph.reach_to("A1", budget=120, metrics=["orders"])
    row = frame[frame["cluster"] == "D1"].iloc[0]
    expected = (graph.edge_value("D1", "C1", "orders")
                + graph.edge_value("C1", "A1", "orders"))
    assert row["orders"] == pytest.approx(expected)
    assert graph.path_total(["D1", "C1", "A1"], "avg_duration") == pytest.approx(90.0)


def test_paths_enumerates_alternatives(graph):
    frame = graph.paths("D1", "A1", budget=120)
    assert list(frame["path"]) == ["D1 -> C1 -> A1"]
    assert frame.iloc[0]["avg_duration"] == pytest.approx(90.0)
    assert frame.iloc[0]["clusters"] == ["D1", "C1", "A1"]
    assert graph.paths("D1", "A1", budget=60).empty          # too far for the budget
    assert graph.paths("D1", "A1", max_hops=1).empty         # not adjacent


def test_paths_and_reach_agree_on_the_best_route(graph):
    best = graph.paths("E1", "A1", max_hops=3).iloc[0]
    row = graph.reach_to("A1", max_hops=3, budget=200)
    row = row[row["cluster"] == "E1"].iloc[0]
    assert best["avg_duration"] == pytest.approx(row["avg_duration"])
    assert best["path"] == row["path"]


def test_filters_apply_to_every_hop(graph):
    # C2 -> A1 is the quickest way in; filtering that leg out must not simply
    # be ignored, and must not take the other legs down with it
    busiest = graph.edge_value("C2", "A1", "orders")
    assert busiest > graph.edge_value("C1", "A1", "orders")

    filtered = graph.reach_to("A1", budget=120, edge_filter={"orders": ("<", busiest)})
    assert "C2" not in set(filtered["cluster"])
    assert {"C1", "D1"} <= set(filtered["cluster"])
    assert set(graph.reach_to("A1", budget=120)["cluster"]) == {"C1", "C2", "D1"}


def test_unknown_clusters_and_silly_limits_are_rejected(graph):
    with pytest.raises(KeyError):
        graph.reach_to("NOPE")
    with pytest.raises(KeyError):
        graph.paths("A1", "NOPE")
    with pytest.raises(ValueError):
        graph.reach_to("A1", max_hops=0)


def test_missing_costs_make_a_hop_impassable():
    frame = make_frame().drop(columns=["avg_duration"])
    graph = RouteGraph.from_dataframe(frame, grain_cols=("week_period", "hour"), ride_time_metric="avg_duration")
    assert graph.reach_to("A1", budget=999).empty            # every leg is NaN


def test_a_cycle_does_not_loop_forever():
    frame = pd.concat([make_frame(), pd.DataFrame([{
        "pickup_cluster": "B2", "drop_cluster": "E1", "week_period": w, "hour": h,
        "orders": 50.0, "requests": 100.0, "avg_distance": 5.0, "avg_duration": 15.0,
    } for w in ("weekday", "weekend") for h in (8, 18)])], ignore_index=True)
    graph = RouteGraph.from_dataframe(frame, grain_cols=("week_period", "hour"))
    frame = graph.reach_from("A1", max_hops=6)
    assert len(frame) == len(set(frame["cluster"]))          # one row per cluster
    for path in frame["path"]:
        legs = path.split(" -> ")
        assert len(legs) == len(set(legs))                   # simple paths only


def test_reach_subgraph_is_shaped_for_plotting(graph):
    tree = graph.reach_subgraph("A1", direction="in", budget=120)
    assert tree.nodes["A1"]["level"] == 0
    assert tree.nodes["C1"]["level"] == -1                   # sources sit on the left
    assert tree.nodes["D1"]["level"] == -2
    assert tree.nodes["D1"]["cost"] == pytest.approx(90.0)
    assert tree.graph["direction"] == "in"
    assert tree.number_of_edges() == 3                        # the cheapest-path tree


def test_explorer_reachability_and_grain():
    explorer = RouteExplorer(make_frame(), grain_cols=("week_period", "hour"))
    assert sorted(explorer.reach_to("A1", budget=60)["cluster"]) == ["C1", "C2"]
    assert sorted(explorer.reach_from("A1", budget=60)["cluster"]) == ["B1", "B2"]
    assert explorer.path_total(["D1", "C1", "A1"], "avg_duration") == pytest.approx(90.0)
    assert not explorer.paths("D1", "A1", budget=120).empty
    # the sticky grain carries into the search
    morning = explorer.where(hour=[8])
    assert sorted(morning.reach_to("A1", budget=60)["cluster"]) == ["C1", "C2"]


def test_plot_reach_draws_the_tree(graph):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = graph.plot_reach("A1", budget=120)
    title = fig.axes[0].get_title()
    assert "reaching A1" in title and "within 120" in title
    # one column per hop band, plus the start
    headers = {text.get_text() for text in fig.axes[0].texts}
    assert {"START", "1 HOP", "2 HOPS"} <= headers
    plt.close(fig)

    fig = graph.plot_reach("A1", direction="out", budget=60)
    assert "reachable from A1" in fig.axes[0].get_title()
    plt.close(fig)

    with pytest.warns(UserWarning, match="Nothing is within reach"):
        fig = graph.plot_reach("A1", budget=1)
    plt.close(fig)


def test_schema_options_pass_through_from_dataframe():
    frame = make_frame()
    graph = RouteGraph.from_dataframe(frame, grain_cols=("week_period", "hour"), weight_col="orders",
                                      length_metric="avg_distance",
                                      ride_time_metric="avg_duration",
                                      directed=True)
    assert graph.schema.weight_col == "orders"
    assert graph.graph.is_directed()
    with pytest.raises(TypeError, match="not both"):
        RouteGraph.from_dataframe(frame, schema=graph.schema, weight_col="requests")
