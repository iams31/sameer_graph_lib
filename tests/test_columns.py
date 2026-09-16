import warnings

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from sameer_graph_lib import RouteExplorer
from sameer_graph_lib.columns import (ask_columns, describe_columns,
                                      suggest_columns, suggest_speed_pair,
                                      suggest_weight)


def make_frame():
    """Nothing here is named the way the library's defaults are."""
    rng = np.random.default_rng(2)
    rows = []
    for source, target in [("Z1", "Y1"), ("Z1", "Y2"), ("X1", "Z1"), ("X2", "Z1")]:
        for day_part in ("morning", "evening"):
            for slot in (1, 2, 3):
                rows.append({
                    "origin_zone": source,
                    "dest_zone": target,
                    "day_part": day_part,
                    "time_slot": slot,
                    "trips": float(rng.integers(50, 400)),
                    "bookings": float(rng.integers(80, 600)),
                    "avg_km": float(rng.uniform(1, 9)),
                    "avg_mins": float(rng.uniform(5, 40)),
                    "operator": "acme",
                })
    return pd.DataFrame(rows)


def test_columns_are_proposed_from_the_frame():
    proposal = suggest_columns(make_frame())
    assert proposal["pickup_col"] == "origin_zone"
    assert proposal["drop_col"] == "dest_zone"
    assert proposal["grain_cols"] == ["day_part", "time_slot"]
    assert set(proposal["metrics"]) == {"trips", "bookings", "avg_km", "avg_mins"}
    # a single-valued text column is neither a grain nor a metric
    assert "operator" not in proposal["grain_cols"] + proposal["metrics"]


def test_the_weight_column_is_proposed_too():
    proposal = suggest_columns(make_frame())
    # averages must be weighted by volume, so the volume column is found
    assert proposal["weight_col"] == "trips"
    assert suggest_weight(["gmv", "sessions"]) == "sessions"
    assert suggest_weight(["gmv"], weight_col="gmv") == "gmv"


def test_the_sum_and_average_split_is_proposed_and_overridable():
    proposal = suggest_columns(make_frame())
    # guessed from the names: avg_km and avg_mins say what they are
    assert proposal["mean_metrics"] == ["avg_km", "avg_mins"]
    assert proposal["sum_metrics"] == ["trips", "bookings"]
    assert sorted(proposal["mean_metrics"] + proposal["sum_metrics"]) == sorted(
        proposal["metrics"])

    # and the guess can be overridden either way round
    by_mean = suggest_columns(make_frame(), mean_metrics=["trips"])
    assert by_mean["mean_metrics"] == ["trips"]
    assert "trips" not in by_mean["sum_metrics"]
    by_sum = suggest_columns(make_frame(), sum_metrics=["trips", "bookings", "avg_km"])
    assert by_sum["mean_metrics"] == ["avg_mins"]


def test_a_name_that_gives_nothing_away_can_be_corrected():
    """eta and fill are averages, but nothing in the name says so."""
    frame = pd.DataFrame([{"pickup_cluster": "A", "drop_cluster": "B",
                           "trips": 100.0, "gmv": 500.0, "eta": 12.0,
                           "fill": 0.8} for _ in range(3)])
    guessed = RouteExplorer(frame, weight_col="trips")
    assert "eta" in guessed.schema.sum_metrics          # the wrong guess
    assert guessed.value("A", "B", "eta") == pytest.approx(36.0)

    told = RouteExplorer(frame, weight_col="trips",
                         sum_metrics=["trips", "gmv"],
                         mean_metrics=["eta", "fill"])
    assert told.schema.mean_metrics[:2] == ["eta", "fill"]
    assert told.value("A", "B", "eta") == pytest.approx(12.0)    # averaged
    assert told.value("A", "B", "gmv") == pytest.approx(1500.0)  # still summed


def test_the_speed_pair_is_proposed_too():
    proposal = suggest_columns(make_frame())
    assert proposal["length_metric"] == "avg_km"
    assert proposal["ride_time_metric"] == "avg_mins"
    assert suggest_speed_pair(["distance_travelled", "trip_duration"]) == (
        "distance_travelled", "trip_duration")
    assert suggest_speed_pair(["trips"]) == (None, None)


def test_anything_passed_in_is_respected():
    proposal = suggest_columns(make_frame(), grain_cols=["day_part"],
                               metrics=["trips"], pickup_col="dest_zone")
    assert proposal["pickup_col"] == "dest_zone"
    assert proposal["grain_cols"] == ["day_part"]
    assert proposal["metrics"] == ["trips"]


def test_describe_columns_lists_every_column_and_its_role():
    text = describe_columns(make_frame())
    for column in make_frame().columns:
        assert column in text
    assert "pickup id" in text and "drop id" in text
    assert "grain" in text
    assert "metric (avg, distance)" in text and "metric (sum)" in text


def test_ask_falls_back_to_the_proposal_without_a_terminal():
    # no input_fn and no tty: nothing should block, the proposal stands
    assert ask_columns(make_frame()) == suggest_columns(make_frame())


def test_ask_accepts_names_numbers_and_blanks():
    frame = make_frame()
    replies = iter([
        "",                 # keep the proposed pickup
        "dest_zone",        # by name
        "3",               # by number: day_part
        "5 6",             # trips, bookings
        "avg_km",          # of those, only this one is an average
        "avg_km",
        "",                 # keep the proposed duration
        "bookings",         # weight the averages by this
    ])
    chosen = ask_columns(frame, input_fn=lambda prompt: next(replies),
                         output_fn=lambda *a: None)
    assert chosen["pickup_col"] == "origin_zone"
    assert chosen["drop_col"] == "dest_zone"
    assert chosen["grain_cols"] == ["day_part"]
    assert chosen["metrics"] == ["trips", "bookings"]
    assert chosen["mean_metrics"] == ["avg_km"]
    assert chosen["sum_metrics"] == ["trips", "bookings"]
    assert chosen["length_metric"] == "avg_km"
    assert chosen["ride_time_metric"] == "avg_mins"
    assert chosen["weight_col"] == "bookings"


def test_ask_rejects_an_answer_that_is_not_a_column():
    with pytest.raises(ValueError, match="not a column"):
        ask_columns(make_frame(), input_fn=lambda prompt: "nonsense",
                    output_fn=lambda *a: None)
    with pytest.raises(ValueError, match="not one of the listed"):
        ask_columns(make_frame(), input_fn=lambda prompt: "99",
                    output_fn=lambda *a: None)


def test_none_clears_a_selection():
    replies = iter(["", "", "none", "", "", "", "", ""])
    chosen = ask_columns(make_frame(), input_fn=lambda prompt: next(replies),
                         output_fn=lambda *a: None)
    assert chosen["grain_cols"] == []


def test_explorer_ask_builds_a_working_graph():
    frame = make_frame()
    explorer = RouteExplorer.ask(frame)
    assert explorer.schema.grain_names == ["day_part", "time_slot"]
    assert explorer.schema.sum_metrics == ["trips", "bookings"]
    assert explorer.schema.length_metric == "avg_km"
    assert explorer.schema.ride_time_metric == "avg_mins"
    assert explorer.weight_col == "trips"
    # speed is only computable because the pair was identified
    assert not np.isnan(explorer.value("Z1", "Y1", "speed"))
    assert explorer.sources("Z1", top=1)["partner"].iloc[0] in ("X1", "X2")


def test_explorer_preview_does_not_prompt():
    text = RouteExplorer.preview(make_frame())
    assert "origin_zone" in text and "pickup id" in text


def test_explorer_ask_needs_two_id_columns():
    frame = make_frame().drop(columns=["dest_zone"])
    with pytest.raises(ValueError, match="HexMetricGraph"):
        RouteExplorer.ask(frame)


def test_metrics_can_be_named_explicitly():
    explorer = RouteExplorer(make_frame(), pickup_col="origin_zone",
                             drop_col="dest_zone",
                             grain_cols=("day_part", "time_slot"),
                             metrics=["trips", "avg_km", "avg_mins"],
                             length_metric="avg_km", ride_time_metric="avg_mins")
    assert explorer.schema.sum_metrics == ["trips"]
    assert "bookings" not in explorer.metrics
    assert explorer.value("Z1", "Y1", "trips") > 0
    with pytest.raises(KeyError):
        explorer.value("Z1", "Y1", "bookings")


def test_bad_metric_selection_is_reported():
    frame = make_frame()
    common = dict(pickup_col="origin_zone", drop_col="dest_zone",
                  grain_cols=("day_part", "time_slot"))
    with pytest.raises(KeyError):
        RouteExplorer(frame, metrics=["not_a_column"], **common)
    with pytest.raises(TypeError):
        RouteExplorer(frame, metrics=["operator"], **common)
    with pytest.raises(ValueError):
        RouteExplorer(frame, metrics=[], **common)


def test_grains_are_optional():
    # nothing passed: one flat bucket, and no grain filtering to do
    flat = RouteExplorer(make_frame(), pickup_col="origin_zone", drop_col="dest_zone")
    assert flat.schema.is_flat
    assert flat.value("Z1", "Y1", "trips") > 0


def test_a_named_grain_that_is_absent_says_what_to_pass():
    with pytest.raises(KeyError, match="candidates here"):
        RouteExplorer(make_frame(), pickup_col="origin_zone", drop_col="dest_zone",
                      grain_cols=("week_period",))


def weighted_frame():
    return pd.DataFrame([
        {"pickup_cluster": "A", "drop_cluster": "B", "orders": 10.0,
         "requests": 100.0, "avg_distance": 2.0, "avg_duration": 10.0},
        {"pickup_cluster": "A", "drop_cluster": "B", "orders": 90.0,
         "requests": 300.0, "avg_distance": 8.0, "avg_duration": 10.0},
    ])


def test_a_weight_column_that_is_not_there_is_refused():
    """A typo would silently leave every mean unweighted."""
    with pytest.raises(KeyError, match="would be unweighted"):
        RouteExplorer(weighted_frame(), weight_col="requsts")
    # the real one still weights
    explorer = RouteExplorer(weighted_frame(), weight_col="requests")
    assert explorer.value("A", "B", "avg_distance") == pytest.approx(
        (2 * 100 + 8 * 300) / 400)


def test_the_schema_does_not_claim_a_weight_column_it_has_not_got():
    frame = weighted_frame().drop(columns=["requests"])
    explorer = RouteExplorer(frame)
    assert explorer.weight_col is None                  # not the absent default
    assert explorer.value("A", "B", "avg_distance") == pytest.approx(5.0)   # plain mean


def test_weight_col_none_is_still_allowed():
    explorer = RouteExplorer(weighted_frame(), weight_col=None)
    assert explorer.weight_col is None
    assert explorer.value("A", "B", "avg_distance") == pytest.approx(5.0)


def test_one_column_cannot_be_both_ends_of_a_route():
    with pytest.raises(ValueError, match="self loop"):
        RouteExplorer(weighted_frame(), pickup_col="pickup_cluster",
                      drop_col="pickup_cluster")


def test_a_missing_route_id_column_is_named():
    with pytest.raises(KeyError, match="origin"):
        RouteExplorer(weighted_frame(), pickup_col="origin")


def test_a_repeated_metric_name_is_not_an_error():
    explorer = RouteExplorer(weighted_frame(),
                             metrics=["orders", "orders", "avg_distance"])
    assert explorer.schema.sum_metrics == ["orders"]
    assert "avg_distance" in explorer.schema.mean_metrics


def test_average_hints_match_words_not_substrings():
    """Real column names, including ones the old substring test got wrong."""
    from sameer_graph_lib.route_graph import classify_metrics

    frame = pd.DataFrame([{
        "Total_Orders": 1.0, "Total_Pings": 1.0, "FE_Count": 1.0,
        "Bikelite_FE_per": 0.4, "Avg_LM": 1.0, "AOR": 1.0,
        "orders_generated": 5.0, "km_per_trip": 2.0, "Surge%": 0.1,
    }])
    sums, means = classify_metrics(frame)
    # a trailing "per" is a percentage; a substring test looking for "per_" missed it
    assert "Bikelite_FE_per" in means
    assert "km_per_trip" in means and "Avg_LM" in means and "Surge%" in means
    # and "rate" inside "generated" is not a rate
    assert "orders_generated" in sums
    assert "Total_Orders" in sums and "FE_Count" in sums


def test_a_named_metric_the_frame_lacks_is_flagged():
    frame = pd.DataFrame([{"pickup_cluster": "A", "drop_cluster": "B",
                           "Total_Orders": 10.0, "Total_Pings": 100.0,
                           "Avg_LM": 2.0, "Avg_Ride_Time": 10.0}])
    common = dict(length_metric="Avg_LM", ride_time_metric="Avg_Ride_Time")

    # named through mean_metrics: it stays in the schema as NaN, so it warns
    with pytest.warns(UserWarning, match="will be NaN"):
        explorer = RouteExplorer(frame, mean_metrics=["Avg_LM", "Bikelite_FE_per"],
                                 **common)
    assert np.isnan(explorer.value("A", "B", "Bikelite_FE_per"))

    # named through metrics=: that one refuses, and says what is available
    with pytest.raises(KeyError, match="Numeric columns here"):
        RouteExplorer(frame, metrics=["Total_Orders", "Bikelite_FE_per"], **common)

    # the length and ride-time pair may be absent on purpose, and stay quiet
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RouteExplorer(frame.drop(columns=["Avg_LM"]), length_metric="Avg_LM",
                      ride_time_metric="Avg_Ride_Time")


def test_two_columns_is_enough_to_build_and_plot():
    """The smallest useful frame: two id columns and no metrics at all."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two = pd.DataFrame({"pickup_cluster": ["A", "A", "C", "C", "C", "B"],
                        "drop_cluster": ["B", "B", "A", "A", "D", "A"]})
    explorer = RouteExplorer(two)

    assert explorer.schema.default_metric == "row_count"      # always available
    assert explorer.metrics == ["row_count", "count"]         # nothing invented
    assert explorer.schema.sum_metrics == [] == explorer.schema.mean_metrics
    assert explorer.weight_col is None
    assert not explorer.schema.has_speed

    # and it answers the ordinary questions by counting rows
    assert explorer.value("A", "B") == pytest.approx(2.0)
    assert explorer.graph.top_drops("A", 1) == [("B", 2.0)]
    assert [n for n, _ in explorer.graph.top_sources("A", 2)] == ["C", "B"]

    fig = explorer.plot("A", upstream=2, downstream=2)
    assert "row_count" in fig.axes[0].get_title()
    plt.close(fig)


def test_no_column_name_is_assumed():
    frame = pd.DataFrame({"from_zone": ["A", "A", "C"], "to_zone": ["B", "B", "A"],
                          "trips": [10.0, 20.0, 30.0], "km": [2.0, 4.0, 6.0],
                          "mins": [10.0, 10.0, 10.0]})
    explorer = RouteExplorer(frame, pickup_col="from_zone", drop_col="to_zone",
                             metrics=["trips", "km", "mins"],
                             sum_metrics=["trips"], mean_metrics=["km", "mins"],
                             weight_col="trips", length_metric="km",
                             ride_time_metric="mins")
    assert explorer.schema.sum_metrics == ["trips"]
    assert explorer.value("A", "B", "speed") == pytest.approx(
        explorer.value("A", "B", "km") / explorer.value("A", "B", "mins"))
    assert explorer.reach_to("A", budget=15)["cluster"].tolist() == ["C"]


def test_what_is_not_there_is_reported_not_invented():
    frame = pd.DataFrame({"pickup_cluster": ["A"], "drop_cluster": ["B"],
                          "trips": [5.0]})
    explorer = RouteExplorer(frame)
    # no phantom avg_distance / avg_duration / speed in the schema
    assert explorer.metrics == ["row_count", "trips", "count"]
    assert "avg_distance" not in explorer.schema.mean_metrics

    with pytest.raises(KeyError):
        explorer.value("A", "B", "speed")
    with pytest.raises(ValueError, match="No cost metric"):
        explorer.reach_to("A", budget=10)
