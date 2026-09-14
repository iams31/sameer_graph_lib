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
    assert "grain" in text and "metric (distance)" in text


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
    replies = iter(["", "", "none", "", "", "", ""])
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
