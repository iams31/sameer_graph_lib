import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sameer_graph_lib import plotter


@pytest.fixture
def frame():
    rng = np.random.default_rng(3)
    rows = []
    for variant in ("test", "control"):
        shift = 0.0 if variant == "control" else 1.5
        for hour in range(24):
            for _ in range(6):
                rows.append({
                    "variant": variant,
                    "city": rng.choice(["blr", "hyd"]),
                    "hour": hour,
                    "orders": float(rng.gamma(3, 20) + shift * 5),
                    "speed": float(rng.normal(18 + shift, 4)),
                    "distance": float(rng.uniform(1, 12)),
                })
    frame = pd.DataFrame(rows)
    frame.loc[frame.sample(20, random_state=1).index, "speed"] = np.nan
    return frame


def close(fig):
    assert fig.__class__.__name__ == "Figure"
    plt.close(fig)


@pytest.mark.parametrize("kind", ["line", "step", "area", "scatter", "bar", "barh"])
def test_series_kinds(frame, kind):
    close(plotter.plot_columns(frame, ["orders", "speed"], x="hour", agg="mean", kind=kind))


@pytest.mark.parametrize("kind", ["hist", "kde", "box", "violin", "ecdf"])
def test_distribution_kinds(frame, kind):
    close(plotter.plot_distribution(frame, "speed", group="variant", kind=kind))


@pytest.mark.parametrize("mode", ["grid", "overlay", "twin"])
def test_modes_produce_expected_axes(frame, mode):
    fig = plotter.plot_columns(frame, ["orders", "speed", "distance"], x="hour",
                               agg="mean", mode=mode)
    expected = 3 if mode == "grid" else 1 if mode == "overlay" else 3
    assert len(fig.axes) >= expected
    plt.close(fig)


def test_grid_lays_out_one_panel_per_column(frame):
    fig = plotter.plot_columns(frame, ["orders", "speed", "distance"], x="hour",
                               agg="mean", mode="grid", ncols=2)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["orders", "speed", "distance"]
    plt.close(fig)


def test_groups_become_separate_series(frame):
    fig = plotter.plot_columns(frame, ["orders"], x="hour", group="variant",
                               agg="mean", kind="line")
    labels = {text.get_text() for text in fig.legends[0].get_texts()} if fig.legends else set()
    labels |= {text.get_text() for ax in fig.axes if ax.get_legend()
               for text in ax.get_legend().get_texts()}
    assert {"test", "control"} <= labels
    plt.close(fig)


def test_twin_mode_adds_one_axis_per_column(frame):
    fig = plotter.plot_multi_axis(frame, ["orders", "speed", "distance"], x="hour", agg="mean")
    assert len(fig.axes) == 3
    assert [ax.get_ylabel() for ax in fig.axes] == ["orders", "speed", "distance"]
    plt.close(fig)


def test_aggregation_collapses_rows(frame):
    aggregated = plotter._aggregate(frame, "hour", ["orders"], "variant", "mean")
    assert len(aggregated) == 24 * 2
    assert aggregated["orders"].notna().all()
    # x and group can be the same column without blowing up
    same = plotter._aggregate(frame, "variant", ["orders"], "variant", "mean")
    assert len(same) == 2


def test_describe_distribution_reports_shape(frame):
    stats = plotter.describe_distribution(frame, "speed", group="variant")
    assert list(stats["group"]) == ["control", "test"]
    assert {"mean", "std", "skew", "outlier_share", "p50", "missing"} <= set(stats.columns)
    assert stats["missing"].sum() == 20
    control = stats[stats["group"] == "control"].iloc[0]
    assert control["p25"] <= control["p50"] <= control["p75"]

    overall = plotter.describe_distribution(frame, "speed")
    assert "group" not in overall.columns
    assert len(overall) == 1


def test_compare_distributions_overlays_groups(frame):
    fig = plotter.compare_distributions(frame, "speed", "variant", kind="kde")
    lines = [line for ax in fig.axes for line in ax.get_lines()]
    colors = {line.get_color() for line in lines}
    assert len(colors) >= 2          # one colour per group
    plt.close(fig)

    fig = plotter.compare_distributions(frame, "orders", "variant", with_box=True)
    assert len(fig.axes) == 2        # curves plus the companion box plot
    plt.close(fig)


def test_compare_groups_and_normalisation(frame):
    close(plotter.compare_groups(frame, ["orders", "speed"], group="variant", agg="mean"))
    close(plotter.compare_groups(frame, ["orders"], group="variant", x="hour",
                                 agg="mean", kind="line"))
    close(plotter.compare_groups(frame, ["orders"], group="variant", agg="mean",
                                 normalize_to="control"))
    with pytest.raises(KeyError):
        plotter.compare_groups(frame, ["orders"], group="variant", normalize_to="missing")


def test_group_stats_and_correlation(frame):
    stats = plotter.group_stats(frame, ["orders", "speed"], "variant")
    assert len(stats) == 2
    close(plotter.plot_correlation(frame, ["orders", "speed", "distance"]))


def test_kde_and_ecdf_helpers():
    grid, density = plotter.kde_curve([1, 2, 2, 3, 4, 10])
    assert grid.shape == density.shape
    assert density.min() >= 0
    assert plotter.kde_curve([1]) is None

    values, share = plotter.ecdf_points([3, 1, 2])
    assert list(values) == [1, 2, 3]
    assert share[-1] == pytest.approx(1.0)
    assert plotter.ecdf_points([]) is None


def test_colour_mapping_is_stable():
    first = plotter.color_map(["test", "control"])
    assert first == plotter.color_map(["test", "control"])
    assert first["test"] != first["control"]
    assert plotter.group_levels(pd.DataFrame({"g": ["b", "a", "a"]}), "g") == ["a", "b"]


def test_bad_arguments_raise(frame):
    with pytest.raises(KeyError):
        plotter.plot_columns(frame, ["missing"])
    with pytest.raises(ValueError):
        plotter.plot_columns(frame, ["orders"], kind="pie")
    with pytest.raises(ValueError):
        plotter.plot_columns(frame, ["orders"], mode="sideways")
    with pytest.raises(ValueError):
        plotter.plot_columns(frame, [])
    with pytest.raises(KeyError):
        plotter.plot_distribution(frame, "orders", group="missing")


def test_all_nan_column_does_not_crash(frame):
    frame = frame.assign(empty=np.nan)
    close(plotter.plot_distribution(frame, "empty"))
    stats = plotter.describe_distribution(frame, "empty")
    assert stats["count"].iloc[0] == 0
