"""Stateless DataFrame plotting: multi-column, multi-axis, multi-group.

Nothing here holds state - every function takes a DataFrame and returns a
matplotlib ``Figure``, so the same call works in a notebook, a script or a
test.

Three ideas cover most of it:

* ``mode`` decides how several columns share the canvas - ``grid`` (one panel
  each), ``overlay`` (one axes, shared scale) or ``twin`` (one axes, one y-axis
  per column, for columns whose scales differ).
* ``group`` splits every series by a categorical column (``test`` vs
  ``control``) and gives each level its own colour plus a legend entry.
* ``kind`` picks the mark: ``line``, ``bar``, ``barh``, ``scatter``, ``area``,
  ``step``, or a distribution mark (``hist``, ``kde``, ``box``, ``violin``,
  ``ecdf``).

Needs pandas, numpy and matplotlib::

    pip install 'sameer-graph-lib[plot]' pandas
"""

from __future__ import annotations

from typing import Iterable, Sequence

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Plotter helpers require pandas and numpy: "
        "pip install 'sameer-graph-lib[route]'"
    ) from exc


#: Colour-blind friendly default cycle used for groups and columns.
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#EECA3B", "#9D755D", "#FF9DA6", "#79706E",
]

SERIES_KINDS = ("line", "step", "area", "bar", "barh", "scatter")
DISTRIBUTION_KINDS = ("hist", "kde", "box", "violin", "ecdf")
ALL_KINDS = SERIES_KINDS + DISTRIBUTION_KINDS


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Plotting requires matplotlib: pip install 'sameer-graph-lib[plot]'"
        ) from exc
    return plt


def _as_columns(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _check_columns(df, columns) -> list:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not in the frame: {missing}")
    return list(columns)


def _numeric(series) -> "pd.Series":
    return pd.to_numeric(series, errors="coerce")


def _clean(values) -> "np.ndarray":
    arr = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    return arr[np.isfinite(arr)]


def group_levels(df, group, order=None, max_groups=None) -> list:
    """Ordered, de-duplicated levels of a grouping column."""
    if group is None:
        return [None]
    if group not in df.columns:
        raise KeyError(f"Group column {group!r} is not in the frame")
    if order is not None:
        return list(order)
    levels = df[group].dropna().unique().tolist()
    try:
        levels = sorted(levels)
    except TypeError:
        pass
    if max_groups and len(levels) > max_groups:
        counts = df[group].value_counts()
        levels = [lv for lv in counts.index[:max_groups]]
    return levels


def color_map(levels, palette=None) -> dict:
    """Stable ``level -> colour`` mapping so repeated plots stay comparable."""
    colors = list(palette or PALETTE)
    return {level: colors[index % len(colors)] for index, level in enumerate(levels)}


# --------------------------------------------------------------------------- #
# distribution maths (no scipy required)
# --------------------------------------------------------------------------- #
def kde_curve(values, bandwidth=None, points: int = 256, cut: float = 3.0,
              max_samples: int = 20000, seed: int = 0):
    """Gaussian KDE on a fixed grid, returned as ``(grid, density)``.

    Uses a Silverman bandwidth and subsamples very large inputs so the call
    stays cheap; returns ``None`` when there is not enough data.
    """
    v = _clean(values)
    if v.size < 2:
        return None
    if v.size > max_samples:
        v = np.random.default_rng(seed).choice(v, size=max_samples, replace=False)
    std = float(v.std(ddof=1))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(q75 - q25)
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if sigma <= 0:
        sigma = abs(float(v.mean())) * 1e-3 or 1e-6
    bw = float(bandwidth or 0.9 * sigma * v.size ** (-0.2))
    if bw <= 0:
        bw = 1e-6
    grid = np.linspace(v.min() - cut * bw, v.max() + cut * bw, points)
    z = (grid[:, None] - v[None, :]) / bw
    density = np.exp(-0.5 * z * z).sum(axis=1) / (v.size * bw * np.sqrt(2 * np.pi))
    return grid, density


def ecdf_points(values):
    """Empirical CDF as ``(sorted values, cumulative share)``."""
    v = np.sort(_clean(values))
    if v.size == 0:
        return None
    return v, np.arange(1, v.size + 1) / v.size


def describe_distribution(df, column, group=None, order=None, percentiles=(0.01, 0.25, 0.5, 0.75, 0.99)):
    """Distribution stats for a column, optionally one row per group.

    Includes the usual moments plus missing counts, IQR outlier share and the
    IQR fences, which is normally what you want before trusting a mean.
    """
    _check_columns(df, [column])
    levels = group_levels(df, group, order=order)

    rows = []
    for level in levels:
        subset = df if level is None else df[df[group] == level]
        raw = _numeric(subset[column])
        values = _clean(raw)
        row = {
            "group": "all" if level is None else level,
            "count": int(values.size),
            "missing": int(raw.isna().sum()),
            "n_unique": int(pd.Series(values).nunique()) if values.size else 0,
        }
        if values.size:
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            series = pd.Series(values)
            row.update({
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "min": float(values.min()),
                "max": float(values.max()),
                "iqr": float(iqr),
                "skew": float(series.skew()) if values.size > 2 else np.nan,
                "kurtosis": float(series.kurtosis()) if values.size > 3 else np.nan,
                "outlier_low": float(low),
                "outlier_high": float(high),
                "outlier_share": float(((values < low) | (values > high)).mean()),
            })
            for p in percentiles:
                row[f"p{p * 100:g}"] = float(np.percentile(values, p * 100))
        rows.append(row)

    frame = pd.DataFrame(rows)
    if group is None:
        frame = frame.drop(columns=["group"])
    return frame


# --------------------------------------------------------------------------- #
# low level marks
# --------------------------------------------------------------------------- #
def _draw_series(ax, x, y, kind, label=None, color=None, alpha=None, marker=None,
                 linewidth=1.8, width=0.8, offset=0.0, **kwargs):
    """Draw one series with the requested mark onto an existing axes."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    if kind == "line":
        ax.plot(x, y, label=label, color=color, alpha=alpha, marker=marker,
                linewidth=linewidth, **kwargs)
    elif kind == "step":
        ax.step(x, y, where="mid", label=label, color=color, alpha=alpha,
                linewidth=linewidth, **kwargs)
    elif kind == "area":
        ax.fill_between(x, 0, y, label=label, color=color,
                        alpha=0.35 if alpha is None else alpha, **kwargs)
        ax.plot(x, y, color=color, linewidth=linewidth)
    elif kind == "scatter":
        ax.scatter(x, y, label=label, color=color,
                   alpha=0.75 if alpha is None else alpha,
                   marker=marker or "o", **kwargs)
    elif kind == "bar":
        ax.bar(x + offset, y, width=width, label=label, color=color, alpha=alpha,
               edgecolor="#2c3e50", linewidth=0.5, **kwargs)
    elif kind == "barh":
        ax.barh(x + offset, y, height=width, label=label, color=color, alpha=alpha,
                edgecolor="#2c3e50", linewidth=0.5, **kwargs)
    else:
        raise ValueError(f"kind must be one of {ALL_KINDS}, got {kind!r}")


def _shared_bins(arrays, bins):
    """Common bin edges so grouped histograms are comparable."""
    pool = np.concatenate([a for a in arrays if a.size]) if arrays else np.array([])
    if pool.size == 0:
        return bins
    if isinstance(bins, int):
        return np.linspace(pool.min(), pool.max(), bins + 1)
    return bins


def _draw_distribution(ax, series, kind="hist", bins=30, density=False,
                       alpha=None, linewidth=2.0, fill=True, rug=False,
                       show_mean=False, **kwargs):
    """Draw ``series`` (a list of ``(label, colour, values)``) as distributions."""
    arrays = [(label, color, _clean(values)) for label, color, values in series]
    arrays = [item for item in arrays if item[2].size]
    if not arrays:
        ax.text(0.5, 0.5, "no numeric data", ha="center", va="center", transform=ax.transAxes)
        return

    if kind == "hist":
        edges = _shared_bins([a for _, _, a in arrays], bins)
        single = len(arrays) == 1
        for label, color, values in arrays:
            ax.hist(values, bins=edges, label=label, color=color, density=density,
                    alpha=(0.75 if single else 0.45) if alpha is None else alpha,
                    histtype="stepfilled" if fill else "step",
                    edgecolor=color, linewidth=1.2, **kwargs)
        ax.set_ylabel("density" if density else "count")

    elif kind == "kde":
        for label, color, values in arrays:
            curve = kde_curve(values)
            if curve is None:
                continue
            grid, dens = curve
            ax.plot(grid, dens, label=label, color=color, linewidth=linewidth, **kwargs)
            if fill:
                ax.fill_between(grid, 0, dens, color=color,
                                alpha=0.15 if alpha is None else alpha)
        ax.set_ylabel("density")

    elif kind == "ecdf":
        for label, color, values in arrays:
            points = ecdf_points(values)
            if points is None:
                continue
            xs, ys = points
            ax.step(xs, ys, where="post", label=label, color=color,
                    linewidth=linewidth, **kwargs)
        ax.set_ylabel("cumulative share")
        ax.set_ylim(0, 1.02)

    elif kind in ("box", "violin"):
        labels = [label for label, _, _ in arrays]
        data = [values for _, _, values in arrays]
        colors = [color for _, color, _ in arrays]
        positions = np.arange(1, len(data) + 1)
        if kind == "box":
            artists = ax.boxplot(data, positions=positions, patch_artist=True,
                                 widths=0.6, showmeans=show_mean)
            for patch, color in zip(artists["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.65 if alpha is None else alpha)
                patch.set_edgecolor("#2c3e50")
            for median in artists["medians"]:
                median.set_color("#2c3e50")
        else:
            artists = ax.violinplot(data, positions=positions, showmeans=show_mean,
                                    showmedians=True, widths=0.8)
            for body, color in zip(artists["bodies"], colors):
                body.set_facecolor(color)
                body.set_alpha(0.6 if alpha is None else alpha)
                body.set_edgecolor("#2c3e50")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        return

    else:
        raise ValueError(f"Distribution kind must be one of {DISTRIBUTION_KINDS}, got {kind!r}")

    if rug:
        for _, color, values in arrays:
            sample = values if values.size <= 2000 else np.random.default_rng(0).choice(values, 2000, replace=False)
            ax.plot(sample, np.full(sample.shape, ax.get_ylim()[0]), marker="|",
                    linestyle="none", color=color, alpha=0.35, markersize=6)

    if show_mean:
        for _, color, values in arrays:
            ax.axvline(float(values.mean()), color=color, linestyle="--", linewidth=1.2, alpha=0.8)


def _axis_positions(values, categorical):
    """Return plotting positions plus tick labels for an x axis."""
    if not categorical:
        return np.asarray(values, dtype=float), None, None
    categories = list(pd.unique(pd.Series(values)))
    lookup = {category: index for index, category in enumerate(categories)}
    return (np.asarray([lookup[v] for v in values], dtype=float),
            np.arange(len(categories)),
            [str(c) for c in categories])


def _is_categorical(series, kind) -> bool:
    if kind in ("bar", "barh"):
        return True
    return not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series)


def _aggregate(df, x, ycols, group, agg):
    keys = list(dict.fromkeys(k for k in ([x] if x else []) + ([group] if group else []) if k))
    if not keys:
        return df
    columns = [c for c in ycols if c not in keys]
    if not columns:
        return df
    return df.groupby(keys, dropna=False, sort=True)[columns].agg(agg).reset_index()


# --------------------------------------------------------------------------- #
# main entry points
# --------------------------------------------------------------------------- #
def plot_columns(
    df,
    y,
    x=None,
    group=None,
    kind: str = "line",
    mode: str = "grid",
    agg=None,
    order=None,
    ncols: int = 2,
    figsize=None,
    panel_size: tuple[float, float] = (6.5, 4.0),
    sharex: bool = False,
    sharey: bool = False,
    palette=None,
    title: str | None = None,
    legend: bool = True,
    grid: bool = True,
    bins=30,
    density: bool = False,
    alpha=None,
    marker=None,
    linewidth: float = 1.8,
    bar_width: float = 0.8,
    rotate_xticks: float = 0,
    ylabels=None,
    axes=None,
    **plot_kwargs,
):
    """Plot any number of columns, in panels, overlaid, or on twin y-axes.

    Parameters
    ----------
    y:
        One column name or a list of them.
    x:
        Column for the x axis. ``None`` uses the row order.
    group:
        Categorical column (``variant``, ``city``, ...). Every level gets its
        own colour and legend entry, so ``test`` vs ``control`` compares
        directly.
    kind:
        ``line``, ``step``, ``area``, ``bar``, ``barh``, ``scatter``, or a
        distribution mark: ``hist``, ``kde``, ``box``, ``violin``, ``ecdf``
        (those ignore ``x`` and describe each column instead).
    mode:
        ``grid`` gives one panel per column, ``overlay`` puts them on one axes,
        ``twin`` gives each column its own y-axis on a shared x axis.
    agg:
        Optional aggregation (``mean``, ``sum``, ...) applied per x and group
        before plotting, so raw row-level frames can be charted directly.

    Returns the matplotlib ``Figure``.
    """
    plt = _require_matplotlib()

    ycols = _check_columns(df, _as_columns(y))
    if not ycols:
        raise ValueError("Pass at least one column in y")
    if kind not in ALL_KINDS:
        raise ValueError(f"kind must be one of {ALL_KINDS}, got {kind!r}")
    if mode not in ("grid", "overlay", "twin"):
        raise ValueError("mode must be one of: grid, overlay, twin")
    if group is not None:
        _check_columns(df, [group])

    data = df
    if agg:
        data = _aggregate(df, x, ycols, group, agg)
    if x is not None:
        _check_columns(data, [x])

    levels = group_levels(data, group, order=order)
    colors = color_map(levels if group else ycols, palette=palette)
    distribution = kind in DISTRIBUTION_KINDS

    def series_for(column):
        out = []
        for level in levels:
            subset = data if level is None else data[data[group] == level]
            if level is None:
                label = str(column)
                color = colors[column] if not group else colors[None]
            else:
                label = str(level) if (len(ycols) == 1 or mode == "grid") else f"{column} | {level}"
                color = colors[level]
            out.append((label, color, subset))
        return out

    # ---- build the axes ------------------------------------------------ #
    if mode == "grid":
        panels = len(ycols)
        ncols = max(1, min(ncols, panels))
        nrows = int(np.ceil(panels / ncols))
        if axes is None:
            figsize = figsize or (panel_size[0] * ncols, panel_size[1] * nrows)
            fig, axes_array = plt.subplots(nrows, ncols, figsize=figsize,
                                           sharex=sharex, sharey=sharey, squeeze=False)
            axes_list = list(axes_array.ravel())
        else:
            axes_list = list(np.ravel(axes))
            fig = axes_list[0].figure
        target_axes = axes_list[:panels]
        for extra in axes_list[panels:]:
            extra.axis("off")
    else:
        if axes is None:
            figsize = figsize or (panel_size[0] * 1.7, panel_size[1] * 1.35)
            fig, base_ax = plt.subplots(1, 1, figsize=figsize)
        else:
            base_ax = np.ravel(axes)[0]
            fig = base_ax.figure
        target_axes = [base_ax] * len(ycols)

    twin_axes = []
    handles_seen = {}

    for index, column in enumerate(ycols):
        if mode == "twin" and index > 0:
            ax = target_axes[0].twinx()
            ax.spines["right"].set_position(("outward", 55 * (index - 1)))
            twin_axes.append(ax)
        else:
            ax = target_axes[index]

        column_series = series_for(column)

        if distribution:
            _draw_distribution(
                ax, [(label, color, subset[column]) for label, color, subset in column_series],
                kind=kind, bins=bins, density=density, alpha=alpha,
                linewidth=linewidth, **plot_kwargs,
            )
            ax.set_xlabel(str(column))
        else:
            x_series = data[x] if x is not None else pd.Series(np.arange(len(data)))
            categorical = _is_categorical(x_series, kind)
            positions_all, ticks, tick_labels = _axis_positions(x_series, categorical)
            count = len(column_series)
            width = bar_width / count if kind in ("bar", "barh") and count > 1 else bar_width

            for slot, (label, color, subset) in enumerate(column_series):
                if x is not None:
                    xs, _, _ = _axis_positions(subset[x], categorical) if categorical else (
                        np.asarray(_numeric(subset[x]), dtype=float), None, None)
                    if categorical:
                        lookup = {c: i for i, c in enumerate(pd.unique(pd.Series(x_series)))}
                        xs = np.asarray([lookup.get(v, np.nan) for v in subset[x]], dtype=float)
                else:
                    xs = np.asarray(subset.index, dtype=float)
                offset = ((slot - (count - 1) / 2) * width) if kind in ("bar", "barh") and count > 1 else 0.0
                _draw_series(ax, xs, _numeric(subset[column]), kind=kind, label=label,
                             color=color, alpha=alpha, marker=marker, linewidth=linewidth,
                             width=width, offset=offset, **plot_kwargs)

            if categorical and ticks is not None:
                if kind == "barh":
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(tick_labels)
                else:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(tick_labels, rotation=rotate_xticks,
                                       ha="right" if rotate_xticks else "center")
            elif rotate_xticks:
                ax.tick_params(axis="x", labelrotation=rotate_xticks)

            if kind == "barh":
                ax.set_xlabel(str(column))
                ax.set_ylabel(str(x) if x is not None else "row")
            else:
                ax.set_xlabel(str(x) if x is not None else "row")

        if mode != "twin" or index == 0:
            label_text = None
            if ylabels is not None:
                label_text = ylabels[index] if not isinstance(ylabels, str) else ylabels
            elif not distribution and kind != "barh":
                label_text = str(column)
            if label_text:
                ax.set_ylabel(label_text)
        else:
            axis_color = colors.get(column, "#2c3e50") if not group else "#2c3e50"
            ax.set_ylabel(str(column), color=axis_color)
            ax.tick_params(axis="y", colors=axis_color)

        if mode == "grid":
            ax.set_title(str(column), fontsize=11, fontweight="bold")
        if grid:
            ax.grid(True, linewidth=0.4, alpha=0.3)
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_seen.setdefault(label, handle)

    # ---- legend and title ---------------------------------------------- #
    show_legend = bool(legend and handles_seen and kind not in ("box", "violin"))
    # one shared legend under the title for panels, an in-axes legend otherwise
    shared_legend = show_legend and mode == "grid" and (group is not None or len(ycols) > 1)
    if show_legend and not shared_legend:
        target_axes[0].legend(handles_seen.values(), handles_seen.keys(),
                              loc="best", fontsize=9, framealpha=0.9)

    if title is None:
        bits = [", ".join(str(c) for c in ycols)]
        if group:
            bits.append(f"by {group}")
        if agg:
            bits.append(f"{agg} per {x}" if x else str(agg))
        title = f"{kind} | " + "  ".join(bits)

    fig.tight_layout(rect=(0, 0, 1, 0.90) if shared_legend else (0, 0, 1, 0.95))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    if shared_legend:
        fig.legend(handles_seen.values(), handles_seen.keys(),
                   loc="upper center", ncol=min(len(handles_seen), 6),
                   frameon=False, bbox_to_anchor=(0.5, 0.955))
    return fig


def plot_multi_axis(df, y, x=None, group=None, kind="line", **kwargs):
    """Several columns on one chart, each with its own y-axis (different scales)."""
    return plot_columns(df, y=y, x=x, group=group, kind=kind, mode="twin", **kwargs)


def plot_grid(df, y, x=None, group=None, kind="line", **kwargs):
    """One panel per column, groups overlaid inside each panel."""
    return plot_columns(df, y=y, x=x, group=group, kind=kind, mode="grid", **kwargs)


def plot_overlay(df, y, x=None, group=None, kind="line", **kwargs):
    """All columns on a single shared axes."""
    return plot_columns(df, y=y, x=x, group=group, kind=kind, mode="overlay", **kwargs)


def plot_distribution(
    df,
    column,
    group=None,
    kind: str = "hist",
    order=None,
    bins=30,
    density: bool = False,
    palette=None,
    figsize: tuple[float, float] = (11, 6),
    title: str | None = None,
    legend: bool = True,
    grid: bool = True,
    show_stats: bool = True,
    show_mean: bool = False,
    rug: bool = False,
    alpha=None,
    linewidth: float = 2.0,
    ax=None,
    **plot_kwargs,
):
    """Check the distribution of one column, split by group when asked.

    ``show_stats`` prints count / mean / median / std / skew in the corner so
    the shape and the numbers are visible together.
    """
    plt = _require_matplotlib()
    _check_columns(df, [column])
    levels = group_levels(df, group, order=order)
    colors = color_map(levels, palette=palette)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    series = []
    for level in levels:
        subset = df if level is None else df[df[group] == level]
        label = str(column) if level is None else str(level)
        series.append((label, colors[level], subset[column]))

    _draw_distribution(ax, series, kind=kind, bins=bins, density=density, alpha=alpha,
                       linewidth=linewidth, rug=rug, show_mean=show_mean, **plot_kwargs)

    if show_stats:
        stats = describe_distribution(df, column, group=group, order=order)
        lines = []
        for _, row in stats.iterrows():
            name = row["group"] if group else "all"
            lines.append(
                f"{name}: n={row['count']:,.0f}  mean={row.get('mean', np.nan):,.2f}  "
                f"med={row.get('p50', np.nan):,.2f}  sd={row.get('std', np.nan):,.2f}  "
                f"skew={row.get('skew', np.nan):,.2f}"
            )
        ax.text(0.99, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="#d0d0d0"))

    if legend and group is not None and kind not in ("box", "violin"):
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
                  title=str(group))
    if grid:
        ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.set_xlabel(str(column))
    ax.set_title(title or (f"Distribution of {column}" + (f" by {group}" if group else "")),
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def compare_distributions(
    df,
    column,
    group,
    kind: str = "kde",
    order=None,
    with_box: bool = False,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    **kwargs,
):
    """Overlay one variable's distribution for every group, one colour each.

    ``with_box`` adds a companion box plot under the curves, which makes the
    medians and spreads easy to read off next to the shapes.
    """
    plt = _require_matplotlib()
    _check_columns(df, [column, group])

    if not with_box:
        return plot_distribution(df, column, group=group, kind=kind, order=order,
                                 figsize=figsize, title=title, **kwargs)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    plot_distribution(df, column, group=group, kind=kind, order=order,
                      ax=axes[0], title="", **kwargs)
    axes[0].set_title("")
    axes[0].set_xlabel("")

    levels = group_levels(df, group, order=order)
    colors = color_map(levels, palette=kwargs.get("palette"))
    data = [_clean(df[df[group] == level][column]) for level in levels]
    try:
        artists = axes[1].boxplot(data, orientation="horizontal", patch_artist=True, widths=0.6)
    except TypeError:  # matplotlib below 3.10 only understands vert=
        artists = axes[1].boxplot(data, vert=False, patch_artist=True, widths=0.6)
    for patch, level in zip(artists["boxes"], levels):
        patch.set_facecolor(colors[level])
        patch.set_alpha(0.65)
        patch.set_edgecolor("#2c3e50")
    axes[1].set_yticks(range(1, len(levels) + 1))
    axes[1].set_yticklabels([str(level) for level in levels])
    axes[1].invert_yaxis()   # keep the group order matching the legend above
    axes[1].set_xlabel(str(column))
    axes[1].grid(True, axis="x", linewidth=0.4, alpha=0.3)

    fig.suptitle(title or f"{column} distribution by {group}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def compare_groups(
    df,
    y,
    group,
    x=None,
    agg: str = "mean",
    kind: str = "bar",
    order=None,
    normalize_to=None,
    **kwargs,
):
    """Aggregate columns per group and chart the comparison (test vs control).

    ``normalize_to`` expresses every group as a share of one baseline level,
    which is usually what an A/B readout wants.
    """
    ycols = _check_columns(df, _as_columns(y))
    _check_columns(df, [group])
    if x is None:
        summary = df.groupby(group, dropna=False)[ycols].agg(agg).reset_index()
        if normalize_to is not None:
            if normalize_to not in summary[group].values:
                raise KeyError(f"normalize_to={normalize_to!r} is not a level of {group!r}")
            base = summary.loc[summary[group] == normalize_to, ycols].iloc[0]
            summary[ycols] = summary[ycols].div(base.replace(0, np.nan))
        kwargs.setdefault("title", f"{agg} of {', '.join(ycols)} by {group}"
                                   + (f" (relative to {normalize_to})" if normalize_to is not None else ""))
        return plot_columns(summary, y=ycols, x=group, kind=kind, mode="grid",
                            order=order, **kwargs)
    return plot_columns(df, y=ycols, x=x, group=group, kind=kind, agg=agg,
                        order=order, **kwargs)


def group_stats(df, y, group, agg=("count", "mean", "median", "std")):
    """Tidy per-group aggregates for the given columns."""
    ycols = _check_columns(df, _as_columns(y))
    _check_columns(df, [group])
    table = df.groupby(group, dropna=False)[ycols].agg(list(agg))
    return table.reset_index()


def plot_correlation(
    df,
    columns=None,
    method: str = "pearson",
    figsize: tuple[float, float] = (9, 8),
    cmap: str = "RdBu_r",
    annotate: bool = True,
    title: str | None = None,
    ax=None,
):
    """Correlation heatmap for the numeric columns you pass."""
    plt = _require_matplotlib()
    frame = df[_check_columns(df, _as_columns(columns))] if columns else df.select_dtypes("number")
    matrix = frame.corr(method=method)
    if matrix.empty:
        raise ValueError("No numeric columns to correlate")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    values = matrix.to_numpy(dtype=float)
    mesh = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns], rotation=90, fontsize=8)
    ax.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index], fontsize=8)
    fig.colorbar(mesh, ax=ax, label=f"{method} r", shrink=0.85)

    if annotate and values.size <= 400:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(value) > 0.6 else "#2c3e50")

    ax.set_title(title or f"{method.title()} correlation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
