"""Work out which column is which, by asking or by suggesting.

Column names differ from one export to the next, so nothing here assumes
``pickup_cluster`` or ``week_period``. :func:`suggest_columns` proposes a
mapping from the frame itself, and :func:`ask_columns` shows that proposal and
lets you correct it before the graph is built.

    from sameer_graph_lib import RouteExplorer

    ex = RouteExplorer.ask(df)          # prompts, with the guesses pre-filled
    print(RouteExplorer.preview(df))    # just the guesses, no prompt
"""

from __future__ import annotations

import sys

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Column selection requires pandas: pip install 'sameer-graph-lib[route]'"
    ) from exc

from .route_graph import (_words, classify_metrics, suggest_grains,
                          suggest_ids)

PICKUP_HINTS = ("pickup", "origin", "source", "start", "from")
DROP_HINTS = ("drop", "dest", "target", "end", "to")
DISTANCE_HINTS = ("distance", "dist", "length", "km", "lm", "miles", "metres", "meters")
DURATION_HINTS = ("duration", "time", "mins", "minutes", "minute", "min",
                  "secs", "seconds", "dur", "eta", "tat")
WEIGHT_HINTS = ("requests", "request", "pings", "ping", "trips", "sessions",
                "volume", "exposure", "count", "weight", "n")


def _first_hit(columns, hints, taken=()):
    """First column matching a hint, whole words first.

    Matching on raw substrings is too eager for short hints - ``to`` appears
    inside ``operator`` - so a hint has to be a whole word in the name, and
    only hints of four characters or more fall back to a substring match.
    """
    for hint in hints:
        for column in columns:
            if column not in taken and hint in _words(column):
                return column
    for hint in hints:
        if len(hint) < 4:
            continue
        for column in columns:
            if column not in taken and hint in str(column).lower():
                return column
    return None


def suggest_speed_pair(candidates, length_metric=None, ride_time_metric=None):
    """Guess which metrics are the distance and the duration.

    ``speed`` is distance over time, so the schema needs to know which of your
    columns those are. Without this, speed stays NaN on data that does not
    happen to use the default names.
    """
    length = length_metric or _first_hit(candidates, DISTANCE_HINTS)
    duration = ride_time_metric or _first_hit(candidates, DURATION_HINTS,
                                              taken=(length,))
    return length, duration


def suggest_weight(candidates, weight_col=None):
    """Guess which metric should weight the averages.

    An average of averages is wrong unless it is weighted by how much each
    route carried, so this looks for the volume column.
    """
    return weight_col or _first_hit(candidates, WEIGHT_HINTS)


def suggest_columns(df, pickup_col=None, drop_col=None, grain_cols=None,
                    metrics=None, length_metric=None, ride_time_metric=None,
                    weight_col=None) -> dict:
    """Guess which columns are the ids, the grains and the metrics.

    Anything passed in is kept as given; the rest is inferred from the frame.
    Returns a dict with ``pickup_col``, ``drop_col``, ``grain_cols``,
    ``metrics``, ``length_metric`` and ``ride_time_metric``.
    """
    columns = list(df.columns)
    ids = suggest_ids(df)
    pickup = pickup_col or _first_hit(columns, PICKUP_HINTS) or (ids[0] if ids else None)
    drop = drop_col or _first_hit(columns, DROP_HINTS, taken=(pickup,))
    if drop == pickup:
        drop = None

    taken = {c for c in (pickup, drop) if c}
    grains = list(grain_cols) if grain_cols is not None else [
        c for c in suggest_grains(df, exclude=taken) if c not in taken
    ]
    if metrics is not None:
        chosen = list(metrics)
    else:
        skip = taken | set(grains)
        sums, means = classify_metrics(df, exclude=skip)
        chosen = sums + means
    length, duration = suggest_speed_pair(chosen, length_metric, ride_time_metric)
    return {"pickup_col": pickup, "drop_col": drop, "grain_cols": grains,
            "metrics": chosen, "length_metric": length,
            "ride_time_metric": duration,
            "weight_col": suggest_weight(chosen, weight_col)}


def describe_columns(df, **overrides) -> str:
    """A readable table of every column, its dtype and the role proposed."""
    proposal = suggest_columns(df, **overrides)
    roles = {}
    if proposal["pickup_col"]:
        roles[proposal["pickup_col"]] = "pickup id"
    if proposal["drop_col"]:
        roles[proposal["drop_col"]] = "drop id"
    for column in proposal["grain_cols"]:
        roles[column] = "grain"
    for column in proposal["metrics"]:
        roles[column] = "metric"
    if proposal["length_metric"]:
        roles[proposal["length_metric"]] = "metric (distance)"
    if proposal["ride_time_metric"]:
        roles[proposal["ride_time_metric"]] = "metric (duration)"
    if proposal["weight_col"]:
        roles[proposal["weight_col"]] = "metric (weight)"

    width = max((len(str(c)) for c in df.columns), default=6)
    lines = [f"{'column'.ljust(width)}  {'dtype':<10} {'levels':>7}  role",
             "-" * (width + 30)]
    for column in df.columns:
        series = df[column]
        lines.append(f"{str(column).ljust(width)}  {str(series.dtype):<10} "
                     f"{series.nunique():>7}  {roles.get(column, '-')}")
    return "\n".join(lines)


def _parse(answer, columns, default):
    """Turn an answer into a column list: names, numbers, or blank for default."""
    answer = (answer or "").strip()
    if not answer:
        return list(default)
    if answer.lower() in ("none", "-", "skip"):
        return []
    picked = []
    for token in answer.replace(",", " ").split():
        if token.isdigit():
            index = int(token) - 1
            if not 0 <= index < len(columns):
                raise ValueError(f"{token} is not one of the listed columns")
            picked.append(columns[index])
        elif token in columns:
            picked.append(token)
        else:
            raise ValueError(f"{token!r} is not a column in the frame")
    return list(dict.fromkeys(picked))


def ask_columns(df, input_fn=None, output_fn=print, **overrides) -> dict:
    """Show the proposal and let the caller correct it.

    Answers may be column names or the numbers in the listing; blank keeps the
    proposal and ``none`` clears it. With no interactive input available (a
    script, a test, a CI run) the proposal is returned unchanged.
    """
    proposal = suggest_columns(df, **overrides)
    columns = list(df.columns)
    interactive = input_fn is not None or (
        sys.stdin is not None and getattr(sys.stdin, "isatty", lambda: False)()
    )
    if not interactive:
        return proposal
    ask = input_fn or input

    output_fn(describe_columns(df, **overrides))
    output_fn("")
    output_fn("Answer with names or the numbers below, blank to accept, "
              "'none' to clear.")
    for index, column in enumerate(columns, start=1):
        output_fn(f"  {index:>2}. {column}")

    questions = [
        ("pickup_col", "pickup id column", proposal["pickup_col"], True),
        ("drop_col", "drop id column", proposal["drop_col"], True),
        ("grain_cols", "grain columns", proposal["grain_cols"], False),
        ("metrics", "metric columns", proposal["metrics"], False),
        # speed is distance over time, so the schema has to know which is which
        ("length_metric", "distance metric (for speed)", proposal["length_metric"], True),
        ("ride_time_metric", "duration metric (for speed)", proposal["ride_time_metric"], True),
        # every mean metric is weighted by this, so an average of averages is
        # never taken; it is fixed once the tensors are built
        ("weight_col", "weight column (for averaging)", proposal["weight_col"], True),
    ]
    answers = dict(proposal)
    for key, label, default, single in questions:
        shown = default if single else ", ".join(str(c) for c in default) or "none"
        reply = ask(f"{label} [{shown}]: ")
        picked = _parse(reply, columns, [default] if single and default else
                        ([] if single else default))
        answers[key] = (picked[0] if picked else None) if single else picked
    return answers
