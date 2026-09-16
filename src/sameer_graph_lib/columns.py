
from __future__ import annotations

import sys

try:
    import pandas as pd
except ImportError as exc:
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
    length = length_metric or _first_hit(candidates, DISTANCE_HINTS)
    duration = ride_time_metric or _first_hit(candidates, DURATION_HINTS,
                                              taken=(length,))
    return length, duration


def suggest_weight(candidates, weight_col=None):
    return weight_col or _first_hit(candidates, WEIGHT_HINTS)


def suggest_columns(df, pickup_col=None, drop_col=None, grain_cols=None,
                    metrics=None, mean_metrics=None, sum_metrics=None,
                    length_metric=None, ride_time_metric=None,
                    weight_col=None) -> dict:
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
    if mean_metrics is not None:
        means = [m for m in mean_metrics if m in chosen]
    elif sum_metrics is not None:
        means = [m for m in chosen if m not in sum_metrics]
    else:
        _, guessed = classify_metrics(df[[c for c in chosen if c in df.columns]])
        means = [m for m in chosen if m in guessed]
    sums = [m for m in chosen if m not in means]

    length, duration = suggest_speed_pair(chosen, length_metric, ride_time_metric)
    return {"pickup_col": pickup, "drop_col": drop, "grain_cols": grains,
            "metrics": chosen, "mean_metrics": means, "sum_metrics": sums,
            "length_metric": length, "ride_time_metric": duration,
            "weight_col": suggest_weight(chosen, weight_col)}


def describe_columns(df, **overrides) -> str:
    proposal = suggest_columns(df, **overrides)
    roles = {}
    if proposal["pickup_col"]:
        roles[proposal["pickup_col"]] = "pickup id"
    if proposal["drop_col"]:
        roles[proposal["drop_col"]] = "drop id"
    for column in proposal["grain_cols"]:
        roles[column] = "grain"
    for column in proposal["metrics"]:
        roles[column] = ("metric (avg)" if column in proposal["mean_metrics"]
                         else "metric (sum)")
    if proposal["length_metric"]:
        roles[proposal["length_metric"]] = "metric (avg, distance)"
    if proposal["ride_time_metric"]:
        roles[proposal["ride_time_metric"]] = "metric (avg, duration)"
    if proposal["weight_col"]:
        roles[proposal["weight_col"]] = "metric (sum, weight)"

    width = max((len(str(c)) for c in df.columns), default=6)
    lines = [f"{'column'.ljust(width)}  {'dtype':<10} {'levels':>7}  role",
             "-" * (width + 30)]
    for column in df.columns:
        series = df[column]
        lines.append(f"{str(column).ljust(width)}  {str(series.dtype):<10} "
                     f"{series.nunique():>7}  {roles.get(column, '-')}")
    return "\n".join(lines)


def _parse(answer, columns, default):
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
        ("mean_metrics", "which of those are averages (rest are summed)",
         proposal["mean_metrics"], False),
        ("length_metric", "distance metric (for speed)", proposal["length_metric"], True),
        ("ride_time_metric", "duration metric (for speed)", proposal["ride_time_metric"], True),
        ("weight_col", "weight column (for averaging)", proposal["weight_col"], True),
    ]
    answers = dict(proposal)
    for key, label, default, single in questions:
        shown = default if single else ", ".join(str(c) for c in default) or "none"
        reply = ask(f"{label} [{shown}]: ")
        picked = _parse(reply, columns, [default] if single and default else
                        ([] if single else default))
        answers[key] = (picked[0] if picked else None) if single else picked
    answers["sum_metrics"] = [m for m in answers["metrics"]
                              if m not in answers["mean_metrics"]]
    return answers
