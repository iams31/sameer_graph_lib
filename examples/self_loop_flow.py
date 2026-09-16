"""A cluster that feeds itself, in a pickup -> drop flow graph.

Some exports carry rows whose pickup and drop are the same cluster - a trip that
starts and ends in the same place. That row is real, and it is often the largest
single route the cluster has, so what the flow plot does with it matters.

By default the loop is kept out of the drawing (an arrow from a node to itself
says nothing), but it is not kept out of the arithmetic:

* it does not use up a place in the top x, so asking for three sources gives
  three real sources rather than two plus a hole
* with ``rest=True`` its volume lands in the rest node, so the drawn routes plus
  the rest still come back to the cluster's real total

Pass ``exclude_self_loops=False`` and it is drawn instead, as a ring on the
cluster it loops on, carrying its own metric table. It then does take a place in
the top x - because you asked for it.

Run:

    uv run --extra analysis python examples\\self_loop_flow.py

Saves ``self_loop_counted.png`` and ``self_loop_drawn.png``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sameer_graph_lib import RouteExplorer


def make_frame() -> pd.DataFrame:
    """Routes into and out of A1, including a large A1 -> A1."""
    routes = [
        ("A1", "A1", 9000.0, 1.2),      # the biggest route A1 has, in or out
        ("C1", "A1", 3000.0, 4.0),
        ("C2", "A1", 2000.0, 3.0),
        ("C3", "A1", 1000.0, 2.5),
        ("C4", "A1", 300.0, 7.0),
        ("C5", "A1", 120.0, 6.0),
        ("A1", "B1", 500.0, 5.0),
        ("A1", "B2", 400.0, 6.0),
        ("A1", "B3", 150.0, 8.0),
    ]
    return pd.DataFrame([
        {
            "pickup_cluster": pickup,
            "drop_cluster": drop,
            "week_period": week_period,
            "orders": orders,
            "requests": orders * 1.4,
            "avg_distance": distance,
        }
        for pickup, drop, orders, distance in routes
        for week_period in ("weekday", "weekend")
    ])


def report(explorer: RouteExplorer) -> dict:
    """Print what the loop does to the top-k and to the totals."""
    graph = explorer.graph
    inbound = graph.node_value("A1", "orders", "in")

    print("every source of A1, ranked by orders:")
    for name, value in graph.top_sources("A1", metric="orders"):
        mark = "   <- the self loop" if name == "A1" else ""
        print(f"   {name:<4} {value:>12,.1f}{mark}")

    sub = graph.flow_subgraph("A1", upstream=3, downstream=0,
                              metric="orders", rest=True)
    drawn = [(data["cluster"], data["value"]) for _, data in sub.nodes(data=True)
             if not data.get("is_focus") and not data.get("is_rest")]
    rest = [data for _, data in sub.nodes(data=True) if data.get("is_rest")][0]
    total = sum(value for _, value in drawn) + rest["value"]

    print()
    print(f"asked for the top 3 sources, drew : {[name for name, _ in drawn]}")
    print(f"the rest node stands for          : {rest['partners']}")
    print(f"drawn {sum(v for _, v in drawn):>14,.1f}")
    print(f"rest  {rest['value']:>14,.1f}")
    print(f"total {total:>14,.1f}   every route into A1 {inbound:>14,.1f}")
    assert round(total, 6) == round(inbound, 6), "the top-k and the rest must close"

    # A1 -> A1 is a route out of A1 as well, so it lands in the rest node on the
    # drop side too. That rest node stands for A1 itself, and the table drawn on
    # it is A1 measured as a pickup - which is why it repeats the focus cluster's
    # own number. The edge into it is the loop, and that is what closes the sum.
    outbound = graph.node_value("A1", "orders", "out")
    drops = graph.flow_subgraph("A1", upstream=0, downstream=3, metric="orders",
                                rest=True)
    drop_rest = [data for _, data in drops.nodes(data=True)
                 if data.get("is_rest")][0]
    drawn_out = sum(data["value"] for _, data in drops.nodes(data=True)
                    if not data.get("is_focus") and not data.get("is_rest"))
    print()
    print(f"the same on the way out, where the loop is the whole rest: "
          f"{drop_rest['partners']}")
    print(f"drawn {drawn_out:>14,.1f}")
    print(f"rest  {drop_rest['value']:>14,.1f}")
    print(f"total {drawn_out + drop_rest['value']:>14,.1f}"
          f"   every route out of A1 {outbound:>14,.1f}")
    return {"inbound": inbound, "outbound": outbound}


def save(figure, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path.name}")


def main() -> None:
    frame = make_frame()
    explorer = RouteExplorer(
        frame,
        grain_cols=("week_period",),
        sum_metrics=["orders", "requests"],
        mean_metrics=["avg_distance"],
        weight_col="requests",
        metric="orders",
    )
    report(explorer)
    here = Path(__file__).resolve().parent

    # the default: the loop is not drawn, but rest accounts for it
    save(explorer.plot(
        "A1", upstream=3, downstream=3, rest=True,
        node_metrics=["orders"], edge_metrics=["orders", "avg_distance"],
        title="A1 -> A1 left out of the drawing, counted in rest",
    ), here / "self_loop_counted.png")

    # asked for, so drawn - as a ring on A1, and it takes a place in the top 3
    save(explorer.plot(
        "A1", upstream=3, downstream=3, exclude_self_loops=False,
        node_metrics=["orders"], edge_metrics=["orders", "avg_distance"],
        title="A1 -> A1 drawn as a ring, and holding a place in the top 3",
    ), here / "self_loop_drawn.png")

    print()
    print("with the loop drawn it holds one of the three places, so raise the")
    print("top-k if you want the same number of outside clusters:")
    for keep, kept_loop in ((3, False), (3, True), (4, True)):
        sub = explorer.graph.flow_subgraph(
            "A1", upstream=keep, downstream=0, metric="orders",
            exclude_self_loops=not kept_loop)
        outside = [data["cluster"] for _, data in sub.nodes(data=True)
                   if not data.get("is_focus") and data["cluster"] != "A1"]
        print(f"   upstream={keep}, loop {'drawn ' if kept_loop else 'hidden'}"
              f" -> {len(outside)} outside clusters {outside}")


if __name__ == "__main__":
    main()
