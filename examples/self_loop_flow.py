"""A cluster that feeds itself, in a pickup -> drop flow graph.

Some exports carry rows whose pickup and drop are the same cluster - a trip that
starts and ends in the same place. That row is real, and it is often the largest
single route the cluster has, so what the flow plot does with it matters.

For the cluster you are looking at, the loop gets its own node, drawn beside it
in the focus colour and carrying the loop's own metrics. It is:

* never ranked against the real partners, so it costs no place in the top x
* never folded into a rest node - a rest node only ever stands for other
  clusters
* drawn whatever the top x or ``min_value`` happens to be, because it is not a
  partner competing for a place, it is the cluster itself
* only for the focus - a partner further out that loops on itself is drawn as an
  ordinary partner; focus on it instead and its loop appears

which keeps the accounting whole: on either side, the drawn partners plus the
rest plus the loop come back to the cluster's total.

``self_loops=`` picks the treatment - ``"node"`` (the default), ``"ring"`` to
draw it on the cluster as a ring instead, or ``"hide"`` to leave it out
altogether.

Run:

    uv run --extra analysis python examples/self_loop_flow.py

Saves ``self_loop_node.png``, ``self_loop_ring.png`` and
``self_loop_one_side.png``.
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


def make_modest_loop_frame() -> pd.DataFrame:
    """A loop that is small beside A1's sources and large beside its drops."""
    routes = [
        ("A1", "A1", 1000.0, 1.2),
        ("C1", "A1", 3000.0, 4.0),
        ("C2", "A1", 2000.0, 3.0),
        ("C3", "A1", 1500.0, 2.5),
        ("A1", "B1", 500.0, 5.0),
        ("A1", "B2", 400.0, 6.0),
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
             if data.get("side") == "source" and not data.get("is_rest")]
    rest = [data for _, data in sub.nodes(data=True) if data.get("is_rest")][0]
    total = sum(value for _, value in drawn) + rest["value"]

    loop = [data for _, data in sub.nodes(data=True) if data.get("is_self")][0]
    total += loop["value"]

    print()
    print(f"asked for the top 3 sources, drew : {[name for name, _ in drawn]}")
    print(f"the loop stands on its own        : {loop['value']:,.1f}")
    print(f"the rest node stands for          : {rest['partners']}")
    assert "A1" not in rest["partners"], "rest must never hold A1 -> A1"
    print(f"drawn {sum(v for _, v in drawn):>14,.1f}")
    print(f"rest  {rest['value']:>14,.1f}")
    print(f"loop  {loop['value']:>14,.1f}")
    print(f"total {total:>14,.1f}   every route into A1 {inbound:>14,.1f}")
    assert round(total, 6) == round(inbound, 6), "drawn + rest + loop must close"

    # A1 -> A1 is a route out of A1 as well, so it lands in the rest node on the
    # drop side too. That rest node stands for A1 itself, and the table drawn on
    # it is A1 measured as a pickup - which is why it repeats the focus cluster's
    # own number. The edge into it is the loop, and that is what closes the sum.
    outbound = graph.node_value("A1", "orders", "out")
    drops = graph.flow_subgraph("A1", upstream=0, downstream=2, metric="orders",
                                rest=True)
    drop_rest = [data for _, data in drops.nodes(data=True)
                 if data.get("is_rest")][0]
    drawn_out = sum(data["value"] for _, data in drops.nodes(data=True)
                    if data.get("side") == "drop" and not data.get("is_rest"))
    out_total = drawn_out + drop_rest["value"] + loop["value"]
    print()
    print("and the same on the way out:")
    print(f"drawn {drawn_out:>14,.1f}")
    print(f"rest  {drop_rest['value']:>14,.1f}   {drop_rest['partners']}")
    print(f"loop  {loop['value']:>14,.1f}")
    print(f"total {out_total:>14,.1f}   every route out of A1 {outbound:>14,.1f}")
    assert round(out_total, 6) == round(outbound, 6)
    assert "A1" not in drop_rest["partners"]
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

    # the default: its own node, beside the cluster, outside the top-k
    save(explorer.plot(
        "A1", upstream=3, downstream=3, rest=True,
        node_metrics=["orders"], edge_metrics=["orders", "avg_distance"],
        title="A1 -> A1 on its own node, never inside rest",
    ), here / "self_loop_node.png")

    # the same loop drawn on the cluster instead
    save(explorer.plot(
        "A1", upstream=3, downstream=3, rest=True, self_loops="ring",
        node_metrics=["orders"], edge_metrics=["orders", "avg_distance"],
        title='self_loops="ring": drawn on A1 rather than beside it',
    ), here / "self_loop_ring.png")

    print()
    print("the loop is there whatever the top x is, and takes no place in it:")
    for keep in (1, 3, 5):
        sub = explorer.graph.flow_subgraph(
            "A1", upstream=keep, downstream=0, metric="orders", rest=True)
        outside = [data["cluster"] for _, data in sub.nodes(data=True)
                   if data.get("side") == "source" and not data.get("is_rest")]
        loops = [data["cluster"] for _, data in sub.nodes(data=True)
                 if data.get("is_self")]
        print(f"   upstream={keep} -> {len(outside)} outside clusters {outside}"
              f", loop node {loops}")

    # each side judges the loop on its own threshold, so a loop can be drawn
    # upstream, downstream, on both, or on neither
    print()
    print('with self_loops="ring" each side decides for itself, against its own')
    print("top x, so a loop can be drawn on one side, both, or neither:")
    modest = RouteExplorer(
        make_modest_loop_frame(), grain_cols=("week_period",),
        sum_metrics=["orders", "requests"], mean_metrics=["avg_distance"],
        weight_col="requests", metric="orders",
    )
    for up, down in ((3, 3), (3, 0), (0, 3)):
        sub = modest.graph.flow_subgraph(
            "A1", upstream=up, downstream=down, metric="orders",
            self_loops="ring")
        focus = [node for node, data in sub.nodes(data=True)
                 if data.get("is_focus")][0]
        rings = sorted(sub.nodes[focus].get("loops") or {})
        print(f"   upstream={up} downstream={down} -> rings on {rings or 'neither side'}")

    save(modest.plot(
        "A1", upstream=3, downstream=3, self_loops="ring",
        node_metrics=["orders"], edge_metrics=["orders"],
        title='self_loops="ring": the loop clears the downstream threshold only',
    ), here / "self_loop_one_side.png")


if __name__ == "__main__":
    main()
