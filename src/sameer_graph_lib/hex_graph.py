"""Backwards-compatible HexGraph facade."""

from __future__ import annotations

from .affinity_graph import AffinityGraph


class HexGraph(AffinityGraph):
    """Compatibility wrapper around :class:`AffinityGraph`.

    The original single-class API is still available through this name, with
    additional route ingestion, corridor extraction, branch decomposition,
    editing helpers, and JSON persistence inherited from ``AffinityGraph``.
    """

