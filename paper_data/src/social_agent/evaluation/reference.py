"""Look up the reference answer of a pair in the community's compatibility matrices (evaluator/matrices.json)."""
from __future__ import annotations

from typing import Any


def _cell(m: dict[str, Any], rid: str, pid: str, j: str, kind: str) -> dict[str, Any] | None:
    if kind == "primary":
        return m["main"][pid].get(j)
    sw = m["switches"].get(rid)
    return sw["row"].get(j) if sw else None


def _reverse_cell(m: dict[str, Any], rid: str, pid: str, j: str, kind: str) -> dict[str, Any] | None:
    """j's PRIMARY request judged against pid's current state (the b->a direction of the pair)."""
    if kind == "primary":
        return m["main"][j].get(pid)
    sw = m["switches"].get(rid)
    return sw["column"].get(j) if sw else None


def _joint(a: str, b: str) -> str:
    if "reject" in (a, b):
        return "reject"
    if a == "recommend" and b == "recommend":
        return "recommend"
    return "insufficient_info"
