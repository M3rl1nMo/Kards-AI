"""Shared runtime semantics for Pin."""

from __future__ import annotations


def apply(unit) -> bool:
    """Pin a unit unless its card text grants pin immunity."""
    if unit.status.get("cannot_be_pinned") or unit.status.get("cannot") in ("be_pinned", "be_suppressed", True):
        return False
    unit.status["pinned"] = True
    return True
