"""Shared runtime semantics for the Suppression state."""

from __future__ import annotations

from copy import deepcopy


def apply(unit, card) -> bool:
    """Suppress *unit*, saving exactly the mutable state it overrides.

    Suppression is not Pin: it removes abilities and modifiers and resets the
    unit to its printed combat values.  The snapshot lets temporary effects
    restore the previous state without accidentally restoring a later change.
    """
    if unit.status.get("cannot_be_suppressed") or unit.status.get("cannot") in ("be_suppressed", True):
        return False
    if unit.status.get("suppressed"):
        return True
    snapshot = {"attack": unit.attack, "defense": unit.defense,
                "status": deepcopy(unit.status), "modifiers": deepcopy(unit.modifiers)}
    unit.attack = card.attack or 0
    unit.defense = card.defense or 0
    unit.modifiers.clear()
    unit.status.clear()
    unit.status.update({"suppressed": True, "suppression_snapshot": snapshot})
    return True


def restore(unit) -> bool:
    """Restore a unit only when it is still under this suppression layer."""
    snapshot = unit.status.get("suppression_snapshot")
    if not unit.status.get("suppressed") or not isinstance(snapshot, dict):
        return False
    unit.attack = snapshot["attack"]
    unit.defense = snapshot["defense"]
    unit.modifiers[:] = snapshot["modifiers"]
    unit.status.clear()
    unit.status.update(snapshot["status"])
    return True
