"""Locations for mutable training artifacts.

Runtime data is deliberately kept off the repository (and, by default, off
the system drive) so an unattended training run does not churn OneDrive/C:.
Set KARDS_RUNS_DIR to override the location for another machine or disk.
"""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runs_dir() -> Path:
    configured = os.environ.get("KARDS_RUNS_DIR")
    if configured:
        return Path(configured).expanduser()
    d_drive = Path("D:/")
    return d_drive / "KardsAI" / "runs" if d_drive.exists() else ROOT / "runs"


def runtime_path(value: str | Path | None, default_name: str) -> Path:
    """Resolve legacy ``runs/...`` defaults into the selected runtime folder."""
    path = Path(value) if value else Path(default_name)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "runs":
        path = Path(*path.parts[1:])
    return runs_dir() / path
