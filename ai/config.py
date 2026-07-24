"""YAML configuration loader used by all AI command-line entry points."""
from pathlib import Path
import yaml

ROOT = Path(__file__).parents[1]

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
