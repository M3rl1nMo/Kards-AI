"""Formal unattended-training launcher with environment and lock checks."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from ai.config import load_config
from ai.runtime import runs_dir

ROOT = Path(__file__).parent

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default=ROOT / "configs/formal_training.yaml"); p.add_argument("--runs"); p.add_argument("--foreground", action="store_true")
    a = p.parse_args(); config = Path(a.config).resolve(); cfg = load_config(config)
    if not torch.cuda.is_available(): raise RuntimeError("Formal training requires CUDA")
    if cfg["self_play"]["workers"] < 1 or cfg["self_play"]["simulations"] < 1: raise ValueError("Invalid self-play configuration")
    runs = Path(a.runs) if a.runs else runs_dir(); runs.mkdir(parents=True, exist_ok=True)
    lock = runs / "training.lock"; pid = runs / "training.pid"
    if pid.exists():
        try: os.kill(int(pid.read_text(encoding="utf-8")), 0)
        except (OSError, ValueError): pid.unlink(missing_ok=True); lock.unlink(missing_ok=True)
    if lock.exists() or pid.exists(): raise RuntimeError(f"Training lock already exists: {lock}")
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable, "cuda": torch.cuda.get_device_name(0), "config": str(config), "config_snapshot": cfg}
    (runs / "startup.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    env = os.environ.copy(); env["KARDS_RUNS_DIR"] = str(runs)
    cmd = [sys.executable, "auto_train.py", "--config", str(config)]
    if a.foreground:
        lock.touch(); pid.write_text(str(os.getpid()), encoding="utf-8")
        try: subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        finally: lock.unlink(missing_ok=True); pid.unlink(missing_ok=True)
    else:
        (runs / "logs").mkdir(exist_ok=True)
        with (runs / "logs" / "training.log").open("a", encoding="utf-8") as out:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        lock.touch(); pid.write_text(str(proc.pid), encoding="utf-8"); print(json.dumps({**info, "pid": proc.pid, "runs": str(runs)}))
if __name__ == "__main__": main()
