"""Read-only local web dashboard for a long-running KARDS AI training run."""
from __future__ import annotations

from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlparse

from ai.runtime import runs_dir

ROOT = Path(__file__).parent
RUNS = runs_dir()
_SYSTEM_CACHE: tuple[float, dict] = (0.0, {})


def records(name: str, limit: int = 2_000) -> list[dict]:
    path = RUNS / name
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 2_000_000))
        lines = handle.read().decode("utf-8", errors="ignore").splitlines()[-limit:]
    result: list[dict] = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def tail_log(lines: int = 120) -> list[str]:
    path = RUNS / "formal_training.log"
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 256_000))
        return handle.read().decode("utf-8", errors="replace").splitlines()[-lines:]


def system_usage() -> dict:
    global _SYSTEM_CACHE
    now = time.monotonic()
    if now - _SYSTEM_CACHE[0] < 2:
        return _SYSTEM_CACHE[1]
    data = {"cpu_percent": None, "ram_percent": None, "ram_used_gb": None, "gpu_percent": None, "gpu_memory_mb": None}
    try:
        import psutil  # optional; dashboard remains dependency-free without it
        memory = psutil.virtual_memory(); data.update(cpu_percent=psutil.cpu_percent(), ram_percent=memory.percent, ram_used_gb=round(memory.used / 2**30, 2))
    except ImportError:
        pass
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=1).splitlines()[0]
        gpu, memory = (int(value.strip()) for value in output.split(",")); data.update(gpu_percent=gpu, gpu_memory_mb=memory)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    _SYSTEM_CACHE = now, data
    return data


def status() -> str:
    log = tail_log(30)
    if (RUNS / "PAUSE").exists():
        return "Paused"
    if any("Traceback" in line or "RuntimeError:" in line for line in log):
        return "Error"
    if (RUNS / "STOP").exists():
        return "Stopping"
    metrics_path = RUNS / "training_metrics.jsonl"
    if metrics_path.exists() and time.time() - metrics_path.stat().st_mtime < 180:
        return "Running"
    # A long self-play episode can be quiet for several minutes.  Check the
    # command line only after lightweight file checks, so a quiet live worker
    # is not misreported as stopped.
    try:
        commands = subprocess.check_output(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"], text=True, timeout=1)
        if any("auto_train.py" in line or "selfplay.py" in line for line in commands.splitlines()):
            return "Running"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    if log and (time.time() - (RUNS / "formal_training.log").stat().st_mtime < 180):
        return "Running"
    return "Stopped"


def dashboard_data() -> dict:
    metrics = records("training_metrics.jsonl")
    selfplay = [item for item in metrics if item.get("event") in {"selfplay_progress", "selfplay_complete"}]
    train = [item for item in metrics if item.get("event") in {"train_progress", "train_complete"}]
    evaluations = records("cycle_history.jsonl")
    latest = (train + selfplay)[-1] if train or selfplay else {}
    total_games = max((int(item.get("total_episodes", 0)) for item in selfplay), default=0)
    replay = RUNS / "replay.pkl"; checkpoint = RUNS / "kardsnet.pt"
    return {
        "status": status(), "timestamp_utc": datetime.now(timezone.utc).isoformat(), "latest": latest,
        "training": {"total_games": total_games},
        "selfplay": selfplay[-200:], "train": train[-500:], "evaluations": evaluations[-100:],
        "games": records("games.jsonl", 50), "log": tail_log(), "system": system_usage(),
        "artifacts": {"replay_bytes": replay.stat().st_size if replay.exists() else 0,
                      "checkpoint_time": checkpoint.stat().st_mtime if checkpoint.exists() else None,
                      "replay_exists": replay.exists(), "checkpoint_exists": checkpoint.exists()},
        "controls": {"paused": (RUNS / "PAUSE").exists(), "workers": latest.get("workers")},
    }


def control(name: str) -> dict:
    if name == "pause":
        (RUNS / "PAUSE").touch()
    elif name == "resume":
        (RUNS / "PAUSE").unlink(missing_ok=True)
    elif name == "checkpoint":
        (RUNS / "SAVE_CHECKPOINT").touch()
    else:
        raise ValueError("Unknown control")
    return {"ok": True, "status": status()}


class Handler(SimpleHTTPRequestHandler):
    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if request_path == "/api/status":
            self._json(dashboard_data()); return
        # Query parameters are used for cache busting.  Route by the parsed
        # path so `/?v=...` cannot fall through to a repository directory
        # listing.
        self.path = "/monitor/index.html" if request_path == "/" else request_path
        super().do_GET()

    def do_POST(self) -> None:
        try:
            self._json(control(urlparse(self.path).path.rsplit("/", 1)[-1]))
        except ValueError as error:
            self._json({"ok": False, "error": str(error)}, 404)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("KARDS training dashboard: http://127.0.0.1:8765", flush=True)
    server.serve_forever()
