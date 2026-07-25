"""Local, zero-dependency live dashboard for KARDS AI training."""
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse
from ai.runtime import runs_dir

ROOT = Path(__file__).parent
RUNS = runs_dir()


def records(name: str) -> list[dict]:
    path = RUNS / name
    if not path.exists(): return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: continue
    return result


def dashboard_data() -> dict:
    metrics = records("training_metrics.jsonl")
    progress = [x for x in metrics if x.get("event") in {"selfplay_progress", "train_progress"}]
    latest = progress[-1] if progress else {}
    train = [x for x in metrics if x.get("event") == "train_progress"][-40:]
    games = records("games.jsonl")[-12:]
    diagnosis = {"label": "Collecting self-play", "detail": "Overfit/underfit requires training and evaluation loss data."}
    if train:
        losses = [float(x["loss"]) for x in train if "loss" in x]
        if len(losses) < 5: diagnosis = {"label": "Insufficient training history", "detail": "Need more optimizer updates before judging fit."}
        elif losses[-1] < losses[0] * 0.9: diagnosis = {"label": "Learning", "detail": "Training loss is falling. Validation/evaluation data is still needed to detect overfit."}
        else: diagnosis = {"label": "Possible underfit or stalled learning", "detail": "Training loss has not meaningfully improved; compare against evaluation next."}
    return {"latest": latest, "metrics": progress[-80:], "train": train, "games": games, "diagnosis": diagnosis}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/status":
            body = json.dumps(dashboard_data(), ensure_ascii=False).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        self.path = "/monitor/index.html" if self.path == "/" else self.path
        super().do_GET()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("KARDS training dashboard: http://127.0.0.1:8765", flush=True)
    server.serve_forever()
