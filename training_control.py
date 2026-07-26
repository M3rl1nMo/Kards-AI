"""Safe command-line control for the formal training loop."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from ai.runtime import runs_dir

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("command", choices=("start","pause","resume","save","stop","status")); p.add_argument("--config", default="configs/formal_training.yaml"); p.add_argument("--runs")
    a=p.parse_args(); runs=Path(a.runs) if a.runs else runs_dir(); runs.mkdir(parents=True, exist_ok=True)
    if a.command=="start":
        os.execv(os.sys.executable,[os.sys.executable,"start_training.py","--config",a.config,"--runs",str(runs)])
    if a.command=="pause": (runs/"PAUSE").touch(); (runs/"SAVE_CHECKPOINT").touch()
    elif a.command=="resume": (runs/"PAUSE").unlink(missing_ok=True); (runs/"STOP").unlink(missing_ok=True)
    elif a.command=="save": (runs/"SAVE_CHECKPOINT").touch()
    elif a.command=="stop": (runs/"SAVE_CHECKPOINT").touch(); (runs/"STOP").touch()
    print(json.dumps({"runs":str(runs),"pid":(runs/"training.pid").read_text(encoding="utf-8") if (runs/"training.pid").exists() else None,"paused":(runs/"PAUSE").exists(),"stopping":(runs/"STOP").exists(),"checkpoint":(runs/"latest.pt").exists()}))
if __name__=="__main__": main()
