"""Small Windows control panel for the long-running KARDS AI trainer."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox
import webbrowser


WORKSPACE = Path(os.environ.get("KARDS_WORKSPACE", "D:/KardsAI/workspace"))
RUNS = Path(os.environ.get("KARDS_RUNS_DIR", "D:/KardsAI/runs"))
PYTHON = Path(os.environ.get("KARDS_VENV", "D:/KardsAI/training-venv-py312")) / "Scripts" / "python.exe"
LOG = RUNS / "formal_training.log"


class TrainingControl(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("KARDS AI 训练控制")
        self.resizable(False, False)
        self.configure(padx=22, pady=18, bg="#18212d")
        self.status = tk.StringVar(value="正在读取训练状态…")
        self.detail = tk.StringVar(value=str(RUNS))
        tk.Label(self, text="KARDS AI 训练控制", font=("Microsoft YaHei UI", 16, "bold"), bg="#18212d", fg="#f4f7fb").pack(anchor="w")
        tk.Label(self, textvariable=self.status, font=("Microsoft YaHei UI", 11), bg="#18212d", fg="#63d7a4").pack(anchor="w", pady=(12, 2))
        tk.Label(self, textvariable=self.detail, font=("Microsoft YaHei UI", 9), bg="#18212d", fg="#aab8ca").pack(anchor="w", pady=(0, 14))
        buttons = tk.Frame(self, bg="#18212d"); buttons.pack(fill="x")
        self._button(buttons, "开始训练", self.start, "#2878d4").grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        self._button(buttons, "暂停训练", self.pause, "#b97716").grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self._button(buttons, "继续训练", self.resume, "#238d5b").grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        self._button(buttons, "结束训练", self.stop, "#b44141").grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        for col in range(2): buttons.columnconfigure(col, weight=1)
        self._button(self, "打开 Web 监控 Dashboard", self.open_dashboard, "#53667e").pack(fill="x", pady=(12, 0))
        self.after(200, self.refresh)

    def _button(self, parent: tk.Misc, label: str, command, color: str) -> tk.Button:
        return tk.Button(parent, text=label, command=command, bg=color, fg="white", activebackground=color,
                         activeforeground="white", relief="flat", font=("Microsoft YaHei UI", 10, "bold"), padx=12, pady=9)

    def start(self) -> None:
        if not PYTHON.exists() or not (WORKSPACE / "auto_train.py").exists():
            messagebox.showerror("无法启动", f"未找到 D 盘训练环境：\n{WORKSPACE}")
            return
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "STOP").unlink(missing_ok=True); (RUNS / "PAUSE").unlink(missing_ok=True)
        command = [str(PYTHON), "auto_train.py", "--cycles", "0", "--episodes", "32", "--mcts-simulations", "16", "--workers", "4", "--updates", "100", "--batch-size", "64", "--evaluation-games", "20"]
        with LOG.open("a", encoding="utf-8") as output:
            subprocess.Popen(command, cwd=WORKSPACE, stdout=output, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        self.status.set("已启动训练；正在初始化 self-play…")
        self.open_dashboard()

    def pause(self) -> None:
        RUNS.mkdir(parents=True, exist_ok=True); (RUNS / "PAUSE").touch()
        self.status.set("已请求暂停；当前安全阶段结束后暂停。")

    def resume(self) -> None:
        (RUNS / "PAUSE").unlink(missing_ok=True); (RUNS / "STOP").unlink(missing_ok=True)
        self.status.set("已请求继续训练。")

    def stop(self) -> None:
        if messagebox.askyesno("结束训练", "将安全结束训练：当前 self-play 或更新阶段完成后保存结果并退出。确定吗？"):
            RUNS.mkdir(parents=True, exist_ok=True); (RUNS / "PAUSE").unlink(missing_ok=True); (RUNS / "STOP").touch()
            self.status.set("已请求安全结束；不会强制终止进程。")

    def open_dashboard(self) -> None:
        webbrowser.open("http://127.0.0.1:8765")

    def refresh(self) -> None:
        if (RUNS / "PAUSE").exists():
            text, color = "状态：已暂停（将在安全边界保持暂停）", "#ffd166"
        elif (RUNS / "STOP").exists():
            text, color = "状态：正在安全结束训练…", "#ff9f9f"
        elif LOG.exists() and time.time() - LOG.stat().st_mtime < 300:
            text, color = "状态：训练正在运行", "#63d7a4"
        else:
            text, color = "状态：未运行或等待下一次启动", "#aab8ca"
        self.status.set(text); self.detail.set(f"运行目录：{RUNS}")
        for child in self.winfo_children():
            if isinstance(child, tk.Label) and child.cget("textvariable") == str(self.status): child.configure(fg=color)
        self.after(2000, self.refresh)


if __name__ == "__main__":
    TrainingControl().mainloop()
