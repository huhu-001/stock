"""提醒通道：终端 / macOS 系统通知 / Webhook（企业微信、钉钉、飞书、Server酱）。"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from .timeutil import now_cn

log = logging.getLogger(__name__)

RED, GREEN, YELLOW, RESET = "\033[31m", "\033[32m", "\033[33m", "\033[0m"


class Notifier:
    def __init__(self, ncfg: dict, log_path: str | Path | None = None):
        self.cfg = ncfg or {}
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.color = sys.stdout.isatty()

    def send(self, title: str, body: str, level: str = "info"):
        ts = now_cn().strftime("%H:%M:%S")
        if self.cfg.get("console", True):
            c = {"buy": RED, "sell": GREEN, "warn": YELLOW}.get(level, "") if self.color else ""
            print(f"{c}[{ts}] 🔔 {title} | {body}{RESET if c else ''}", flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{now_cn():%Y-%m-%d %H:%M:%S}\t{title}\t{body}\n")
        if self.cfg.get("macos", True):
            self._macos(title, body)
        wh = self.cfg.get("webhook") or {}
        if wh.get("type") and wh.get("url"):
            try:
                self._webhook(wh["type"], wh["url"], title, body)
            except Exception as e:  # noqa: BLE001
                log.warning("Webhook 推送失败: %s", e)

    # ------------------------------------------------------------------
    @staticmethod
    def _macos(title: str, body: str):
        if sys.platform != "darwin" or not shutil.which("osascript"):
            return
        script = (f"display notification {json.dumps(body, ensure_ascii=False)} "
                  f"with title {json.dumps(title, ensure_ascii=False)} sound name \"Glass\"")
        try:
            subprocess.run(["osascript", "-e", script], check=False, timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _webhook(kind: str, url: str, title: str, body: str):
        text = f"{title}\n{body}"
        kind = kind.lower()
        if kind == "serverchan":
            data = urllib.parse.urlencode({"title": title, "desp": body}).encode()
            req = urllib.request.Request(url, data=data)
        else:
            if kind in ("wecom", "dingtalk"):
                payload = {"msgtype": "text", "text": {"content": text}}
            elif kind == "feishu":
                payload = {"msg_type": "text", "content": {"text": text}}
            else:
                raise ValueError(f"不支持的 webhook 类型: {kind}")
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
