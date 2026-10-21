from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULTS: dict[str, Any] = {
    "data": {"cache_dir": "data/cache", "history_days": 400, "adjust": "qfq",
             "max_workers": 4, "retries": 3},
    "output_dir": "output",
    "universe": {"exclude_st": True, "exclude_bj": True, "exclude_kcb": False,
                 "exclude_cyb": False, "min_amount": 1e8, "min_total_mv": 3e9,
                 "min_bars": 60, "all_top_n": 300},
    "strategies": {},
    "watchlist": [],
    "monitor": {"interval": 60, "strategies": ["ma_cross", "macd", "mean_reversion"],
                "notify": {"console": True, "macos": True, "webhook": {"type": "", "url": ""}}},
}


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    user = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user)
    # 相对路径统一以项目根目录为基准
    for key in ("output_dir",):
        cfg[key] = str((ROOT / cfg[key]).resolve()) if not Path(cfg[key]).is_absolute() else cfg[key]
    cd = cfg["data"]["cache_dir"]
    cfg["data"]["cache_dir"] = str((ROOT / cd).resolve()) if not Path(cd).is_absolute() else cd
    # 自选股代码统一为 6 位字符串
    wl = []
    for item in cfg.get("watchlist") or []:
        if isinstance(item, (str, int)):
            item = {"code": item}
        item = dict(item)
        item["code"] = str(item["code"]).zfill(6)
        wl.append(item)
    cfg["watchlist"] = wl
    return cfg
