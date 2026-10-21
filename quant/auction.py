"""次日盘前验证（9:25 集合竞价结束后运行）：
  - 昨日各模型选股：竞价高开/低开幅度、竞价量能（竞价成交额 / 昨日全天成交额）
  - 昨日涨停股：竞价平均溢价（情绪延续还是退潮）
  - 昨日炸板（烂板/断板）股：是否“弱转强”（高开 >2%）
  - 昨日跌停股：是否继续“核按钮”（低开 <-5%）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import norm_code


def auction_table(spot: pd.DataFrame, codes: list[str], prev_amount: dict | None = None) -> pd.DataFrame:
    s = spot.set_index("code")
    rows = []
    for c in codes:
        c = norm_code(c)
        if c not in s.index:
            continue
        q = s.loc[c]
        pc, op = q.get("prev_close"), q.get("open")
        if pd.isna(pc) or not pc or pd.isna(op) or not op:
            continue
        gap = (op / pc - 1) * 100
        amt = q.get("amount")
        ratio = None
        if prev_amount and prev_amount.get(c) and pd.notna(amt):
            ratio = round(amt / prev_amount[c] * 100, 2)
        rows.append({"代码": c, "名称": q.get("name", ""), "昨收": pc, "竞价开盘": op,
                     "竞价涨幅%": round(gap, 2), "竞价额/昨日成交额%": ratio})
    return pd.DataFrame(rows)


def judge(gap: float, ratio: float | None) -> str:
    vol_up = ratio is not None and ratio >= 5      # 竞价额占昨日 5% 以上视为放量
    if gap >= 3 and vol_up:
        return "高开放量（强，注意高开低走）"
    if gap >= 1:
        return "高开"
    if gap <= -3:
        return "大幅低开（弱，优先放弃）"
    if gap <= -1:
        return "低开（偏弱）"
    return "平开"


def run(spot: pd.DataFrame, archive_root: Path, prev_day: str, picks_dir: Path,
        hist: dict | None = None, universe: dict | None = None) -> dict:
    out = {}
    if universe is not None:
        from .screener import code_allowed
        spot = spot[spot["code"].map(lambda c: code_allowed(c, universe))]
    prev_amount = {}
    if hist:
        for c, df in hist.items():
            if len(df):
                prev_amount[c] = float(df["amount"].iloc[-1])

    picks = {}
    for f in sorted(picks_dir.glob(f"picks_*_{prev_day}.csv")):
        model = f.stem.split("_")[1]
        picks[model] = pd.read_csv(f, dtype={"代码": str})["代码"].tolist()
    for m, codes in picks.items():
        t = auction_table(spot, codes, prev_amount)
        if not t.empty:
            t["判断"] = [judge(g, r) for g, r in zip(t["竞价涨幅%"], t["竞价额/昨日成交额%"])]
        out[f"选股:{m}"] = t

    def pool(name):
        p = archive_root / prev_day / f"{name}.csv"
        return pd.read_csv(p, dtype={"代码": str}) if p.exists() else None

    zt, zb, dt = pool("zt_pool"), pool("zb_pool"), pool("dt_pool")
    if zt is not None:
        t = auction_table(spot, zt["代码"].tolist())
        out["昨日涨停股"] = t
        out["昨日涨停竞价均涨%"] = round(float(t["竞价涨幅%"].mean()), 2) if len(t) else None
    if zb is not None:
        t = auction_table(spot, zb["代码"].tolist())
        if len(t):
            t["弱转强"] = t["竞价涨幅%"] >= 2
        out["昨日炸板股"] = t
    if dt is not None:
        t = auction_table(spot, dt["代码"].tolist())
        if len(t):
            t["继续核按钮"] = t["竞价涨幅%"] <= -5
        out["昨日跌停股"] = t
    return out
