"""市场情绪温度：涨跌停、封板率、炸板率、连板高度与晋级率、昨日涨停溢价、指数尾盘、期货升贴水、板块主线。

温度分 0~100（经验加权，尚未经过历史回测，数据会每天归档，攒够后可以检验）：
  冰点 <25 / 低迷 25~40 / 中性 40~60 / 活跃 60~75 / 过热 >75
冰点或低迷时，尾盘选股会提示“建议观望”。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import norm_code


def _limit(code: str, name: str = "") -> float:
    if "ST" in str(name):
        return 4.9
    if code.startswith(("300", "301", "688", "689")):
        return 19.8
    if code.startswith(("8", "4", "92")):
        return 29.7
    return 9.8


def stats_from_pools(zt, zb, dt, zt_prev, zt_yesterday_count: int | None) -> dict:
    s = {}
    if zt is not None:
        s["涨停家数"] = len(zt)
        if "连板数" in zt:
            lb = pd.to_numeric(zt["连板数"], errors="coerce")
            s["连板高度"] = int(lb.max()) if lb.notna().any() else 1
            s["连板家数"] = int((lb >= 2).sum())
            if zt_yesterday_count:
                s["连板晋级率%"] = round(s["连板家数"] / zt_yesterday_count * 100, 1)
    if zb is not None:
        s["炸板家数"] = len(zb)
    if zt is not None and zb is not None and len(zt) + len(zb):
        s["封板率%"] = round(len(zt) / (len(zt) + len(zb)) * 100, 1)
    if dt is not None:
        s["跌停家数"] = len(dt)
    if zt_prev is not None and "涨跌幅" in zt_prev:
        s["昨日涨停今日均涨%"] = round(float(pd.to_numeric(zt_prev["涨跌幅"], errors="coerce").mean()), 2)
    return s


def stats_from_spot(spot: pd.DataFrame) -> dict:
    """涨停池不可用时，用全市场快照估算涨跌停、炸板（拿不到连板与昨日涨停溢价）。"""
    if spot is None or spot.empty:
        return {}
    df = spot.dropna(subset=["price", "prev_close"]).copy()
    df = df[df["prev_close"] > 0]
    lim = np.array([_limit(c, n) for c, n in zip(df["code"], df.get("name", [""] * len(df)))])
    pct = (df["price"] / df["prev_close"] - 1) * 100
    hpct = (df["high"] / df["prev_close"] - 1) * 100
    zt = (pct >= lim) & (df["price"] >= df["high"] - 1e-6)
    touched = hpct >= lim
    s = {"涨停家数": int(zt.sum()), "炸板家数": int((touched & ~zt).sum()),
         "跌停家数": int((pct <= -lim).sum()),
         "上涨家数": int((pct > 0).sum()), "下跌家数": int((pct < 0).sum())}
    if touched.sum():
        s["封板率%"] = round(zt.sum() / touched.sum() * 100, 1)
    s["全市场平均涨幅%"] = round(float(pct.clip(-20, 20).mean()), 2)
    return s


def tail_index_move(minute: pd.DataFrame | None) -> float | None:
    """指数 14:30 以后的涨跌幅（%）。"""
    if minute is None or minute.empty:
        return None
    m = minute.set_index("time")["close"]
    before = m[m.index.time <= pd.Timestamp("14:30").time()]
    if before.empty:
        return None
    return round((m.iloc[-1] / before.iloc[-1] - 1) * 100, 2)


def temperature(s: dict, idx_tail: float | None = None, basis: pd.DataFrame | None = None
                ) -> tuple[float | None, str, list[str]]:
    """把各项指标合成温度分，返回 (分数, 档位, 说明)。缺失的项不计入。"""
    parts, notes = [], []

    def add(val, lo, hi, w, label):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return
        parts.append((float(np.clip((val - lo) / (hi - lo), 0, 1)), w))
        notes.append(label)

    zt, dt = s.get("涨停家数"), s.get("跌停家数")
    if zt is not None and dt is not None:
        add(zt / (zt + dt + 1), 0.3, 0.95, 2.0, f"涨停{zt}/跌停{dt}")
    add(s.get("封板率%"), 50, 85, 1.5, f"封板率{s.get('封板率%')}%")
    add(s.get("昨日涨停今日均涨%"), -3, 4, 2.0, f"昨日涨停今日均{s.get('昨日涨停今日均涨%')}%")
    add(s.get("连板高度"), 2, 7, 1.0, f"连板高度{s.get('连板高度')}")
    add(s.get("连板晋级率%"), 10, 45, 1.0, f"晋级率{s.get('连板晋级率%')}%")
    if s.get("上涨家数") is not None and s.get("下跌家数") is not None:
        up, dn = s["上涨家数"], s["下跌家数"]
        add(up / (up + dn + 1), 0.2, 0.8, 1.5, f"涨{up}/跌{dn}")
    add(idx_tail, -1, 1, 0.8, f"中证500尾盘{idx_tail}%")
    if basis is not None and not basis.empty:
        b = float(basis["升贴水%"].mean())
        add(b, -1.5, 0.3, 0.8, f"期货升贴水均{b:.2f}%")
    if not parts:
        return None, "未知", ["情绪数据都不可用"]
    score = round(sum(v * w for v, w in parts) / sum(w for _, w in parts) * 100, 1)
    level = ("冰点" if score < 25 else "低迷" if score < 40 else "中性" if score < 60
             else "活跃" if score < 75 else "过热")
    return score, level, notes


def main_sectors(sectors: pd.DataFrame | None, n: int = 5) -> pd.DataFrame | None:
    if sectors is None or "涨跌幅" not in sectors:
        return None
    cols = [c for c in ("板块", "涨跌幅", "上涨家数", "下跌家数", "领涨股票") if c in sectors]
    return sectors.sort_values("涨跌幅", ascending=False)[cols].head(n).reset_index(drop=True)


def stock_industry_map(zt: pd.DataFrame | None) -> dict:
    if zt is None or "所属行业" not in zt:
        return {}
    return dict(zip(zt["代码"].map(norm_code), zt["所属行业"]))
