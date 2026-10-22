"""把选出的股票整理成交易参考表：板块、参考买入区间、明日参考止盈、严格止损、核心逻辑。

价位按个股自身波动（14 日 ATR，占股价的百分比）计算，波动大的股票区间更宽：
  参考买入区间  收盘价 × (1 − a) ~ 收盘价 × (1 + a/3)，a = 0.3×ATR%，限制在 0.5%~1.5%
  明日参考止盈  收盘价 × (1 + 0.5×ATR%) ~ 收盘价 × (1 + 0.8×ATR%)，至少 +2%，不超过涨停价
  严格止损      收盘价 × (1 − 0.6×ATR%)，限制在 −2.5%~−5%
这些价位只是按波动给出的参考，不是预测，也没有经过回测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr_pct(df: pd.DataFrame | None, n: int = 14) -> float:
    if df is None or len(df) < n + 1:
        return 0.03
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
                   axis=1).max(axis=1)
    v = float(tr.iloc[-n:].mean() / df["close"].iloc[-1])
    return v if np.isfinite(v) and v > 0 else 0.03


def _limit(code: str) -> float:
    return 0.2 if code.startswith(("300", "301", "688", "689")) else 0.1


def levels(code: str, close: float, df: pd.DataFrame | None) -> dict:
    a = atr_pct(df)
    lim = _limit(code)
    up_cap = round(close * (1 + lim), 2)
    buy_a = float(np.clip(0.3 * a, 0.005, 0.015))
    tp1 = min(close * (1 + max(0.5 * a, 0.02)), up_cap)
    tp2 = min(close * (1 + max(0.8 * a, 0.03)), up_cap)
    sl = close * (1 - float(np.clip(0.6 * a, 0.025, 0.05)))
    return {"参考买入区间": f"{close * (1 - buy_a):.2f}-{close * (1 + buy_a / 3):.2f}",
            "明日参考止盈": f"{tp1:.2f}~{tp2:.2f}",
            "严格止损": f"{sl:.2f}",
            "ATR%": round(a * 100, 1)}


def logic_text(r: dict, big_col: str) -> str:
    parts = []
    p = r.get(big_col)
    if p is not None and pd.notna(p):
        parts.append(f"明日大涨概率{p:.0f}%")
    if r.get("在主线板块"):
        parts.append("所属板块今日走强")
    src = str(r.get("入选依据") or "")
    if "回落" in src:
        parts.append("尾盘回落型")
    if "趋势" in src or r.get("趋势") == "多头":
        parts.append("均线多头")
    if "反转" in src:
        parts.append("超跌反转信号")
    mf = r.get("主力净占比%")
    if mf is not None and pd.notna(mf):
        parts.append(f"主力净流入{mf:+.1f}%" if mf > 0 else f"主力净流出{mf:.1f}%")
    bo = r.get("大单主动买占比")
    if bo is not None and pd.notna(bo) and bo >= 0.55:
        parts.append("大单主动买入为主")
    vr = r.get("量比")
    if vr is not None and pd.notna(vr):
        parts.append(f"量比{vr:.1f}")
    tv = r.get("换手%")
    if tv is not None and pd.notna(tv):
        parts.append(f"换手{tv:.1f}%")
    if r.get("上龙虎榜"):
        parts.append("今日上龙虎榜" + ("·机构游资共振" if r.get("机构游资共振") else ""))
    shape = r.get("尾盘形态")
    if shape and shape not in ("尾盘平稳", "nan"):
        parts.append(str(shape))
    pos = r.get("位置")
    if pos in ("低位", "高位"):
        parts.append(f"60日{pos}")
    return "，".join(parts) if parts else "-"


def build_table(short: pd.DataFrame, full: pd.DataFrame | None, hist: dict, big_col: str) -> pd.DataFrame:
    extra = full.set_index("代码") if full is not None and not full.empty else pd.DataFrame()
    rows = []
    for _, r in short.iterrows():
        code = r["代码"]
        d = dict(r)
        if not extra.empty and code in extra.index:
            for k, v in extra.loc[code].items():
                d.setdefault(k, v)
                if pd.isna(d.get(k)) if not isinstance(d.get(k), (list, dict)) else False:
                    d[k] = v
        lv = levels(code, float(r["现价"]), hist.get(code))
        rows.append({"排名": r.get("排名"), "代码": code, "名称": r.get("名称", ""),
                     "板块": d.get("所属行业") or "-", "现价": r["现价"],
                     "参考买入区间": lv["参考买入区间"], "明日参考止盈": lv["明日参考止盈"],
                     "严格止损": lv["严格止损"], big_col: r.get(big_col),
                     "核心逻辑": logic_text(d, big_col)})
    return pd.DataFrame(rows)
