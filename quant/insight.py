"""个股综合评估：在模型候选股的基础上，补充尾盘分时、大单、资金流、活跃度、位置、龙虎榜、事件风险，
得到“综合精选”。

打分是经验规则（初始权重，尚未经过历史检验）。所有原始数据每天归档到 data/archive，
攒够几周后可以用 `python3 main.py tail --backtest` 之外的复盘结果来校准权重。
注意：在本项目已检验过的日线数据里，“尾盘强势、收在高位”的股票次日反而偏弱，
所以这里把尾盘急拉视为风险项，而“站稳均价线”只给小幅加分。
"""
from __future__ import annotations

import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .data import norm_code
from .marketdata import MarketData

log = logging.getLogger(__name__)

RISK_WORDS = re.compile(r"立案|调查|问询|监管|警示|处罚|退市|终止上市|诉讼|仲裁|冻结|违规|风险提示|减持|质押.*平仓")
BAD_FORECAST = ("预减", "首亏", "续亏", "增亏", "略减")


# ------------------------------------------------------------------ 单项特征
def tail_features(mb: pd.DataFrame | None) -> dict:
    """14:30 之后的分时特征。"""
    if mb is None or len(mb) < 30:
        return {}
    t = mb["time"].dt.time
    cut = pd.Timestamp("14:30").time()
    pre, post = mb[t <= cut], mb[t > cut]
    if pre.empty or post.empty:
        return {"尾盘数据": "未到14:30"}
    p1430, last = pre["close"].iloc[-1], post["close"].iloc[-1]
    tail_ret = (last / p1430 - 1) * 100
    vol_share = post["volume"].sum() / mb["volume"].sum() * 100 if mb["volume"].sum() else np.nan
    above = float((post["close"] >= post["vwap"]).mean())
    end_vs_vwap = (last / post["vwap"].iloc[-1] - 1) * 100 if post["vwap"].iloc[-1] else np.nan
    # 量价同步：尾盘每分钟涨跌与成交量的相关性
    sync = post["close"].diff().corr(post["volume"]) if len(post) > 5 else np.nan
    if tail_ret >= 2:
        shape = "尾盘急拉"
    elif tail_ret >= 0.8:
        shape = "尾盘拉升"
    elif tail_ret <= -2:
        shape = "尾盘跳水"
    elif tail_ret <= -0.8:
        shape = "尾盘走弱"
    else:
        shape = "尾盘平稳"
    if vol_share >= 20 and shape in ("尾盘跳水", "尾盘走弱"):
        shape = "放量" + shape
    return {"尾盘涨跌%": round(tail_ret, 2), "尾盘成交占比%": round(vol_share, 1),
            "尾盘站上均价线比例": round(above, 2), "收盘相对均价%": round(end_vs_vwap, 2),
            "尾盘量价相关": round(sync, 2) if pd.notna(sync) else None, "尾盘形态": shape}


def activity_features(df: pd.DataFrame | None) -> dict:
    if df is None or len(df) < 60:
        return {}
    v, a, c = df["volume"], df["amount"], df["close"]
    hi, lo = df["high"].iloc[-60:].max(), df["low"].iloc[-60:].min()
    pos = (c.iloc[-1] - lo) / (hi - lo) * 100 if hi > lo else 50
    return {"量比": round(v.iloc[-1] / v.iloc[-6:-1].mean(), 2) if v.iloc[-6:-1].mean() else None,
            "换手%": round(float(df["turnover"].iloc[-1]), 2) if "turnover" in df else None,
            "成交额/5日均": round(a.iloc[-1] / a.iloc[-6:-1].mean(), 2) if a.iloc[-6:-1].mean() else None,
            "60日位置%": round(pos, 0),
            "位置": "低位" if pos < 30 else "高位" if pos > 80 else "中位"}


def lhb_features(seats: pd.DataFrame | None) -> dict:
    if seats is None or seats.empty:
        return {}
    s = seats.drop_duplicates(subset=["营业部"]).copy()
    s["净额"] = s["买入"].fillna(0) - s["卖出"].fillna(0)
    inst = s["营业部"].astype(str).str.contains("机构专用")
    north = s["营业部"].astype(str).str.contains("股通")
    hot = ~inst & ~north
    buy_side = seats[seats["方向"] == "买入"]
    top_share = (buy_side["买入"].max() / buy_side["买入"].sum()) if buy_side["买入"].sum() else np.nan
    inst_net, hot_net = s.loc[inst, "净额"].sum(), s.loc[hot, "净额"].sum()
    return {"机构净额(万)": round(inst_net / 1e4, 0), "游资净额(万)": round(hot_net / 1e4, 0),
            "北向净额(万)": round(s.loc[north, "净额"].sum() / 1e4, 0),
            "买一占买五比例": round(top_share, 2) if pd.notna(top_share) else None,
            "一家独大": bool(top_share > 0.5) if pd.notna(top_share) else False,
            "机构游资共振": bool(inst_net > 0 and hot_net > 0)}


def event_flags(code: str, notices, forecasts, unlocks, reductions) -> list[str]:
    flags = []
    if notices is not None:
        n = notices[notices["代码"] == code]
        for _, r in n.iterrows():
            title = str(r.get("公告标题", ""))
            if RISK_WORDS.search(title) or str(r.get("公告类型", "")) == "风险提示":
                flags.append(f"公告:{title[:30]}")
    if forecasts is not None:
        f = forecasts[forecasts["代码"] == code]
        for _, r in f.iterrows():
            if any(k in str(r.get("预告类型", "")) for k in BAD_FORECAST):
                flags.append(f"业绩预告:{r.get('预告类型')}")
                break
    if unlocks is not None:
        u = unlocks[unlocks["代码"] == code]
        for _, r in u.iterrows():
            ratio = r.get("占解禁前流通市值比例")
            if pd.notna(ratio) and ratio >= 0.05:
                flags.append(f"{str(r.get('解禁时间'))[:10]}解禁占流通{ratio * 100:.0f}%")
    if reductions is not None and code in set(reductions["代码"]):
        flags.append("近30天有股东减持公告")
    return flags


# ------------------------------------------------------------------ 打分
def score_row(r: dict) -> tuple[float, list[str], list[str], bool]:
    """返回 (综合分, 加分项, 扣分项, 是否剔除)。"""
    plus, minus = [], []
    score = float(r.get("模型分", 50)) * 0.5

    def adj(cond, pts, text):
        nonlocal score
        if cond:
            score += pts
            (plus if pts > 0 else minus).append(f"{text}({pts:+d})")

    shape = r.get("尾盘形态", "")
    adj(shape == "尾盘急拉", -6, "尾盘急拉，次日易回落")
    adj(shape.startswith("放量尾盘跳水"), -15, "放量跳水")
    adj(shape == "尾盘跳水", -8, "尾盘跳水")
    above, vw = r.get("尾盘站上均价线比例"), r.get("收盘相对均价%")
    adj(above is not None and above >= 0.6 and vw is not None and vw >= 0, 4, "尾盘站稳均价线")
    bo = r.get("大单主动买占比")
    adj(bo is not None and bo >= 0.6, 6, "大单主动买入为主")
    adj(bo is not None and bo <= 0.4, -6, "大单主动卖出为主")
    mf = r.get("主力净占比%")
    adj(mf is not None and mf >= 5, 5, "主力净流入")
    adj(mf is not None and mf <= -8, -5, "主力大幅净流出")
    vr = r.get("量比")
    adj(vr is not None and vr >= 3, -4, "异常放量")
    pos = r.get("位置")
    adj(pos == "低位", 5, "处于60日低位")
    adj(pos == "高位", -6, "处于60日高位")
    adj(bool(r.get("机构游资共振")), 6, "机构游资共振买入")
    adj(bool(r.get("一家独大")), -5, "龙虎榜一家独大")
    inst = r.get("机构净额(万)")
    adj(inst is not None and inst < -1000, -5, "机构净卖出")
    adj(bool(r.get("在主线板块")), 4, "属于今日强势板块")
    flags = r.get("风险事件") or []
    for f in flags:
        score -= 15
        minus.append(f"{f}(-15)")
    exclude = bool(flags) or shape.startswith("放量尾盘跳水")
    return round(score, 1), plus, minus, exclude


# ------------------------------------------------------------------ 主流程
def evaluate(md: MarketData, candidates: pd.DataFrame, hist: dict, day: str,
             events: dict, lhb: pd.DataFrame | None, hot_sectors: set,
             industry: dict, max_workers: int = 3) -> pd.DataFrame:
    """candidates 需含：代码 名称 现价 涨幅% 模型分 来源模型。"""
    lhb_codes = set(lhb["代码"]) if lhb is not None else set()

    def one(row):
        code = norm_code(row["代码"])
        r = dict(row)
        r.update(activity_features(hist.get(code)))
        r.update(tail_features(md.minute_bars(code)))
        bo = md.big_orders(code)
        if bo:
            r["大单主动买占比"] = round(bo["大单主动买占比"], 2) if pd.notna(bo["大单主动买占比"]) else None
            r["大单净额(万)"] = round(bo["大单净额"] / 1e4, 0)
        ff = md.fund_flow(code)
        if ff:
            r["主力净占比%"] = ff["主力净占比%"]
            r["主力净流入(万)"] = round(ff["主力净流入"] / 1e4, 0) if pd.notna(ff["主力净流入"]) else None
        r["上龙虎榜"] = code in lhb_codes
        if r["上龙虎榜"]:
            r.update(lhb_features(md.lhb_seats(code, day)))
        r["所属行业"] = industry.get(code) or (md.industry(code) if hot_sectors else "")
        r["在主线板块"] = bool(r["所属行业"]) and r["所属行业"] in hot_sectors
        r["风险事件"] = event_flags(code, events.get("notices"), events.get("forecasts"),
                                  events.get("unlocks"), events.get("reductions"))
        s, plus, minus, excl = score_row(r)
        r.update({"综合分": s, "加分项": "；".join(plus), "扣分项": "；".join(minus),
                  "剔除": excl, "风险事件": "；".join(r["风险事件"])})
        return r

    rows = []
    print(f"  综合评估 {len(candidates)} 只候选股（分时/逐笔/资金流/龙虎榜/公告）…",
          file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(one, [row for _, row in candidates.iterrows()]):
            rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values(["剔除", "综合分"], ascending=[True, False]).reset_index(drop=True)


DISPLAY_COLS = ["代码", "名称", "现价", "涨幅%", "综合分", "来源模型", "尾盘形态", "尾盘涨跌%",
                "大单主动买占比", "主力净占比%", "量比", "位置", "上龙虎榜", "加分项", "扣分项"]


def display(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keep = df[~df["剔除"]].head(top_n)
    cols = [c for c in DISPLAY_COLS if c in keep.columns]
    out = keep[cols].copy()
    out.insert(0, "排名", range(1, len(out) + 1))
    return out
