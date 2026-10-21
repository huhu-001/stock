"""把扩展数据串起来：市场情绪 → 候选股综合评估 → 综合精选；并把当天所有数据归档。"""
from __future__ import annotations

import sys
from datetime import timedelta

import pandas as pd

from . import insight, sentiment
from .data import norm_code
from .marketdata import MarketData, last_trade_day
from .timeutil import now_cn


def collect_market(md: MarketData, spot: pd.DataFrame | None, day: str | None = None) -> dict:
    """收集情绪、指数、期货、板块数据，返回 dict，并归档。"""
    day = day or now_cn().strftime("%Y%m%d")
    today = pd.Timestamp(day).date()
    prev = last_trade_day(today).strftime("%Y%m%d")
    print("  收集市场情绪数据（涨停/炸板/跌停池、指数、期货、板块）…", file=sys.stderr, flush=True)
    zt, zb, dt, ztp = md.zt_pool(day), md.zb_pool(day), md.dt_pool(day), md.zt_prev_pool(day)
    zt_y = md.load_archive("zt_pool", prev)
    if zt_y is None and zt is not None:
        zt_y = md.zt_pool(prev)
    stats = sentiment.stats_from_pools(zt, zb, dt, ztp, len(zt_y) if zt_y is not None else None)
    if spot is not None:
        sp = sentiment.stats_from_spot(spot)
        for k, v in sp.items():
            stats.setdefault(k, v)          # 涨停池拿不到时用快照估算
    idx = md.index_spot()
    idx_tail = sentiment.tail_index_move(md.index_minute("sh000905"))
    basis = md.futures_basis(idx)
    sectors = md.sectors()
    score, level, notes = sentiment.temperature(stats, idx_tail, basis)
    for name, df in (("zt_pool", zt), ("zb_pool", zb), ("dt_pool", dt), ("zt_prev_pool", ztp),
                     ("index", idx), ("futures_basis", basis), ("sectors", sectors)):
        md.archive(name, df, day)
    summary = pd.DataFrame([{**stats, "中证500尾盘%": idx_tail, "温度分": score, "情绪": level}])
    md.archive("market_summary", summary, day)
    return {"stats": stats, "score": score, "level": level, "notes": notes, "index": idx,
            "idx_tail": idx_tail, "basis": basis, "sectors": sectors,
            "hot": sentiment.main_sectors(sectors), "zt": zt,
            "industry": sentiment.stock_industry_map(zt)}


def market_text(m: dict) -> str:
    if m["score"] is None:
        lines = ["市场情绪：无法计算（情绪数据都没取到，请先运行 python3 main.py check 检查数据源）"]
    else:
        lines = [f"市场情绪：{m['level']}（温度 {m['score']}/100）"]
    s = m["stats"]
    keys = ["涨停家数", "跌停家数", "炸板家数", "封板率%", "连板高度", "连板晋级率%",
            "昨日涨停今日均涨%", "上涨家数", "下跌家数", "全市场平均涨幅%"]
    lines.append("  " + "  ".join(f"{k} {s[k]}" for k in keys if k in s))
    if m["idx_tail"] is not None:
        lines.append(f"  中证500 尾盘(14:30后) {m['idx_tail']:+.2f}%")
    if m["basis"] is not None:
        lines.append("  股指期货升贴水：" + "  ".join(
            f"{r['品种']} {r['升贴水%']:+.2f}%" for _, r in m["basis"].iterrows()))
    if m["hot"] is not None:
        lines.append("  今日最强板块：" + "、".join(
            f"{r['板块']}({r['涨跌幅']:+.1f}%)" for _, r in m["hot"].iterrows()))
    if m["level"] in ("冰点", "低迷"):
        lines.append("  ⚠ 情绪偏冷：短线资金在撤退，追涨抄底都容易亏，建议观望或降低仓位。")
    elif m["level"] == "过热":
        lines.append("  ⚠ 情绪过热：容易出现高位分歧和大面，注意不追高。")
    return "\n".join(lines)


def build_candidates(results: dict, labels: dict) -> pd.DataFrame:
    """合并各模型的选股：来源模型、模型分（取平均）。"""
    rows = {}
    for m, df in results.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            c = norm_code(r["代码"])
            e = rows.setdefault(c, {"代码": c, "名称": r.get("名称", ""), "现价": r.get("现价"),
                                    "涨幅%": r.get("涨幅%"), "_scores": [], "_models": []})
            e["_scores"].append(float(r.get("得分", 50)))
            e["_models"].append(labels.get(m, m))
    out = []
    for e in rows.values():
        n = len(e["_models"])
        e["模型分"] = round(sum(e["_scores"]) / n + (n - 1) * 5, 1)   # 被多个模型同时选中略加分
        e["来源模型"] = "+".join(e["_models"])
        del e["_scores"], e["_models"]
        out.append(e)
    return pd.DataFrame(out)


def run_deep(md: MarketData, results: dict, labels: dict, hist: dict, spot, cfg: dict,
             market: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    day = now_cn().strftime("%Y%m%d")
    market = market or collect_market(md, spot, day)
    cands = build_candidates(results, labels)
    if cands.empty:
        return pd.DataFrame(), pd.DataFrame(), market
    print("  收集事件数据（公告/业绩预告/解禁/减持）、龙虎榜…", file=sys.stderr, flush=True)
    events = {"notices": md.notices(day), "forecasts": md.forecasts(),
              "unlocks": md.unlocks(), "reductions": md.reductions()}
    lhb = md.lhb(day)
    md.archive("lhb", lhb, day)
    hot = set(market["hot"]["板块"]) if market.get("hot") is not None else set()
    full = insight.evaluate(md, cands, hist, day, events, lhb, hot, market.get("industry", {}),
                            max_workers=int(cfg.get("deep", {}).get("max_workers", 3)))
    md.archive("insight", full, day)
    top = insight.display(full, int(cfg.get("tail", {}).get("top_n", 10)))
    return top, full, market
