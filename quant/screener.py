"""选股筛选：技术信号扫描 + 多因子打分。"""
from __future__ import annotations

import logging
import re
import sys

import pandas as pd

from . import strategies as S
from .data import BaseProvider, norm_code

log = logging.getLogger(__name__)

INDEX_ALIASES = {"hs300": "000300", "zz500": "000905", "zz1000": "000852", "sz50": "000016",
                 "a50": "000016", "kc50": "000688", "cyb": "399006"}


def _progress(i: int, n: int):
    if sys.stderr.isatty():
        bar = int(30 * i / n)
        sys.stderr.write(f"\r  下载行情 [{'#' * bar}{'.' * (30 - bar)}] {i}/{n}")
        if i == n:
            sys.stderr.write("\n")
        sys.stderr.flush()
    elif i % 50 == 0 or i == n:
        print(f"  已下载 {i}/{n}", file=sys.stderr, flush=True)


def code_allowed(code, u: dict) -> bool:
    """按代码过滤：只保留沪深 A 股；按配置剔除创业板 / 科创板 / 北交所；港股、B 股一律不要。"""
    c = str(code).strip()
    if not re.fullmatch(r"\d{6}", c):          # 港股是 5 位代码，带 hk 前缀等都排除
        return False
    if c.startswith(("900", "200")):          # B 股
        return False
    if u.get("exclude_bj", True) and re.match(r"^(8|4|92)", c):
        return False
    if u.get("exclude_kcb") and c.startswith(("688", "689")):
        return False
    if u.get("exclude_cyb") and c.startswith(("300", "301")):
        return False
    return True


def apply_universe_filters(spot: pd.DataFrame, u: dict) -> pd.DataFrame:
    df = spot.copy()
    code = df["code"].astype(str)
    if u.get("exclude_st", True):
        df = df[~df["name"].astype(str).str.contains("ST|退", regex=True)]
        code = df["code"].astype(str)
    if u.get("exclude_bj", True):
        df = df[~code.str.match(r"^(8|4|92)")]
        code = df["code"].astype(str)
    if u.get("exclude_kcb"):
        df = df[~code.str.startswith("688")]
        code = df["code"].astype(str)
    if u.get("exclude_cyb"):
        df = df[~code.str.match(r"^30[01]")]
    df = df[df["code"].map(lambda c: code_allowed(c, u))]
    if "price" in df:
        df = df[df["price"] > 0]  # 剔除停牌（无最新价）
    if u.get("min_amount") and "amount" in df:
        df = df[df["amount"] >= float(u["min_amount"])]
    if u.get("min_total_mv") and "total_mv" in df:
        df = df[df["total_mv"] >= float(u["min_total_mv"])]
    return df.reset_index(drop=True)


def get_spot_safe(provider: BaseProvider) -> pd.DataFrame | None:
    try:
        print("  获取全市场实时行情（约 10~60 秒）…", file=sys.stderr, flush=True)
        return provider.spot()
    except Exception as e:  # noqa: BLE001
        log.warning("实时行情获取失败（将使用股票列表代替）: %s", e)
        return None


def resolve_universe(provider: BaseProvider, cfg: dict, universe: str,
                     spot: pd.DataFrame | None) -> list[str]:
    """解析股票池，并统一按代码剔除创业板/科创板/北交所/港股（见 config.yaml 的 universe）。"""
    codes = _resolve_universe_raw(provider, cfg, universe, spot)
    u = cfg["universe"]
    kept = [c for c in codes if code_allowed(c, u)]
    if len(kept) < len(codes):
        print(f"  已按配置剔除 {len(codes) - len(kept)} 只（创业板/科创板/北交所/港股等）",
              file=sys.stderr)
    return kept


def _resolve_universe_raw(provider: BaseProvider, cfg: dict, universe: str,
                          spot: pd.DataFrame | None) -> list[str]:
    """把股票池名称解析为代码列表。

    watchlist           自选股（config.yaml）
    hs300/zz500/zz1000  指数成分股；其他指数写 index:000905
    all                 全市场，过滤后按成交额取前 N
    600519,000858       逗号分隔的代码
    """
    u = cfg["universe"]
    key = universe.strip().lower()
    if key == "watchlist":
        return [w["code"] for w in cfg["watchlist"]]
    if key in INDEX_ALIASES or key.startswith("index:"):
        idx = INDEX_ALIASES.get(key) or key.split(":", 1)[1]
        codes = provider.index_constituents(idx)["code"].tolist()
        if spot is not None:
            allowed = set(apply_universe_filters(spot, u)["code"])
            codes = [c for c in codes if c in allowed]
        return codes
    if key == "all":
        if spot is None:
            raise SystemExit("universe=all 需要实时行情数据，但获取失败，请稍后重试或换用 hs300。")
        df = apply_universe_filters(spot, u).sort_values("amount", ascending=False)
        return df["code"].head(int(u.get("all_top_n", 300))).tolist()
    # 其余视为逗号分隔的股票代码
    return [norm_code(c) for c in key.split(",") if c.strip()]


def _names(provider: BaseProvider, spot: pd.DataFrame | None) -> dict[str, str]:
    if spot is not None:
        return dict(zip(spot["code"], spot["name"]))
    try:
        sl = provider.stock_list()
        return dict(zip(sl["code"], sl["name"]))
    except Exception:  # noqa: BLE001
        return {}


def screen_technical(provider: BaseProvider, cfg: dict, strategy_names: list[str],
                     universe: str = "hs300", lookback: int = 1, direction: str = "buy",
                     spot: pd.DataFrame | None = None):
    """扫描股票池，返回 (信号表 DataFrame, {code: 日线DataFrame})。"""
    if spot is None:
        spot = get_spot_safe(provider)
    codes = resolve_universe(provider, cfg, universe, spot)
    if not codes:
        return pd.DataFrame(), {}
    print(f"  股票池 {universe}: {len(codes)} 只，策略: {', '.join(strategy_names)}", file=sys.stderr)
    hist = provider.daily_batch(codes, max_workers=int(cfg["data"]["max_workers"]),
                                progress=_progress)
    min_bars = int(cfg["universe"].get("min_bars", 60))
    names = _names(provider, spot)
    spot_idx = spot.set_index("code") if spot is not None else None
    strats = [S.build(n, cfg) for n in strategy_names]
    want = {"buy": {S.BUY}, "sell": {S.SELL}, "both": {S.BUY, S.SELL}}[direction]

    rows = []
    for code, df in hist.items():
        if len(df) < min_bars:
            continue
        for st in strats:
            for sig in st.latest_signals(code, df, lookback=lookback):
                if sig.direction not in want:
                    continue
                row = {"代码": code, "名称": names.get(code, ""), "策略": st.label,
                       "信号": sig.label, "信号日期": sig.date.strftime("%Y-%m-%d"),
                       "信号价": round(sig.price, 2), "说明": sig.reason,
                       "最新价": round(float(df["close"].iloc[-1]), 2)}
                if spot_idx is not None and code in spot_idx.index:
                    s = spot_idx.loc[code]
                    row["最新价"] = round(float(s.get("price", row["最新价"])), 2)
                    row["今日涨跌%"] = round(float(s.get("pct_chg", float("nan"))), 2)
                    row["总市值(亿)"] = round(float(s.get("total_mv", float("nan"))) / 1e8, 1)
                rows.append(row)
    res = pd.DataFrame(rows)
    if not res.empty:
        # 同一只股票被多个策略同时选中 → 共振数
        res["共振数"] = res.groupby(["代码", "信号"])["策略"].transform("nunique")
        res = res.sort_values(["共振数", "信号日期", "代码"], ascending=[False, False, True])
        res = res.reset_index(drop=True)
    return res, hist


def screen_multi_factor(provider: BaseProvider, cfg: dict, top: int = 30,
                        spot: pd.DataFrame | None = None) -> pd.DataFrame:
    spot = spot if spot is not None else provider.spot()
    pool = apply_universe_filters(spot, cfg["universe"])
    st = S.build("multi_factor", cfg)
    scored = st.score(pool).head(top)
    if scored.empty:
        return scored
    from .strategies.multi_factor import FACTOR_LABELS
    out = scored[["rank", "code", "name", "price", "pct_chg", "score"]
                 + [f for f in st.factors if f in scored.columns]].copy()
    if "total_mv" in out:
        out["total_mv"] = (out["total_mv"] / 1e8).round(1)
    labels = {"rank": "排名", "code": "代码", "name": "名称", "price": "最新价",
              "pct_chg": "今日涨跌%", "score": "综合得分", **FACTOR_LABELS, "total_mv": "总市值(亿)"}
    return out.rename(columns=labels).round(2)
