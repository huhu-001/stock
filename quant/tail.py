"""尾盘选股（14:30）：每天给出 10 只候选股，并用历史数据回测、每日复盘。

两个模型，每天都会计算、分别保存与复盘，方便用实盘结果比较：

1) rule  反转规则（默认）
   依据 2025-04 ~ 2026-09 中证500 真实数据的因子检验：下列指标越“强”，次日表现越差
     当日涨幅、K线实体、偏离20日线、20日涨幅、收盘相对均价、近5日放量、换手、波动率、RSI6
   对这 9 个指标做横截面反向排名取平均 → 得分 0~100。

2) ml    每日训练的机器学习模型（HistGradientBoosting）
   每次运行用最近 window 个交易日的数据重新训练：20 个量价特征（横截面排名）
   → 预测次日收益在股票池中的排名。训练只用已完全确定的标签（不偷看未来）。
   在同样数据上的滚动（walk-forward）回测里，它不如反转规则，作为对照观察。

两个模型的回测都只接近股票池平均水平，不能可靠预测次日涨跌，请当作观察名单。
过滤：涨跌停（买不到/卖不掉）、ST、成交额最低的 30%。
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BaseProvider
from .monitor import merge_live_bar
from .screener import _names, _progress, get_spot_safe, resolve_universe
from .timeutil import SESSIONS, is_trading_time, now_cn

log = logging.getLogger(__name__)

MODELS = {"rule": "反转规则", "pattern": "尾盘回落型", "ml": "ML模型", "combo": "综合精选"}

# 尾盘回落型：由 2345 条量价规则的分段检验 + 随机对照总结而来（中证500，2025-12 ~ 2026-10）
# 条件均为当天股票池内的横截面分位（0=最低, 1=最高）
PATTERN = {
    "pos_max": 0.4,            # 收盘在当日高低区间的位置：池内最低的 40%
    "r1_range": (0.2, 0.6),    # 当日涨幅：池内 20%~60% 分位（小跌或平盘，约 -1.8% ~ +0.3%）
    "turn_max": 0.6,           # 换手率：不在最高的 40%
    "vol20_max": 0.8,          # 20日波动率：不在最高的 20%
}


def pattern_score(x: pd.DataFrame) -> pd.Series:
    """满足尾盘回落型条件的行给出得分（收盘位置越低越高，0~100），不满足的为 NaN。"""
    g = x.groupby("date")
    q = {f: g[f].rank(pct=True) for f in ("pos", "r1", "turn", "vol20")}
    lo, hi = PATTERN["r1_range"]
    ok = ((q["pos"] <= PATTERN["pos_max"]) & (q["r1"] > lo) & (q["r1"] <= hi)
          & (q["turn"] <= PATTERN["turn_max"]) & (q["vol20"] <= PATTERN["vol20_max"]))
    return ((1 - q["pos"]) * 100).round(1).where(ok)

DEFAULTS = {
    "universe": "zz500",       # 选股池：zz500 / hs300 / zz1000 / all / watchlist ...
    "top_n": 10,
    "min_amount_pct": 0.3,     # 剔除成交额最低的 30%
    "cost": 0.0015,            # 回测往返交易成本
    "ml": {
        "window": 120,         # 训练窗口（交易日）
        "retrain_every": 5,    # 回测时每隔几天重新训练（实盘每次运行都重新训练）
        "max_iter": 150, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 200,
    },
}

# 反转因子：数值越小越好（得分取反向排名）
FACTORS = {
    "r1": "当日涨幅", "body": "K线实体", "ma20gap": "偏离20日线", "r20": "20日涨幅",
    "vwapgap": "收盘相对均价", "vr20": "5日/60日量比", "turn": "换手率", "vol20": "20日波动率",
    "rsi6": "RSI6",
}
ML_FEATURES = list(FACTORS) + ["r3", "r5", "r60", "vr", "turn20", "pos", "upper", "lower",
                               "ma60gap", "disthigh", "amt"]


def params(cfg: dict) -> dict:
    user = cfg.get("tail") or {}
    p = {k: user.get(k, v) for k, v in DEFAULTS.items() if k != "ml"}   # 忽略旧版废弃参数
    p["ml"] = {**DEFAULTS["ml"], **(user.get("ml") or {})}
    return p


def limit_pct(code: str) -> float:
    return 0.195 if code.startswith(("300", "301", "688")) else 0.097


def elapsed_fraction(dt: datetime | None = None) -> float:
    """当日已交易时间占全天 240 分钟的比例，用于把盘中成交量折算成全天。"""
    dt = dt or now_cn()
    if not is_trading_time(dt):
        return 1.0
    mins = 0.0
    t = dt.hour * 60 + dt.minute
    for s, e in SESSIONS:
        a, b = s.hour * 60 + s.minute, e.hour * 60 + e.minute
        mins += max(0, min(t, b) - a)
    return max(mins / 240, 0.05)


# ----------------------------------------------------------------------------
def stock_features(df: pd.DataFrame, code: str, with_forward: bool = False) -> pd.DataFrame:
    """单只股票逐日特征（最后一行可以是盘中实时K线）。"""
    c, o, h, l, v, a = df["close"], df["open"], df["high"], df["low"], df["volume"], df["amount"]
    f = pd.DataFrame(index=df.index)
    f["code"] = code
    f["close"] = c
    f["r1"] = c.pct_change()
    f["r3"] = c / c.shift(3) - 1
    f["r5"] = c / c.shift(5) - 1
    f["r20"] = c / c.shift(20) - 1
    f["r60"] = c / c.shift(60) - 1
    f["body"] = (c - o) / o
    f["ma20gap"] = c / c.rolling(20).mean() - 1
    f["ma60gap"] = c / c.rolling(60).mean() - 1
    vwap = a / (v * 100)
    vwap = vwap.where((vwap > l * 0.95) & (vwap < h * 1.05))
    f["vwapgap"] = c / vwap - 1
    f["vr"] = v / v.shift().rolling(5).mean()
    f["vr20"] = v.rolling(5).mean() / v.rolling(60).mean()
    f["turn"] = df["turnover"] if "turnover" in df else np.nan
    f["turn20"] = f["turn"].rolling(20).mean()
    f["vol20"] = f["r1"].rolling(20).std()
    rng = (h - l).replace(0, np.nan)
    f["pos"] = (c - l) / rng
    f["upper"] = (h - np.maximum(o, c)) / c
    f["lower"] = (np.minimum(o, c) - l) / c
    f["disthigh"] = c / h.shift().rolling(20).max() - 1
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 6, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 6, adjust=False).mean()
    f["rsi6"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["amt20"] = a.rolling(20).mean()
    f["amt"] = np.log(f["amt20"])
    f["limit"] = f["r1"].abs() >= limit_pct(code)
    if with_forward:
        f["ret_open"] = o.shift(-1) / c - 1
        f["ret_high"] = h.shift(-1) / c - 1
        f["ret_close"] = c.shift(-1) / c - 1
        f["ret_2d"] = c.shift(-2) / c - 1
    return f


def tradable(feat: pd.DataFrame, p: dict) -> pd.DataFrame:
    """剔除涨跌停、成交额最低的一部分、因子缺失过多的行。feat 需含 date 列。"""
    x = feat.reset_index(drop=True).dropna(subset=["r1", "amt20"]).copy()
    x = x[~x["limit"]]
    x = x[x.groupby("date")["amt20"].rank(pct=True) > p["min_amount_pct"]]
    x = x[x[list(FACTORS)].notna().sum(axis=1) >= len(FACTORS) - 2]
    return x


def score_cross_section(feat: pd.DataFrame, p: dict) -> pd.DataFrame:
    """反转规则：按日期横截面打分，返回过滤后带 score(0~100) 的数据。"""
    x = tradable(feat, p)
    ranks = x.groupby("date")[list(FACTORS)].rank(pct=True)
    x["score"] = ((1 - ranks).mean(axis=1) * 100).round(1)
    return x


# ---------------------------- ML 模型 -------------------------------------
def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise RuntimeError("ML 模型需要 scikit-learn，请执行: pip3 install scikit-learn") from e


def add_ml_columns(x: pd.DataFrame) -> pd.DataFrame:
    """横截面排名特征（-0.5~0.5，缺失填 0）与标签（次日收益在池内的排名）。"""
    k = x.groupby("date")[ML_FEATURES].rank(pct=True) - 0.5
    x = x.copy()
    for f in ML_FEATURES:
        x["k_" + f] = k[f].fillna(0.0)
    if "ret_close" in x:
        x["label"] = x.groupby("date")["ret_close"].rank(pct=True) - 0.5
    return x


def fit_ml(train: pd.DataFrame, mp: dict):
    _require_sklearn()
    from sklearn.ensemble import HistGradientBoostingRegressor
    train = train.dropna(subset=["label"])
    model = HistGradientBoostingRegressor(
        max_iter=int(mp["max_iter"]), learning_rate=float(mp["learning_rate"]),
        max_depth=int(mp["max_depth"]), min_samples_leaf=int(mp["min_samples_leaf"]),
        l2_regularization=1.0, random_state=0)
    model.fit(train[["k_" + f for f in ML_FEATURES]].values, train["label"].values)
    return model


def predict_ml(model, x: pd.DataFrame) -> pd.Series:
    pred = model.predict(x[["k_" + f for f in ML_FEATURES]].values)
    s = pd.Series(pred, index=x.index)
    return (s.groupby(x["date"]).rank(pct=True) * 100).round(1)   # 池内百分位 0~100


def ml_score_today(panel: pd.DataFrame, today, p: dict) -> pd.DataFrame:
    """用 today 之前、标签已完全确定的数据训练，给 today 打分。"""
    mp = p["ml"]
    dates = sorted(panel["date"].unique())
    hist_dates = [d for d in dates if d < today]
    # 最近一个历史日的标签依赖今天的收盘（盘中还没确定），不参与训练
    train_dates = hist_dates[:-1][-int(mp["window"]):]
    if len(train_dates) < 40:
        raise RuntimeError(f"ML 训练数据不足（{len(train_dates)} 天），至少需要 40 个交易日")
    train = panel[panel["date"].isin(train_dates)]
    model = fit_ml(train, mp)
    now = panel[panel["date"] == today].copy()
    now["score"] = predict_ml(model, now)
    return now


def _tags(r: pd.Series) -> str:
    t = []
    if r["r20"] <= -0.08:
        t.append(f"20日回调{r['r20'] * 100:.0f}%")
    elif r["r20"] >= 0.1:
        t.append(f"20日上涨{r['r20'] * 100:.0f}%")
    if r["ma20gap"] < -0.03:
        t.append("低于20日线")
    if r["vr20"] < 0.8:
        t.append("缩量")
    elif r["vr20"] > 1.5:
        t.append("放量")
    if r["vol20"] < 0.02:
        t.append("低波动")
    if r["rsi6"] < 30:
        t.append("RSI超卖")
    elif r["rsi6"] > 70:
        t.append("RSI超买")
    if r["vwapgap"] < -0.005:
        t.append("尾盘弱于均价")
    if r.get("ma60gap", 0) > 0:
        t.append("在60日线上")
    return "、".join(t)


def _format(top: pd.DataFrame, names: dict) -> pd.DataFrame:
    out = []
    for _, r in top.iterrows():
        out.append({"代码": r["code"], "名称": names.get(r["code"], ""),
                    "现价": round(float(r["close"]), 2), "涨幅%": round(float(r["r1"]) * 100, 2),
                    "得分": r["score"], "20日涨幅%": round(float(r["r20"]) * 100, 1),
                    "距20日线%": round(float(r["ma20gap"]) * 100, 1),
                    "换手%": round(float(r["turn"]), 2) if pd.notna(r["turn"]) else None,
                    "特征": _tags(r)})
    res = pd.DataFrame(out)
    if not res.empty:
        res.insert(0, "排名", range(1, len(res) + 1))
    return res


# ----------------------------------------------------------------------------
def pick_live(provider: BaseProvider, cfg: dict, spot: pd.DataFrame | None = None,
              universe: str | None = None, models=("rule", "pattern", "ml")):
    """实时尾盘选股。返回 ({model: 结果表}, {code: 含实时K线的日线}, {model: 错误信息})。"""
    p = params(cfg)
    universe = universe or p["universe"]
    if spot is None:
        spot = get_spot_safe(provider)   # 失败时退回用缓存的日线收盘数据
    codes = resolve_universe(provider, cfg, universe, spot)
    names = _names(provider, spot)
    if not any(names.get(c) for c in codes[:5]):
        try:
            idx = provider.index_constituents({"zz500": "000905", "hs300": "000300",
                                               "zz1000": "000852"}.get(universe, "000905"))
            names.update(dict(zip(idx["code"], idx["name"])))
        except Exception:  # noqa: BLE001
            pass
    print(f"  选股池 {universe}: {len(codes)} 只，下载/读取日线…", file=sys.stderr, flush=True)
    hist = provider.daily_batch(codes, max_workers=int(cfg["data"]["max_workers"]),
                                progress=_progress)
    today = pd.Timestamp(now_cn().date())
    el = elapsed_fraction()
    sidx = spot.set_index("code") if spot is not None else None
    frames, live_hist = [], {}
    for code, df in hist.items():
        if len(df) < 80:
            continue
        live = df
        if sidx is not None and code in sidx.index:
            live = merge_live_bar(df, sidx.loc[code], today)
            if el < 1 and live.index[-1] == today:   # 盘中成交量/额/换手折算成全天
                for col in ("volume", "amount", "turnover"):
                    if col in live:
                        live.iloc[-1, live.columns.get_loc(col)] /= el
        live_hist[code] = live
        frames.append(stock_features(live, code, with_forward="ml" in models))
    if not frames:
        return {}, {}, {}
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    asof = panel["date"].max()                      # 今天（或最近交易日）
    last_dates = panel.groupby("code")["date"].max()
    stale = last_dates[last_dates < asof].index
    if len(stale):
        print(f"  ⚠ {len(stale)} 只股票数据未更新到 {asof:%Y-%m-%d}（停牌或下载失败），已排除",
              file=sys.stderr)
        if len(stale) > len(last_dates) * 0.3:
            raise SystemExit(f"超过 30% 的股票数据不是最新的（{len(stale)}/{len(last_dates)}），"
                             "结果不可靠。请稍后重新运行，或先执行 python3 main.py update -u zz500")
        panel = panel[~panel["code"].isin(stale)]
    results, errors = {}, {}
    if "ml" in models:
        panel = tradable(panel, p)
        panel = add_ml_columns(panel)
    last = panel[panel["date"] == asof]
    if "pattern" in models:
        base = last if "ml" in models else tradable(last, p)
        sc = base.assign(score=pattern_score(base)).dropna(subset=["score"])
        results["pattern"] = _format(sc.sort_values("score", ascending=False)
                                     .head(int(p["top_n"])), names)
    if "rule" in models:
        sc = score_cross_section(last, p) if "ml" not in models else _rule_on(last)
        results["rule"] = _format(sc.sort_values("score", ascending=False).head(int(p["top_n"])),
                                  names)
    if "ml" in models:
        try:
            sc = ml_score_today(panel, asof, p)
            results["ml"] = _format(sc.sort_values("score", ascending=False)
                                    .head(int(p["top_n"])), names)
        except Exception as e:  # noqa: BLE001
            errors["ml"] = str(e)
    return results, live_hist, errors


def _rule_on(x: pd.DataFrame) -> pd.DataFrame:
    """对已经过滤过的数据直接计算反转得分。"""
    x = x.copy()
    ranks = x.groupby("date")[list(FACTORS)].rank(pct=True)
    x["score"] = ((1 - ranks).mean(axis=1) * 100).round(1)
    return x


# ----------------------------------------------------------------------------
def backtest(provider: BaseProvider, cfg: dict, universe: str | None = None, days: int = 250,
             models=("rule", "pattern", "ml")):
    """日线回测：以当日收盘价近似 14:30 买入价，每天取得分前 N，统计次日表现。

    ML 模型为滚动（walk-forward）回测：每个测试日只用此前、标签已确定的数据训练。
    返回 (汇总 DataFrame[指标 × 模型], 交易明细, 每日收益 DataFrame)。
    局限：收盘价≠14:30 价格；使用当前指数成分股（幸存者偏差）。
    """
    p = params(cfg)
    universe = universe or p["universe"]
    spot = get_spot_safe(provider)
    codes = resolve_universe(provider, cfg, universe, spot)
    print(f"  回测股票池 {universe}: {len(codes)} 只，最近 {days} 个交易日", file=sys.stderr)
    print("  下载日线（首次较慢，之后读缓存）…", file=sys.stderr, flush=True)
    hist = provider.daily_batch(codes, max_workers=int(cfg["data"]["max_workers"]),
                                progress=_progress)
    frames = [stock_features(df, code, with_forward=True) for code, df in hist.items()
              if len(df) >= 80]
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    allf = pd.concat(frames)
    allf.index.name = "date"
    allf = allf.reset_index().dropna(subset=["ret_close"])
    all_dates = sorted(allf["date"].unique())
    pool = tradable(allf, p)
    pool = add_ml_columns(pool) if "ml" in models else pool
    mp = p["ml"]

    test_dates = all_dates[-days:]
    if "ml" in models:
        first = all_dates.index(test_dates[0])
        if first < 42:   # 保证 ML 至少有 40 天训练数据
            test_dates = all_dates[42:]
    test = pool[pool["date"].isin(test_dates)].copy()
    if test.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    scores = {}
    if "rule" in models:
        scores["rule"] = _rule_on(test)["score"]
    if "pattern" in models:
        scores["pattern"] = pattern_score(test)
    if "ml" in models:
        _require_sklearn()
        pred = pd.Series(np.nan, index=test.index)
        didx = {d: i for i, d in enumerate(all_dates)}
        model = None
        print(f"  ML 滚动训练 {len(test_dates)} 个交易日（每 {mp['retrain_every']} 天重训）…",
              file=sys.stderr, flush=True)
        for j, d in enumerate(test_dates):
            i = didx[d]
            if model is None or j % int(mp["retrain_every"]) == 0:
                tr_dates = all_dates[max(0, i - 1 - int(mp["window"])):i - 1]  # 截止 d-2
                model = fit_ml(pool[pool["date"].isin(tr_dates)], mp)
            m = test["date"] == d
            pred[m] = model.predict(test.loc[m, ["k_" + f for f in ML_FEATURES]].values)
        scores["ml"] = pred

    bench = test.groupby("date")["ret_close"].mean()
    segs = np.array_split(np.array(sorted(test["date"].unique())), 3)

    def pct(v):
        return round(float(v) * 100, 2)

    summary, curves, trades = {}, {"benchmark": bench}, []
    for m, s in scores.items():
        t = test.assign(score=s).dropna(subset=["score"])
        pk = t.sort_values(["date", "score"], ascending=[True, False]).groupby("date") \
            .head(int(p["top_n"]))
        daily = pk.groupby("date")["ret_close"].mean().reindex(bench.index).fillna(0)
        seg = " | ".join(f"{pct(daily.loc[sd].mean()):+.2f}" for sd in segs)
        summary[MODELS[m]] = {
            "次日开盘平均收益%": pct(pk["ret_open"].mean()),
            "次日最高平均涨幅%": pct(pk["ret_high"].mean()),
            "次日收盘平均收益%": pct(pk["ret_close"].mean()),
            "次日收盘胜率%": pct((pk["ret_close"] > 0).mean()),
            "持有2日平均收益%": pct(pk["ret_2d"].mean()),
            "分段次日均值%(前|中|后)": seg,
            "累计收益%(不计成本)": pct((1 + daily).prod() - 1),
            "累计收益%(扣成本)": pct((1 + daily - p["cost"]).prod() - 1),
        }
        curves[m] = daily - p["cost"]
        trades.append(pk.assign(model=MODELS[m])[["model", "date", "code", "score", "r1", "r20",
                                                  "ret_open", "ret_high", "ret_close",
                                                  "ret_2d"]])
    summary["股票池平均"] = {
        "次日开盘平均收益%": pct(test["ret_open"].mean()),
        "次日最高平均涨幅%": pct(test["ret_high"].mean()),
        "次日收盘平均收益%": pct(test["ret_close"].mean()),
        "次日收盘胜率%": pct((test["ret_close"] > 0).mean()),
        "持有2日平均收益%": pct(test["ret_2d"].mean()),
        "分段次日均值%(前|中|后)": " | ".join(f"{pct(bench.loc[sd].mean()):+.2f}" for sd in segs),
        "累计收益%(不计成本)": pct((1 + bench).prod() - 1),
        "累计收益%(扣成本)": "-",
    }
    summary_df = pd.DataFrame(summary)
    summary_df.index.name = f"{len(test_dates)}个交易日"
    return summary_df, pd.concat(trades, ignore_index=True), pd.DataFrame(curves)


# ----------------------------------------------------------------------------
def _pick_file(out_dir: str | Path, model: str, day: str) -> Path:
    return Path(out_dir) / "tail_picks" / f"picks_{model}_{day}.csv"


def save_picks(res: pd.DataFrame, out_dir: str | Path, model: str = "rule") -> Path | None:
    if res is None or res.empty:
        return None
    path = _pick_file(out_dir, model, now_cn().strftime("%Y%m%d"))
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _list_pick_files(d: Path, model: str) -> list[tuple[pd.Timestamp, Path]]:
    out = []
    for f in d.glob("picks_*.csv"):
        parts = f.stem.split("_")          # picks_rule_20260924 / 旧版 picks_20260924
        m, day = (parts[1], parts[2]) if len(parts) == 3 else ("rule", parts[1])
        if m == model:
            out.append((pd.Timestamp(datetime.strptime(day, "%Y%m%d")), f))
    return sorted(out)


def review(provider: BaseProvider, cfg: dict, spot: pd.DataFrame | None = None,
           max_files: int = 30, model: str = "rule") -> tuple[pd.DataFrame, pd.DataFrame]:
    """复盘历史选股：每期选股在“下一个交易日”的开盘/最高/收盘表现（相对选股价）。"""
    d = Path(cfg["output_dir"]) / "tail_picks"
    today = pd.Timestamp(now_cn().date())
    files = [(day, f) for day, f in _list_pick_files(d, model) if day < today][-max_files:]
    if not files:
        return pd.DataFrame(), pd.DataFrame()
    picks = []
    for day, f in files:
        df = pd.read_csv(f, dtype={"代码": str})
        df["选股日"] = day
        picks.append(df)
    picks = pd.concat(picks, ignore_index=True)
    hist = provider.daily_batch(picks["代码"].unique(),
                                max_workers=int(cfg["data"]["max_workers"]))
    sidx = spot.set_index("code") if spot is not None else None
    rows = []
    for _, r in picks.iterrows():
        code, day = r["代码"], r["选股日"]
        df = hist.get(code)
        if df is None:
            continue
        if sidx is not None and code in sidx.index:
            df = merge_live_bar(df, sidx.loc[code], today)
        after = df[df.index > day]
        if after.empty:
            continue
        nb = after.iloc[0]
        base = float(r["现价"])
        intraday = after.index[0] == today and is_trading_time()
        rows.append({"选股日": day.strftime("%Y-%m-%d"), "代码": code, "名称": r.get("名称", ""),
                     "选股价": base, "得分": r.get("得分"),
                     "次日开盘%": round((nb["open"] / base - 1) * 100, 2),
                     "次日最高%": round((nb["high"] / base - 1) * 100, 2),
                     "次日收盘%": round((nb["close"] / base - 1) * 100, 2),
                     "状态": "盘中" if intraday else "已收盘"})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), detail
    g = detail.groupby("选股日")
    summary = pd.DataFrame({
        "只数": g.size(),
        "次日开盘均%": g["次日开盘%"].mean().round(2),
        "次日最高均%": g["次日最高%"].mean().round(2),
        "次日收盘均%": g["次日收盘%"].mean().round(2),
        "收盘上涨占比%": (g["次日收盘%"].apply(lambda s: (s > 0).mean()) * 100).round(0),
        "状态": g["状态"].first(),
    }).reset_index()
    return summary, detail
