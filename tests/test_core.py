from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant import indicators as ind  # noqa: E402
from quant import strategies as S  # noqa: E402
from quant.config import load_config  # noqa: E402
from quant.data import MockProvider, norm_code  # noqa: E402
from quant.monitor import Monitor, merge_live_bar  # noqa: E402
from quant.notifier import Notifier  # noqa: E402
from quant.report import build_report  # noqa: E402
from quant.screener import apply_universe_filters, screen_multi_factor, screen_technical  # noqa: E402


def make_df(close):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range("2025-01-01", periods=len(close))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": 1e5, "amount": close * 1e7}, index=idx)
    df.index.name = "date"
    return df


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["output_dir"] = str(tmp_path)
    return c


# ---------------- indicators ----------------
def test_norm_code():
    assert norm_code("sh600519") == "600519"
    assert norm_code(1) == "000001"
    assert norm_code("000001.SZ") == "000001"


def test_sma_ema_macd():
    s = pd.Series(np.arange(1, 51, dtype=float))
    assert ind.sma(s, 5).iloc[-1] == pytest.approx(48.0)
    assert ind.sma(s, 5).isna().sum() == 4
    dif, dea, hist = ind.macd(s)
    assert (hist == 2 * (dif - dea)).all()
    assert dif.iloc[-1] > 0  # 单边上涨 DIF 为正


def test_rsi_bounds():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + rng.normal(0, 1, 300).cumsum())
    r = ind.rsi(s).dropna()
    assert r.between(0, 100).all()
    assert ind.rsi(pd.Series(np.arange(1, 40, dtype=float))).iloc[-1] == 100


def test_bollinger_order():
    rng = np.random.default_rng(1)
    s = pd.Series(50 + rng.normal(0, 1, 100).cumsum())
    mid, up, low = ind.bollinger(s)
    ok = mid.notna()
    assert (up[ok] >= mid[ok]).all() and (mid[ok] >= low[ok]).all()


def test_cross():
    a = pd.Series([1, 2, 3, 2, 1.0])
    b = pd.Series([2, 2, 2, 2, 2.0])
    assert ind.cross_up(a, b).tolist() == [False, False, True, False, False]
    assert ind.cross_down(a, b).tolist() == [False, False, False, False, True]


# ---------------- strategies ----------------
def test_ma_cross_signal():
    close = [10] * 30 + list(np.linspace(10, 8, 15)) + list(np.linspace(8, 12, 15))
    st = S.MACrossStrategy(fast=5, slow=20, volume_confirm=False)
    res = st.run(make_df(close))
    assert (res["signal"] == 1).sum() >= 1
    assert (res["signal"] == -1).sum() >= 1
    assert st.state(res.iloc[-1]) == "均线多头"


def test_macd_signal():
    close = list(np.linspace(20, 10, 60)) + list(np.linspace(10, 16, 30))
    res = S.MACDStrategy().run(make_df(close))
    buys = res[res["signal"] == 1]
    assert len(buys) >= 1 and "金叉" in buys["reason"].iloc[0]


def test_mean_reversion_signal():
    close = [20.0] * 40 + [19.5, 19, 18, 17, 16, 15.5, 15.3] + [17.5, 18, 18.5]
    rng = np.random.default_rng(2)
    close = np.array(close) + np.r_[rng.normal(0, 0.05, 40), np.zeros(10)]
    res = S.MeanReversionStrategy().run(make_df(close))
    assert (res["signal"] == 1).any()


def test_short_history_returns_empty():
    assert S.MACDStrategy().run(make_df([10] * 10)).empty


def test_multi_factor_ranking():
    spot = MockProvider().spot()
    scored = S.MultiFactorStrategy().score(spot)
    assert (scored["pe"] > 0).all()
    assert scored["score"].is_monotonic_decreasing
    assert scored["rank"].iloc[0] == 1


# ---------------- screener / monitor / report ----------------
def test_universe_filters(cfg):
    spot = MockProvider().spot()
    out = apply_universe_filters(spot, cfg["universe"])
    assert not out["name"].str.contains("ST").any()
    assert (out["amount"] >= cfg["universe"]["min_amount"]).all()


def test_screen_technical(cfg):
    res, hist = screen_technical(MockProvider(), cfg, ["ma_cross", "macd"], "hs300", lookback=10)
    assert len(hist) > 0
    if not res.empty:
        assert set(res["信号"]) == {"买入"}
        assert {"代码", "策略", "共振数"} <= set(res.columns)


def test_screen_multi_factor(cfg):
    res = screen_multi_factor(MockProvider(), cfg, top=10)
    assert len(res) == 10 and "综合得分" in res.columns


def test_merge_live_bar():
    df = make_df([10, 11, 12])
    today = df.index[-1] + pd.offsets.BDay(1)
    q = pd.Series({"price": 13.0, "open": 12.1, "high": 13.2, "low": 12.0, "volume": 5e4})
    out = merge_live_bar(df, q, today)
    assert len(out) == 4 and out["close"].iloc[-1] == 13.0
    out2 = merge_live_bar(out, pd.Series({**q, "price": 13.5}), today)
    assert len(out2) == 4 and out2["close"].iloc[-1] == 13.5


def test_monitor_tick_and_dedupe(cfg):
    cfg["watchlist"] = [{"code": "600519", "pct_up": -100}]   # 必然触发的价格提醒
    n = Notifier({"console": False, "macos": False}, Path(cfg["output_dir"]) / "alerts.log")
    m = Monitor(MockProvider(), cfg, n)
    m.prepare()
    board = m.tick()
    assert len(board) == 1
    lines = (Path(cfg["output_dir"]) / "alerts.log").read_text(encoding="utf-8").splitlines()
    m.tick()  # 同一天不重复提醒
    lines2 = (Path(cfg["output_dir"]) / "alerts.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1 and len(lines) == len(lines2)


def test_report(cfg, tmp_path):
    res, hist = screen_technical(MockProvider(), cfg, ["macd"], "watchlist", lookback=200,
                                 direction="both")
    p = build_report("测试", res, hist, ["macd", "mean_reversion"], tmp_path / "r.html", cfg=cfg)
    text = p.read_text(encoding="utf-8")
    assert "plotly.min.js" in text and (tmp_path / "plotly.min.js").exists()


def test_akshare_provider_mapping(tmp_path, monkeypatch):
    """用伪造的 akshare 返回值验证列名映射与缓存（不联网）。"""
    import types
    fake = types.SimpleNamespace()
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    fake.stock_zh_a_hist = lambda **kw: pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"), "股票代码": kw["symbol"], "开盘": 10.0, "收盘": 10.5,
        "最高": 11.0, "最低": 9.5, "成交量": 1000, "成交额": 1e6, "振幅": 1.0, "涨跌幅": 0.5,
        "涨跌额": 0.05, "换手率": 1.2})
    fake.stock_zh_a_spot_em = lambda: pd.DataFrame({
        "序号": [1], "代码": ["600519"], "名称": ["贵州茅台"], "最新价": ["1500.1"], "涨跌幅": [1.2],
        "今开": [1490], "最高": [1510], "最低": [1480], "昨收": [1482], "成交量": [1e4],
        "成交额": [1.5e9], "换手率": [0.3], "市盈率-动态": [25.1], "市净率": [8.2],
        "总市值": [1.9e12], "流通市值": [1.9e12], "60日涨跌幅": [3.3], "年初至今涨跌幅": [5.5]})
    monkeypatch.setitem(sys.modules, "akshare", fake)
    from quant.data import AKShareProvider
    p = AKShareProvider(cache_dir=str(tmp_path), history_days=400)
    d = p.daily("600519")
    assert list(d.columns) == ["open", "close", "high", "low", "volume", "amount", "pct_chg",
                               "turnover"]
    assert (tmp_path / "daily" / "600519_qfq.csv").exists()
    fake.stock_zh_a_hist = lambda **kw: (_ for _ in ()).throw(RuntimeError("should use cache"))
    assert len(p.daily("600519")) == 300
    s = p.spot()
    assert s.loc[0, "price"] == pytest.approx(1500.1) and s.loc[0, "pe"] == 25.1


# ---------------- 尾盘选股（反转模型） ----------------
from quant import tail  # noqa: E402


def _feat_df(close, turnover=3.0):
    df = make_df(close)
    df["turnover"] = turnover
    return df


def test_reversal_scoring_prefers_weak_stock():
    rng = np.random.default_rng(3)
    base = 10 + rng.normal(0, 0.05, 100).cumsum()
    strong = _feat_df(np.r_[base, np.linspace(base[-1], base[-1] * 1.25, 20)], turnover=8)
    weak = _feat_df(np.r_[base, np.linspace(base[-1], base[-1] * 0.85, 20)], turnover=1)
    rows = []
    for code, df in [("600001", strong), ("600002", weak), ("600003", _feat_df(np.r_[base, base[-20:]]))]:
        f = tail.stock_features(df, code).iloc[[-1]]
        rows.append(f)
    feat = pd.concat(rows)
    feat["date"] = feat.index
    sc = tail.score_cross_section(feat, {**tail.DEFAULTS, "min_amount_pct": 0})
    sc = sc.set_index("code")["score"]
    assert sc["600002"] > sc["600001"]


def test_limit_up_excluded():
    close = list(np.linspace(10, 12, 90)) + [13.2]
    f = tail.stock_features(_feat_df(close), "600000")
    assert bool(f["limit"].iloc[-1])


def test_params_ignore_legacy_keys():
    p = tail.params({"tail": {"pct_min": 2, "top_n": 5, "ml": {"window": 60}}})
    assert p["top_n"] == 5 and "pct_min" not in p
    assert p["ml"]["window"] == 60 and p["ml"]["max_depth"] == 3


def test_elapsed_fraction():
    from quant.timeutil import CN_TZ
    from datetime import datetime
    assert tail.elapsed_fraction(datetime(2026, 9, 24, 14, 30, tzinfo=CN_TZ)) == pytest.approx(210 / 240)
    assert tail.elapsed_fraction(datetime(2026, 9, 24, 10, 30, tzinfo=CN_TZ)) == pytest.approx(60 / 240)
    assert tail.elapsed_fraction(datetime(2026, 9, 26, 14, 30, tzinfo=CN_TZ)) == 1.0  # 周六


def test_tail_backtest_both_models(cfg):
    summary, trades, curve = tail.backtest(MockProvider(), cfg, "hs300", days=120)
    assert {"反转规则", "ML模型", "股票池平均"} <= set(summary.columns)
    assert trades.groupby(["model", "date"]).size().max() <= 10
    assert {"rule", "ml", "benchmark"} <= set(curve.columns)


def test_ml_no_lookahead(cfg):
    """训练集不能包含 asof 当天及前一天（其标签依赖今天收盘）。"""
    p = tail.params(cfg)
    prov = MockProvider()
    frames = [tail.stock_features(prov.daily(c), c, with_forward=True)
              for c in prov.stock_list()["code"][:30]]
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = tail.add_ml_columns(tail.tradable(panel.reset_index(), p))
    asof = panel["date"].max()
    seen = {}
    orig = tail.fit_ml
    tail.fit_ml = lambda train, mp: (seen.setdefault("d", train["date"].max()), orig(train, mp))[1]
    try:
        out = tail.ml_score_today(panel, asof, p)
    finally:
        tail.fit_ml = orig
    dates = sorted(panel["date"].unique())
    assert seen["d"] == dates[-3]
    assert out["score"].between(0, 100).all()


def test_tail_pick_both_models_and_review(cfg):
    cfg["universe"]["min_amount"] = 0
    results, hist, errors = tail.pick_live(MockProvider(), cfg, universe="hs300")
    assert not errors
    for m in ("rule", "ml"):
        assert len(results[m]) == 10 and results[m]["得分"].is_monotonic_decreasing
    d = Path(cfg["output_dir"]) / "tail_picks"
    d.mkdir(parents=True)
    day = (pd.Timestamp.today() - pd.offsets.BDay(2)).strftime("%Y%m%d")
    results["ml"].to_csv(d / f"picks_ml_{day}.csv", index=False, encoding="utf-8-sig")
    results["rule"].to_csv(d / f"picks_{day}.csv", index=False, encoding="utf-8-sig")  # 旧版文件名
    for m in ("rule", "ml"):
        summary, detail = tail.review(MockProvider(), cfg, model=m)
        assert len(summary) == 1 and len(detail) == 10


def test_fallback_sources_and_units(tmp_path, monkeypatch):
    """东方财富失败时自动换腾讯/新浪，并统一单位（成交量=手、换手率=%）。"""
    import types
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    calls = []

    def em(**kw):
        calls.append("em")
        raise ConnectionError("RemoteDisconnected")

    def tx(**kw):   # 腾讯：volume=股，turnover=小数，带 date 列和日期索引
        calls.append(("tx", kw["symbol"]))
        df = pd.DataFrame({"date": dates.date, "open": 10.0, "close": 10.5, "high": 11.0,
                           "low": 9.8, "volume": 1e6, "turnover": 0.012,
                           "amount": 1e6 * 10.4})
        df.index = pd.to_datetime(df["date"])
        return df

    def sina(**kw):
        calls.append("sina")
        return pd.DataFrame()

    fake = types.SimpleNamespace(stock_zh_a_hist=em, stock_zh_a_hist_tx=tx, stock_zh_a_daily=sina)
    monkeypatch.setitem(sys.modules, "akshare", fake)
    from quant.data import AKShareProvider
    p = AKShareProvider(cache_dir=str(tmp_path), history_days=400, retries=1)
    d = p.daily("600519")
    assert ("tx", "sh600519") in calls
    assert d["volume"].iloc[-1] == pytest.approx(1e4)       # 1e6 股 = 1e4 手
    assert d["turnover"].iloc[-1] == pytest.approx(1.2)     # 0.012 → 1.2%
    vwap = d["amount"] / (d["volume"] * 100)
    assert vwap.iloc[-1] == pytest.approx(10.4)
    assert "腾讯" in p.source_report()
    # 深市代码前缀
    p.daily("000001")
    assert ("tx", "sz000001") in calls


def test_all_sources_fail_raises(tmp_path, monkeypatch):
    import types

    def bad(**kw):
        raise ConnectionError("down")
    fake = types.SimpleNamespace(stock_zh_a_hist=bad, stock_zh_a_hist_tx=bad, stock_zh_a_daily=bad)
    monkeypatch.setitem(sys.modules, "akshare", fake)
    from quant.data import AKShareProvider
    p = AKShareProvider(cache_dir=str(tmp_path), retries=1)
    with pytest.raises(RuntimeError):
        p.daily("600000")


def test_pattern_conditions():
    rows = []
    for i in range(50):
        rows.append({"date": pd.Timestamp("2026-10-12"), "code": f"{600000 + i}",
                     "pos": i / 50, "r1": (i % 10 - 5) / 100, "turn": (i % 7) + 1.0,
                     "vol20": 0.02 + (i % 5) / 100})
    x = pd.DataFrame(rows)
    s = tail.pattern_score(x)
    ok = x[s.notna()]
    q = x.rank(pct=True, numeric_only=True)
    assert len(ok) > 0
    assert (q.loc[ok.index, "pos"] <= 0.4).all()
    assert ((q.loc[ok.index, "r1"] > 0.2) & (q.loc[ok.index, "r1"] <= 0.6)).all()
    assert s.dropna().between(0, 100).all()


def test_tail_three_models(cfg):
    cfg["universe"]["min_amount"] = 0
    results, hist, errors = tail.pick_live(MockProvider(), cfg, universe="hs300")
    assert {"short", "rule", "pattern", "trend", "ml"} <= set(results) and not errors
    assert len(results["pattern"]) <= 10
    summary, trades, curve = tail.backtest(MockProvider(), cfg, "hs300", days=120)
    assert "尾盘回落型" in summary.columns and "pattern" in curve.columns


def test_code_filter_excludes_cyb_and_hk():
    from quant.screener import code_allowed
    u = {"exclude_bj": True, "exclude_cyb": True, "exclude_kcb": False}
    assert code_allowed("600519", u) and code_allowed("000001", u) and code_allowed("688981", u)
    assert not code_allowed("300750", u) and not code_allowed("301001", u)   # 创业板
    assert not code_allowed("00700", u) and not code_allowed("hk00700", u)   # 港股
    assert not code_allowed("830799", u) and not code_allowed("900901", u)   # 北交所 / B 股


def test_resolve_universe_applies_code_filter(cfg):
    from quant.screener import resolve_universe
    cfg["universe"]["exclude_cyb"] = True
    codes = resolve_universe(MockProvider(), cfg, "600519,300750,000858", None)
    assert codes == ["600519", "000858"]


def test_trend_score_requires_uptrend():
    x = pd.DataFrame({"date": pd.Timestamp("2026-10-21"), "trend_up": [True, False, True, True],
                      "r1": [-0.01, -0.01, 0.03, -0.01], "pos": [0.2, 0.2, 0.2, 0.7]})
    s = tail.trend_score(x)
    assert s.notna().tolist() == [True, False, False, False]
    assert s.iloc[0] == 80.0


def test_low_price_excluded_from_tradable():
    close = list(np.linspace(1.6, 1.8, 90))
    f = tail.stock_features(make_df(close), "600157")
    f["date"] = f.index
    f["turn"] = 2.0
    assert tail.tradable(f, {**tail.DEFAULTS, "min_amount_pct": 0}).empty


def test_delisting_flags():
    from quant import insight
    assert insight.event_flags("600001", None, None, None, None, name="*ST某某", price=5)
    assert insight.event_flags("600157", None, None, None, None, name="永泰能源", price=1.42)
    assert not insight.event_flags("600519", None, None, None, None, name="贵州茅台", price=1500)


def test_exit_returns_rules():
    t = pd.DataFrame({
        "o1": [0.0, 0.02, -0.04, 0.0], "h1": [0.016, 0.03, 0.0, 0.005], "l1": [-0.01, 0.0, -0.05, -0.01],
        "c1": [0.01, 0.02, -0.04, 0.0],
        "o2": [0] * 4, "h2": [0] * 4, "l2": [0] * 4, "c2": [0] * 4,
        "o3": [0] * 4, "h3": [0.0, 0, 0, 0.003], "l3": [0] * 4, "c3": [0.0, 0, 0, 0.002]})
    r = tail.exit_returns(t, 0.015, -0.03, 3)
    assert r[0] == pytest.approx(0.015)      # 盘中冲到止盈
    assert r[1] == pytest.approx(0.02)       # 高开直接越过止盈，按开盘价
    assert r[2] == pytest.approx(-0.04)      # 低开越过止损，按开盘价
    assert r[3] == pytest.approx(0.002)      # 到期按第 3 天收盘


def test_short_live_and_backtest(cfg):
    cfg["universe"]["min_amount"] = 0
    results, hist, errors = tail.pick_live(MockProvider(), cfg, universe="hs300")
    s = results["short"]
    assert len(s) >= 5 and {"明日大涨≥3%概率%", "明日上涨概率%", "入选依据"} <= set(s.columns)
    assert s["明日大涨≥3%概率%"].between(0, 100).all() and s["明日大涨≥3%概率%"].is_monotonic_decreasing
    info = results["_short_info"]
    assert "market_5d" in info and isinstance(info["ok"], bool)
    summary, trades, curve = tail.backtest(MockProvider(), cfg, "hs300", days=120)
    st = summary.attrs["short"]
    assert "短线五只（大涨概率模型）" in st.index and "股票池随机买" in st.index
    assert summary.attrs["calib"] is not None
    assert s["明日上涨概率%"].between(0, 100).all()
