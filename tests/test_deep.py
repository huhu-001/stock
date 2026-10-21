from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant import auction, deep, insight, sentiment  # noqa: E402
from quant.marketdata import MarketData  # noqa: E402
from quant.timeutil import now_cn  # noqa: E402

TODAY = now_cn().strftime("%Y-%m-%d")


def minute_df(tail_move=0.0, n=240, vol_tail=1.0):
    times = list(pd.date_range(f"{TODAY} 09:31", periods=120, freq="min")) + \
        list(pd.date_range(f"{TODAY} 13:01", periods=120, freq="min"))
    price = np.full(n, 10.0)
    price[-30:] = 10.0 * (1 + np.linspace(0, tail_move, 30))
    vol = np.full(n, 100.0)
    vol[-30:] *= vol_tail
    return pd.DataFrame({"day": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times], "open": price,
                         "high": price * 1.001, "low": price * 0.999, "close": price,
                         "volume": vol * 100, "amount": price * vol * 100})


def fake_ak(fail_em=True, tail_move=0.0):
    def em_fail(**kw):
        raise ConnectionError("RemoteDisconnected")
    a = types.SimpleNamespace()
    a.stock_zh_a_hist_min_em = em_fail if fail_em else (lambda **kw: pd.DataFrame())
    a.stock_zh_a_minute = lambda **kw: minute_df(tail_move)
    a.stock_intraday_em = em_fail
    a.stock_zh_a_tick_tx_js = lambda **kw: pd.DataFrame({
        "成交时间": ["14:50:00", "14:51:00", "14:52:00"], "成交价格": [10, 10, 10],
        "成交量": [1000, 500, 10], "成交金额": [1e6, 5e5, 1e4], "性质": ["买盘", "卖盘", "买盘"]})
    a.stock_individual_fund_flow = lambda **kw: pd.DataFrame({
        "日期": ["2026-10-20", TODAY], "主力净流入-净额": [1, 2e7], "主力净流入-净占比": [1, 6.5],
        "超大单净流入-净额": [0, 1e7], "大单净流入-净额": [0, 1e7]})
    a.stock_zt_pool_em = lambda date: pd.DataFrame({"代码": ["600001", "000002", "300003"],
                                                    "连板数": [1, 3, 2], "所属行业": ["半导体"] * 3})
    a.stock_zt_pool_zbgc_em = lambda date: pd.DataFrame({"代码": ["600010"]})
    a.stock_zt_pool_dtgc_em = lambda date: pd.DataFrame({"代码": ["600020", "600021"]})
    a.stock_zt_pool_previous_em = lambda date: pd.DataFrame({"代码": ["600030"], "涨跌幅": [2.5]})
    a.stock_lhb_detail_em = em_fail
    a.stock_lhb_detail_daily_sina = lambda date: pd.DataFrame({
        "股票代码": ["600519"], "股票名称": ["贵州茅台"], "指标": ["涨幅偏离7%"]})
    a.stock_lhb_stock_detail_em = em_fail
    a.stock_zh_index_spot_sina = lambda: pd.DataFrame({
        "代码": ["sh000905", "sh000300", "sh000852", "sz399999"], "最新价": [6000, 4000, 6500, 1]})
    a.futures_zh_spot = lambda **kw: pd.DataFrame({"current_price": [3980, 5940, 6420]})
    a.stock_board_industry_name_em = em_fail
    a.stock_sector_spot = lambda indicator: pd.DataFrame({"板块": ["半导体", "银行"],
                                                          "涨跌幅": [3.2, -0.5], "公司家数": [50, 40]})
    a.stock_notice_report = lambda **kw: pd.DataFrame({
        "代码": ["000002"], "名称": ["某某"], "公告标题": ["关于收到立案告知书的公告"], "公告类型": ["风险提示"]})
    a.stock_yjyg_em = em_fail
    a.stock_restricted_release_detail_em = em_fail
    a.stock_ggcg_em = em_fail
    a.stock_individual_info_em = em_fail
    return a


@pytest.fixture
def md(tmp_path):
    return MarketData(tmp_path / "archive", ak=fake_ak(), retries=1)


def test_minute_fallback_to_sina_and_vwap(md):
    mb = md.minute_bars("600519")
    assert md.status["分时:600519"] == "新浪"
    assert mb["volume"].iloc[0] == 100        # 股 → 手
    assert mb["vwap"].iloc[-1] == pytest.approx(10.0, rel=1e-3)


def test_tail_shapes(tmp_path):
    up = MarketData(tmp_path, ak=fake_ak(tail_move=0.03), retries=1).minute_bars("600519")
    down = MarketData(tmp_path, ak=fake_ak(tail_move=-0.03), retries=1).minute_bars("600519")
    assert insight.tail_features(up)["尾盘形态"] == "尾盘急拉"
    assert "跳水" in insight.tail_features(down)["尾盘形态"]


def test_big_orders_from_tencent(md):
    bo = md.big_orders("600519")       # 阈值 20 万：前两笔是大单
    assert bo["大单买入"] == 1e6 and bo["大单卖出"] == 5e5
    assert bo["大单主动买占比"] == pytest.approx(2 / 3)


def test_sentiment_from_pools(md):
    s = sentiment.stats_from_pools(md.zt_pool("x"), md.zb_pool("x"), md.dt_pool("x"),
                                   md.zt_prev_pool("x"), zt_yesterday_count=4)
    assert s["涨停家数"] == 3 and s["炸板家数"] == 1 and s["跌停家数"] == 2
    assert s["封板率%"] == 75.0 and s["连板高度"] == 3 and s["连板晋级率%"] == 50.0
    score, level, _ = sentiment.temperature(s)
    assert 0 <= score <= 100 and level in ("冰点", "低迷", "中性", "活跃", "过热")


def test_sentiment_from_spot_estimates_limits():
    spot = pd.DataFrame({"code": ["600001", "300002", "600003", "600004"],
                         "name": ["a", "b", "c", "d"], "prev_close": [10, 10, 10, 10],
                         "price": [11.0, 12.0, 10.5, 9.0], "high": [11.0, 12.0, 11.0, 9.5]})
    s = sentiment.stats_from_spot(spot)
    assert s["涨停家数"] == 2 and s["炸板家数"] == 1 and s["跌停家数"] == 1


def test_futures_basis(md):
    b = md.futures_basis(md.index_spot())
    assert set(b["品种"]) == {"沪深300期货", "中证500期货", "中证1000期货"}
    assert b.loc[b["品种"] == "沪深300期货", "升贴水%"].iloc[0] == pytest.approx(-0.5)


def test_score_rules_and_exclusion():
    good = {"模型分": 80, "尾盘形态": "尾盘平稳", "尾盘站上均价线比例": 0.8, "收盘相对均价%": 0.3,
            "大单主动买占比": 0.7, "主力净占比%": 6, "位置": "低位", "风险事件": []}
    bad = {**good, "尾盘形态": "放量尾盘跳水", "大单主动买占比": 0.3, "风险事件": ["公告:立案"]}
    s1, plus, _, ex1 = insight.score_row(good)
    s2, _, minus, ex2 = insight.score_row(bad)
    assert s1 > s2 and not ex1 and ex2
    assert any("立案" in m for m in minus)


def test_event_flags(md):
    flags = insight.event_flags("000002", md.notices("x"), None, None, None)
    assert flags and "立案" in flags[0]


def test_run_deep_end_to_end(md):
    results = {"rule": pd.DataFrame({"代码": ["600519", "000002"], "名称": ["A", "B"],
                                     "现价": [10, 20], "涨幅%": [-1, -2], "得分": [80, 70]}),
               "pattern": pd.DataFrame({"代码": ["600519"], "名称": ["A"], "现价": [10],
                                        "涨幅%": [-1], "得分": [90]})}
    idx = pd.bdate_range(end=pd.Timestamp(TODAY), periods=80)
    h = pd.DataFrame({"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1e5,
                      "amount": 1e8, "turnover": 2.0}, index=idx)
    top, full, market = deep.run_deep(md, results, {"rule": "反转规则", "pattern": "尾盘回落型"},
                                      {"600519": h, "000002": h}, None, {"tail": {"top_n": 10}})
    assert "000002" not in set(top["代码"])                 # 立案公告被剔除
    assert full.loc[full["代码"] == "600519", "来源模型"].iloc[0] == "反转规则+尾盘回落型"
    assert market["level"] in ("冰点", "低迷", "中性", "活跃", "过热")
    day = now_cn().strftime("%Y%m%d")
    assert (md.archive_root / day / "insight.csv").exists()
    assert (md.archive_root / day / "zt_pool.csv").exists()
    assert "今日最强板块" in deep.market_text(market)


def test_auction(tmp_path):
    picks = tmp_path / "picks"
    picks.mkdir()
    pd.DataFrame({"代码": ["600001", "600002"]}).to_csv(picks / "picks_rule_20261020.csv", index=False)
    arc = tmp_path / "archive" / "20261020"
    arc.mkdir(parents=True)
    pd.DataFrame({"代码": ["600003"]}).to_csv(arc / "zb_pool.csv", index=False)
    pd.DataFrame({"代码": ["600004"]}).to_csv(arc / "dt_pool.csv", index=False)
    spot = pd.DataFrame({"code": ["600001", "600002", "600003", "600004"], "name": list("abcd"),
                         "prev_close": [10] * 4, "open": [10.4, 9.6, 10.3, 9.2], "amount": [6e6] * 4})
    res = auction.run(spot, tmp_path / "archive", "20261020", picks, {"600001": pd.DataFrame({"amount": [1e8]})})
    t = res["选股:rule"]
    assert t.loc[t["代码"] == "600001", "判断"].iloc[0].startswith("高开放量")
    assert t.loc[t["代码"] == "600002", "判断"].iloc[0].startswith("大幅低开")
    assert bool(res["昨日炸板股"]["弱转强"].iloc[0]) and bool(res["昨日跌停股"]["继续核按钮"].iloc[0])
