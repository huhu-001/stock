#!/usr/bin/env python33
"""A股量化工具 —— 命令行入口

示例:
  python3 main.py strategies                          # 查看内置策略
  python3 main.py screen -s macd -u hs300 --report    # 沪深300中今天出现 MACD 金叉的股票
  python3 main.py screen -s ma_cross,macd -u all -l 3 # 全市场活跃股近3日均线/MACD信号
  python3 main.py screen -s multi_factor --top 30     # 多因子打分前30
  python3 main.py monitor                             # 盘中监控自选股
  python3 main.py chart 600519 -s macd,mean_reversion # 生成个股K线图
  python3 main.py tail                                # 尾盘选股（建议每天 14:30 运行）
  python3 main.py tail --backtest                     # 回测两个尾盘模型（反转规则 vs ML）
  python3 main.py tail -m ml                          # 以 ML 模型为主输出
  python3 main.py tail --fast                         # 只跑三个模型，跳过扩展数据评估
  python3 main.py market                              # 市场情绪温度（涨跌停/封板率/连板/升贴水/主线）
  python3 main.py auction                             # 次日 9:25 后：竞价验证昨日选股
  加 --demo 可使用离线模拟数据体验全部功能
"""
from __future__ import annotations

import argparse
import warnings
import logging
import sys
import webbrowser
from pathlib import Path

warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

from quant import strategies as S
from quant.config import load_config
from quant.data import make_provider, norm_code
from quant.timeutil import now_cn

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)

DISCLAIMER = "⚠ 仅供学习研究，不构成投资建议。股市有风险，入市需谨慎。"


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _open(path: Path, no_open: bool):
    print(f"报告已生成: {path}")
    if not no_open:
        try:
            webbrowser.open(path.resolve().as_uri())
        except Exception:  # noqa: BLE001
            pass


def cmd_strategies(args, cfg):
    print("内置策略:\n")
    for name, cls in S.REGISTRY.items():
        params = (cfg.get("strategies") or {}).get(name, getattr(cls, "defaults", {}))
        if name == "multi_factor":
            params = {k: v["weight"] for k, v in (params.get("factors") or {}).items()}
        print(f"  {name:<16}{cls.label}\n  {'':<16}{cls.description}\n  {'':<16}参数: {params}\n")


def cmd_screen(args, cfg):
    from quant import screener
    from quant.report import build_report

    provider = make_provider(cfg, demo=args.demo)
    names = _split(args.strategy)
    if names == ["all"]:
        names = list(S.TECHNICAL)
    for n in names:
        if n not in S.REGISTRY:
            raise SystemExit(f"未知策略 {n}，可选: {', '.join(S.REGISTRY)}")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_cn().strftime("%Y%m%d_%H%M")
    spot = screener.get_spot_safe(provider)

    if "multi_factor" in names:
        if spot is None:
            raise SystemExit("多因子选股需要实时行情快照，获取失败，请稍后重试。")
        res = screener.screen_multi_factor(provider, cfg, top=args.top, spot=spot)
        print(f"\n多因子打分 Top {args.top}（过滤后股票池 "
              f"{len(screener.apply_universe_filters(spot, cfg['universe']))} 只）:\n")
        print(res.to_string(index=False) if not res.empty else "无结果")
        csv = out_dir / f"screen_multi_factor_{stamp}.csv"
        res.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存: {csv}")
        if args.report and not res.empty:
            tech = [n for n in names if n != "multi_factor"] or list(S.TECHNICAL)
            hist = provider.daily_batch(res["代码"].head(args.charts),
                                        max_workers=int(cfg["data"]["max_workers"]))
            p = build_report(f"多因子选股 Top {args.top}", res, hist, tech,
                             out_dir / f"report_multi_factor_{stamp}.html",
                             max_charts=args.charts, subtitle="因子: " + ", ".join(
                                 (cfg["strategies"].get("multi_factor") or {}).get("factors", {})),
                             cfg=cfg)
            _open(p, args.no_open)
        names = [n for n in names if n != "multi_factor"]
        if not names:
            print("\n" + DISCLAIMER)
            return

    res, hist = screener.screen_technical(provider, cfg, names, universe=args.universe,
                                          lookback=args.lookback, direction=args.direction,
                                          spot=spot)
    label = "/".join(S.REGISTRY[n].label for n in names)
    print(f"\n{label} · 股票池 {args.universe} · 近 {args.lookback} 日 · "
          f"{ {'buy': '买入', 'sell': '卖出', 'both': '买卖'}[args.direction]}信号: "
          f"{res['代码'].nunique() if not res.empty else 0} 只\n")
    print(res.to_string(index=False) if not res.empty else "没有符合条件的股票。")
    if not res.empty:
        csv = out_dir / f"screen_{'_'.join(names)}_{stamp}.csv"
        res.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存: {csv}")
    if args.report:
        p = build_report(f"选股结果 · {label}", res, hist, names,
                         out_dir / f"report_{'_'.join(names)}_{stamp}.html",
                         max_charts=args.charts,
                         subtitle=f"股票池 {args.universe} · 近 {args.lookback} 日信号", cfg=cfg)
        _open(p, args.no_open)
    print("\n" + DISCLAIMER)


def cmd_monitor(args, cfg):
    from quant.monitor import Monitor
    from quant.notifier import Notifier

    provider = make_provider(cfg, demo=args.demo)
    if args.codes:
        cfg["watchlist"] = [{"code": norm_code(c)} for c in _split(args.codes)]
    ncfg = cfg["monitor"]["notify"]
    if args.no_notify:
        ncfg = {**ncfg, "macos": False, "webhook": {}}
    notifier = Notifier(ncfg, Path(cfg["output_dir"]) / "alerts.log")
    m = Monitor(provider, cfg, notifier,
                strategy_names=_split(args.strategy) if args.strategy else None,
                interval=args.interval)
    print(DISCLAIMER)
    try:
        m.run(once=args.once, force=args.force or args.demo)
    except KeyboardInterrupt:
        print("\n已退出监控。")


def cmd_chart(args, cfg):
    from quant.report import build_report

    provider = make_provider(cfg, demo=args.demo)
    names = _split(args.strategy) if args.strategy else list(S.TECHNICAL)
    codes = [norm_code(c) for c in args.codes]
    hist = provider.daily_batch(codes, max_workers=int(cfg["data"]["max_workers"]))
    try:
        sl = provider.stock_list()
        nm = dict(zip(sl["code"], sl["name"]))
    except Exception:  # noqa: BLE001
        nm = {}
    rows = []
    for c in codes:
        if c not in hist:
            print(f"未获取到 {c} 的数据")
            continue
        df = hist[c]
        row = {"代码": c, "名称": nm.get(c, ""), "日期": df.index[-1].strftime("%Y-%m-%d"),
               "收盘": round(df["close"].iloc[-1], 2)}
        for n in names:
            st = S.build(n, cfg)
            r = st.run(df)
            if not r.empty:
                row[st.label] = st.state(r.iloc[-1])
                last = r[r["signal"] != 0].tail(1)
                if not last.empty:
                    row[st.label] += f"（最近信号 {last.index[-1]:%m-%d} " \
                                     f"{'买' if last['signal'].iloc[-1] > 0 else '卖'}）"
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    stamp = now_cn().strftime("%Y%m%d_%H%M")
    p = build_report("个股技术分析", table, hist, names,
                     Path(cfg["output_dir"]) / f"chart_{'_'.join(codes)}_{stamp}.html",
                     max_charts=len(codes), subtitle="策略: " + ", ".join(names), cfg=cfg)
    _open(p, args.no_open)


def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def cmd_tail(args, cfg):
    from quant import tail
    from quant.notifier import Notifier
    from quant.report import build_report
    from quant.screener import get_spot_safe
    from quant.timeutil import is_trading_time

    provider = make_provider(cfg, demo=args.demo)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_cn().strftime("%Y%m%d_%H%M")
    args.universe = args.universe or tail.params(cfg)["universe"]
    models = (["short", "rule", "pattern", "trend", "ml"] if _has_sklearn()
              else ["short", "rule", "pattern", "trend"])
    if args.model == "ml" and "ml" not in models:
        raise SystemExit("ML 模型需要 scikit-learn，请执行: pip3 install scikit-learn")
    if "ml" not in models:
        print("提示：未安装 scikit-learn，只运行反转规则（pip3 install scikit-learn 可启用 ML 模型）")
    primary = args.model

    if args.backtest:
        summary, trades, curve = tail.backtest(provider, cfg, universe=args.universe,
                                               days=args.days, models=models)
        if summary.empty:
            raise SystemExit("回测没有可用数据。")
        print("\n尾盘选股 · 历史回测（以收盘价近似 14:30 买入，ML 为滚动训练）\n")
        short_tbl = summary.attrs.get("short")
        if short_tbl is not None:
            print(f"【{short_tbl.index.name}】\n")
            print(short_tbl.to_string())
            cal = summary.attrs.get("calib")
            if cal is not None:
                print("\n【大涨概率模型校准：预测概率越高，实际大涨比例是否越高】\n")
                print(cal.to_string())
            print("\n【每日换仓（次日收盘卖出）对比】\n")
        print(summary.to_string())
        csv = out_dir / f"tail_backtest_trades_{stamp}.csv"
        trades.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"\n逐笔明细: {csv}")
        import plotly.graph_objects as go
        eq = (1 + curve.fillna(0)).cumprod()
        colors = {"rule": "#e03131", "pattern": "#f08c00", "trend": "#2f9e44", "ml": "#1c7ed6",
                  "benchmark": "#868e96"}
        labels = {"rule": "反转规则（扣成本）", "pattern": "尾盘回落型（扣成本）",
                  "trend": "趋势回踩（扣成本）", "ml": "ML模型（扣成本）", "benchmark": "股票池等权"}
        fig = go.Figure([go.Scatter(x=eq.index, y=eq[c], name=labels[c],
                                    line=dict(color=colors[c])) for c in eq.columns])
        fig.update_layout(title="净值曲线（每日换仓）", template="plotly_white", height=420,
                          legend=dict(orientation="h", y=1.08))
        tbl = summary.reset_index().rename(columns={summary.index.name: "指标"}).astype(str)
        st_tbl = summary.attrs.get("short")
        ext = [("短线卖法回测（" + st_tbl.index.name + "）", st_tbl.reset_index().rename(
            columns={st_tbl.index.name: "模型"}).astype(str))] if st_tbl is not None else []
        p = build_report("尾盘选股 · 历史回测", tbl, None, [], out_dir / f"tail_backtest_{stamp}.html",
                         subtitle=f"股票池 {args.universe} · {summary.index.name}", cfg=cfg,
                         extra_tables=ext,
                         extra_html="<div class='chart'>" + fig.to_html(
                             full_html=False, include_plotlyjs="directory") + "</div>")
        _open(p, args.no_open)
        print("\n" + DISCLAIMER)
        return

    if not is_trading_time() and not args.demo:
        print("提示：当前不在交易时段，将使用最新行情（收盘后即为收盘数据）。")
    spot = get_spot_safe(provider)
    if spot is None:
        print("⚠ 实时行情获取失败，将使用缓存中最近一个交易日的收盘数据。")

    # 1) 复盘之前的选股（两个模型分别统计）
    extra = []
    for m in models + ["combo"]:
        rev_sum, rev_detail = tail.review(provider, cfg, spot=spot, model=m)
        if rev_sum.empty:
            continue
        print(f"\n📋 {tail.MODELS[m]} · 历史选股复盘（相对选股价）:\n")
        print(rev_sum.tail(10).to_string(index=False))
        done = rev_detail[rev_detail["状态"] == "已收盘"]
        if not done.empty:
            print(f"\n  累计 {len(done)} 笔：次日收盘平均 {done['次日收盘%'].mean():+.2f}%，"
                  f"上涨占比 {(done['次日收盘%'] > 0).mean() * 100:.0f}%，"
                  f"次日最高平均 {done['次日最高%'].mean():+.2f}%")
        extra.append((f"{tail.MODELS[m]} · 历史选股复盘", rev_sum.tail(15)))
    if args.review:
        return

    # 2) 今日选股（各模型）
    results, hist, errors = tail.pick_live(provider, cfg, spot=spot, universe=args.universe,
                                           models=models)
    short_info = results.pop("_short_info", None)
    short_all = results.pop("short", None)
    for m, err in errors.items():
        print(f"⚠ {tail.MODELS[m]} 运行失败: {err}")
    others = [m for m in models if m != "short"]
    if args.verbose_models:
        for m in others:
            res = results.get(m)
            if res is None:
                continue
            print(f"\n— {tail.MODELS[m]}（参考）—\n")
            print(res.to_string(index=False) if not res.empty else "没有可用数据。")
    for m in others:
        if results.get(m) is not None:
            tail.save_picks(results[m], out_dir, model=m)

    # 3) 扩展数据：市场情绪 + 风险过滤（公告/ST/低价/放量跳水等）
    combo, market, full = None, None, None
    if args.fast or args.demo:
        print("\n（已跳过扩展数据评估：--fast 或 --demo）")
    else:
        from quant import deep
        from quant.marketdata import MarketData
        md = MarketData(Path(cfg["data"]["cache_dir"]).parent / "archive",
                        big_order_amount=float(cfg.get("deep", {}).get("big_order_amount", 2e5)))
        cand = dict(results)
        if short_all is not None:
            cand["short"] = short_all
        try:
            combo, full, market = deep.run_deep(md, cand, tail.MODELS, hist, spot, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"⚠ 扩展数据评估失败：{e}")
        if market:
            print("\n🌡 " + deep.market_text(market))
        st = md.status_table()
        if not st.empty:
            print("数据来源：" + "；".join(f"{r['数据']} {r['来源/状态']}" for _, r in st.iterrows()))
        if combo is not None and not combo.empty:
            tail.save_picks(combo.drop(columns=["排名"]).rename(columns={"综合分": "得分"})
                            .assign(排名=range(1, len(combo) + 1)), out_dir, model="combo")

    # 4) 短线五只（主输出）
    sp = tail.params(cfg)["short"]
    short = pd.DataFrame()
    if short_all is not None and not short_all.empty:
        short = short_all.copy()
        if full is not None and not full.empty:
            bad = full[full["剔除"]]
            reasons = dict(zip(bad["代码"], bad["扣分项"].fillna("")))
            dropped = short[short["代码"].isin(reasons)]
            for _, r in dropped.iterrows():
                print(f"  剔除 {r['名称']}({r['代码']})：{reasons[r['代码']][:60]}")
            short = short[~short["代码"].isin(reasons)]
            extra_cols = full.set_index("代码")[[c for c in ("趋势", "尾盘形态", "大单主动买占比")
                                                 if c in full.columns]]
            short = short.join(extra_cols, on="代码")
        short = short.head(int(sp["top_n"])).reset_index(drop=True)
        short["排名"] = range(1, len(short) + 1)
    ok = bool(short_info and short_info["ok"])
    m5 = short_info["market_5d"] if short_info else float("nan")
    big_col = f"明日大涨≥{sp['big_move'] * 100:.0f}%概率%"
    print(f"\n🎯 {now_cn():%Y-%m-%d %H:%M} 短线五只（综合所有模型，按“明天大涨”概率排名）\n")
    if short.empty:
        print("没有可用数据。")
    else:
        cols = [c for c in ("排名", "代码", "名称", "现价", "涨幅%", big_col, "明日上涨概率%",
                            "入选依据", "趋势",
                            "尾盘形态", "20日涨幅%", "特征") if c in short.columns]
        print(short[cols].to_string(index=False))
        path = tail.save_picks(short, out_dir, model="short")
        if path:
            print(f"已保存: {path}（明天运行时自动复盘）")
    print(f"\n股票池近 5 日 {m5 * 100:+.1f}%。")
    print(f"说明：“明日大涨概率”= 综合模型（四个模型的信号 + 20 个量价特征，每天重新训练）预测明天收盘涨幅 ≥"
          f"{sp['big_move'] * 100:.0f}% 的概率。普通股票这个概率大约只有 1 成，排在前面的通常是 2~3 成；"
          "它是统计倾向，不是保证，大涨概率高的股票往往大跌概率也高。")

    ncfg = cfg["monitor"]["notify"]
    if args.no_notify:
        ncfg = {**ncfg, "macos": False, "webhook": {}}
    n = Notifier(ncfg, out_dir / "alerts.log")
    if short.empty:
        n.send("短线五只", "没有可用数据", "info")
    else:
        body = "；".join(f"{r['名称']}({r['代码']}) 大涨概率{r.get(big_col, '')}%"
                        for _, r in short.iterrows())
        n.send("短线五只（明天大涨概率排名）", body, "buy")

    tables = [(f"{tail.MODELS[m]}（参考）", results[m]) for m in others if m in results]
    if combo is not None and not combo.empty:
        tables.insert(0, ("综合精选（参考）", combo))
    p = build_report(f"短线五只 {now_cn():%Y-%m-%d}", short, hist,
                     ["ma_cross", "macd"], out_dir / f"tail_{stamp}.html", max_charts=args.charts,
                     subtitle=f"按明天大涨（收盘≥{sp['big_move'] * 100:.0f}%）概率排名；股票池近5日{m5 * 100:+.1f}%",
                     cfg=cfg,
                     extra_tables=tables + extra)
    _open(p, args.no_open)
    print("\n" + DISCLAIMER)


def cmd_market(args, cfg):
    from quant import deep
    from quant.marketdata import MarketData
    from quant.screener import get_spot_safe
    provider = make_provider(cfg, demo=args.demo)
    spot = None if args.demo else get_spot_safe(provider)
    md = MarketData(Path(cfg["data"]["cache_dir"]).parent / "archive")
    m = deep.collect_market(md, spot)
    print("\n🌡 " + deep.market_text(m))
    if m["hot"] is not None:
        print("\n强势板块：\n" + m["hot"].to_string(index=False))
    if m["basis"] is not None:
        print("\n股指期货：\n" + m["basis"].to_string(index=False))
    st = md.status_table()
    print("\n数据来源：\n" + st.to_string(index=False))
    print("\n说明：温度分是经验加权，尚未经过历史检验；每天数据已归档，积累后可以检验。")


def cmd_auction(args, cfg):
    from quant import auction
    from quant.screener import get_spot_safe
    from quant.timeutil import now_cn as _now
    provider = make_provider(cfg, demo=args.demo)
    out_dir = Path(cfg["output_dir"])
    picks_dir = out_dir / "tail_picks"
    archive = Path(cfg["data"]["cache_dir"]).parent / "archive"
    today = _now().strftime("%Y%m%d")
    days = sorted({f.stem.split("_")[-1] for f in picks_dir.glob("picks_*.csv")}
                  | {p.name for p in archive.glob("2*") if p.is_dir()})
    prev = args.day or next((d for d in reversed(days) if d < today), None)
    if not prev:
        raise SystemExit("没有找到之前的选股记录，请先在交易日 14:30 后运行 python3 main.py tail")
    spot = get_spot_safe(provider)
    if spot is None:
        raise SystemExit("获取竞价行情失败（东方财富和新浪都不可用），请稍后重试。")
    codes = set()
    for f in picks_dir.glob(f"picks_*_{prev}.csv"):
        codes |= set(pd.read_csv(f, dtype={"代码": str})["代码"])
    hist = provider.daily_batch(codes, max_workers=int(cfg["data"]["max_workers"])) if codes else {}
    # 只用到选股日为止的日线，取“昨日成交额”
    hist = {c: df[df.index <= pd.Timestamp(prev)] for c, df in hist.items()}
    res = auction.run(spot, archive, prev, picks_dir, hist, universe=cfg["universe"])
    print(f"\n⏰ 集合竞价验证（选股日 {prev}）\n")
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            if v.empty:
                continue
            print(f"【{k}】")
            if k.startswith("昨日") and len(v) > 15:
                flag = [c for c in ("弱转强", "继续核按钮") if c in v.columns]
                print(f"  共 {len(v)} 只，平均竞价涨幅 {v['竞价涨幅%'].mean():+.2f}%"
                      + (f"，{flag[0]} {int(v[flag[0]].sum())} 只" if flag else ""))
                v = v.sort_values("竞价涨幅%", ascending=False).head(10)
            print(v.to_string(index=False) + "\n")
        elif v is not None:
            print(f"{k}: {v}")
    print("说明：竞价额/昨日成交额 ≥5% 视为竞价放量。高开放量也可能高开低走，请结合开盘后 5–15 分钟走势判断。")
    print(DISCLAIMER)


def cmd_check(args, cfg):
    """逐个测试数据源是否可用。"""
    import time as _t
    from datetime import date, timedelta
    from quant.data import SOURCE_LABELS
    provider = make_provider(cfg)
    end = date.today()
    start = end - timedelta(days=30)
    print("测试日线数据源（以 600519 为例）：")
    for src in SOURCE_LABELS:
        t0 = _t.time()
        try:
            df = provider._fetch(src, "600519", start, end)
            last = df.index[-1].strftime("%Y-%m-%d") if not df.empty else "无数据"
            print(f"  ✅ {SOURCE_LABELS[src]:<6} 可用，最新 {last}，耗时 {_t.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {SOURCE_LABELS[src]:<6} 不可用：{str(e)[:150]}")
    t0 = _t.time()
    try:
        n = len(provider.spot())
        print(f"  ✅ 实时行情（东方财富） 可用，{n} 只，耗时 {_t.time() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 实时行情（东方财富） 不可用：{str(e)[:150]}（不影响日线选股，只是用收盘数据代替）")
    print(f"\n当前顺序: {cfg['data'].get('sources')}，可在 config.yaml 的 data.sources 调整。")


def cmd_update(args, cfg):
    from quant import screener

    provider = make_provider(cfg, demo=args.demo)
    spot = screener.get_spot_safe(provider)
    codes = screener.resolve_universe(provider, cfg, args.universe, spot)
    print(f"更新 {len(codes)} 只股票日线…")
    hist = provider.daily_batch(codes, max_workers=int(cfg["data"]["max_workers"]),
                                progress=screener._progress)
    print(f"完成：成功 {len(hist)} / {len(codes)}，缓存目录 {cfg['data']['cache_dir']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A股量化工具：选股筛选 + 实时信号提醒",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("-c", "--config", help="配置文件路径（默认 config.yaml）")
    p.add_argument("--demo", action="store_true", help="使用离线模拟数据（无需联网）")
    p.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("strategies", help="列出内置策略")

    s = sub.add_parser("screen", help="选股筛选")
    s.add_argument("-s", "--strategy", default="all",
                   help="策略名，逗号分隔；all=全部技术策略；multi_factor=多因子")
    s.add_argument("-u", "--universe", default="hs300",
                   help="股票池: hs300/zz500/zz1000/sz50/all/watchlist/index:000905/代码列表")
    s.add_argument("-l", "--lookback", type=int, default=1, help="信号回看K线数（默认 1=仅今天）")
    s.add_argument("-d", "--direction", choices=["buy", "sell", "both"], default="buy")
    s.add_argument("--top", type=int, default=30, help="多因子取前 N")
    s.add_argument("--report", action="store_true", help="生成 HTML 图表报告")
    s.add_argument("--charts", type=int, default=20, help="报告中最多绘制多少只个股K线")
    s.add_argument("--no-open", action="store_true", help="生成报告后不自动打开浏览器")

    m = sub.add_parser("monitor", help="盘中实时监控自选股")
    m.add_argument("-s", "--strategy", help="覆盖配置中的监控策略，逗号分隔")
    m.add_argument("--codes", help="临时指定监控代码，逗号分隔（覆盖 watchlist）")
    m.add_argument("-i", "--interval", type=int, help="刷新间隔秒数")
    m.add_argument("--once", action="store_true", help="只运行一轮")
    m.add_argument("--force", action="store_true", help="非交易时间也运行")
    m.add_argument("--no-notify", action="store_true", help="只在终端输出，不发系统通知/webhook")

    c = sub.add_parser("chart", help="生成个股K线 + 指标图")
    c.add_argument("codes", nargs="+", help="股票代码，可多个")
    c.add_argument("-s", "--strategy", help="叠加的策略指标，逗号分隔（默认全部技术策略）")
    c.add_argument("--no-open", action="store_true")

    t = sub.add_parser("tail", help="尾盘选股：14:30 筛选次日上涨概率较高的股票")
    t.add_argument("--all-models", dest="verbose_models", action="store_true",
                   help="同时打印各单模型的名单（默认只打印短线五只）")
    t.add_argument("-m", "--model", choices=["short", "rule", "pattern", "trend", "ml"], default="short",
                   help="主模型：rule 反转规则（默认）/ pattern 尾盘回落型 / ml 机器学习；三个都会计算并复盘")
    t.add_argument("--backtest", action="store_true", help="用历史数据回测（两个模型对比）")
    t.add_argument("--review", action="store_true", help="只复盘之前的选股结果")
    t.add_argument("-u", "--universe", default=None,
                   help="股票池（默认读 config.yaml 的 tail.universe，初始为 zz500）")
    t.add_argument("--days", type=int, default=250, help="回测最近多少个交易日")
    t.add_argument("--charts", type=int, default=10)
    t.add_argument("--no-open", action="store_true")
    t.add_argument("--no-notify", action="store_true")
    t.add_argument("--fast", action="store_true", help="跳过扩展数据评估（只跑三个模型，更快）")

    sub.add_parser("check", help="测试各数据源是否可用")
    sub.add_parser("market", help="市场情绪温度：涨跌停、封板率、连板、期货升贴水、主线板块")
    a = sub.add_parser("auction", help="次日盘前验证（9:25 后运行）：竞价高低开、量能、弱转强、核按钮")
    a.add_argument("--day", help="要验证的选股日 YYYYMMDD（默认最近一次）")

    u = sub.add_parser("update", help="预下载/更新日线缓存")
    u.add_argument("-u", "--universe", default="watchlist")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    {"strategies": cmd_strategies, "screen": cmd_screen, "monitor": cmd_monitor,
     "chart": cmd_chart, "update": cmd_update, "tail": cmd_tail,
     "check": cmd_check, "market": cmd_market, "auction": cmd_auction}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
