"""盘中实时监控：定时刷新自选股行情，把最新价拼接成当日K线运行策略，出现信号即提醒。

注意：盘中信号基于未收盘的K线，收盘前可能消失，仅作提醒，不构成投资建议。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import strategies as S
from .data import BaseProvider
from .notifier import Notifier
from .timeutil import is_trading_time, now_cn, seconds_to_next_session

log = logging.getLogger(__name__)


def merge_live_bar(hist: pd.DataFrame, quote: pd.Series, today: pd.Timestamp) -> pd.DataFrame:
    """用实时快照生成/替换当日K线。"""
    if hist is None or hist.empty or pd.isna(quote.get("price")) or quote.get("price", 0) <= 0:
        return hist
    bar = {"open": quote.get("open"), "high": quote.get("high"), "low": quote.get("low"),
           "close": quote.get("price"), "volume": quote.get("volume"),
           "amount": quote.get("amount"), "pct_chg": quote.get("pct_chg"),
           "turnover": quote.get("turnover")}
    bar = {k: float(v) for k, v in bar.items() if k in hist.columns and pd.notna(v)}
    df = hist.copy()
    if df.index[-1].normalize() == today:
        for k, v in bar.items():
            df.at[df.index[-1], k] = v
    else:
        new = pd.DataFrame([bar], index=pd.DatetimeIndex([today], name=df.index.name))
        df = pd.concat([df, new])
    return df


class Monitor:
    def __init__(self, provider: BaseProvider, cfg: dict, notifier: Notifier,
                 strategy_names: list[str] | None = None, interval: int | None = None):
        self.p = provider
        self.cfg = cfg
        self.n = notifier
        mcfg = cfg["monitor"]
        self.interval = int(interval or mcfg.get("interval", 60))
        names = strategy_names or mcfg.get("strategies") or list(S.TECHNICAL)
        self.strats = [S.build(n, cfg) for n in names if n in S.TECHNICAL]
        self.watch = {w["code"]: w for w in cfg["watchlist"]}
        self.hist: dict[str, pd.DataFrame] = {}
        self.state_path = Path(cfg["output_dir"]) / "monitor_state.json"
        self.sent: set[str] = self._load_state()

    # -- 去重状态（按天持久化，重启不重复提醒） ----------------------------
    def _load_state(self) -> set[str]:
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            if d.get("date") == now_cn().strftime("%Y-%m-%d"):
                return set(d.get("sent", []))
        except Exception:  # noqa: BLE001
            pass
        return set()

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(
            {"date": now_cn().strftime("%Y-%m-%d"), "sent": sorted(self.sent)},
            ensure_ascii=False), encoding="utf-8")

    def _alert(self, key: str, title: str, body: str, level: str):
        if key in self.sent:
            return False
        self.sent.add(key)
        self.n.send(title, body, level)
        self._save_state()
        return True

    # ------------------------------------------------------------------
    def prepare(self):
        if not self.watch:
            raise SystemExit("config.yaml 中 watchlist 为空，请先添加自选股。")
        print(f"加载 {len(self.watch)} 只自选股历史数据…")
        self.hist = self.p.daily_batch(self.watch.keys(),
                                       max_workers=int(self.cfg["data"]["max_workers"]))
        missing = set(self.watch) - set(self.hist)
        if missing:
            print(f"  ⚠ 以下代码未获取到数据: {', '.join(sorted(missing))}")

    def tick(self) -> pd.DataFrame:
        """执行一轮检查，返回监控面板 DataFrame。"""
        spot = self.p.spot()
        spot = spot[spot["code"].isin(self.watch)].set_index("code")
        today = pd.Timestamp(now_cn().date())
        day = today.strftime("%Y-%m-%d")
        rows = []
        for code, rule in self.watch.items():
            if code not in spot.index:
                continue
            q = spot.loc[code]
            name, price, pct = q.get("name", ""), float(q.get("price", float("nan"))), \
                float(q.get("pct_chg", float("nan")))
            row = {"代码": code, "名称": name, "最新价": round(price, 2), "涨跌%": round(pct, 2)}
            tag = f"{name}({code})"

            # 1) 价格提醒
            checks = [("above", price >= rule["above"] if "above" in rule else False,
                       f"突破 {rule.get('above')}", "buy"),
                      ("below", price <= rule["below"] if "below" in rule else False,
                       f"跌破 {rule.get('below')}", "sell"),
                      ("pct_up", pct >= rule["pct_up"] if "pct_up" in rule else False,
                       f"涨幅达 {rule.get('pct_up')}%", "buy"),
                      ("pct_down", pct <= rule["pct_down"] if "pct_down" in rule else False,
                       f"跌幅达 {rule.get('pct_down')}%", "sell")]
            for k, hit, desc, lvl in checks:
                if hit:
                    self._alert(f"{day}|{code}|price|{k}", f"价格提醒 {tag}",
                                f"{desc}，现价 {price:.2f}（{pct:+.2f}%）", lvl)

            # 2) 策略信号
            hist = self.hist.get(code)
            if hist is not None and not hist.empty:
                live = merge_live_bar(hist, q, today)
                for st in self.strats:
                    res = st.run(live)
                    if res.empty:
                        continue
                    last = res.iloc[-1]
                    row[st.label] = st.state(last)
                    sig = int(last["signal"])
                    if sig:
                        lbl = "买入" if sig == S.BUY else "卖出"
                        bar_day = res.index[-1].strftime("%Y-%m-%d")
                        self._alert(f"{bar_day}|{code}|{st.name}|{sig}",
                                    f"{lbl}信号 {tag}",
                                    f"[{st.label}] {last['reason']}，现价 {price:.2f}（{pct:+.2f}%）",
                                    "buy" if sig == S.BUY else "sell")
            rows.append(row)
        return pd.DataFrame(rows)

    def run(self, once: bool = False, force: bool = False):
        self.prepare()
        print(f"开始监控，刷新间隔 {self.interval}s，策略: "
              f"{', '.join(s.label for s in self.strats)}。Ctrl+C 退出。\n")
        while True:
            if not force and not is_trading_time():
                wait = seconds_to_next_session()
                if once:
                    print("当前非交易时间（可加 --force 强制运行）。")
                    return
                print(f"[{now_cn():%H:%M:%S}] 非交易时间，{wait / 60:.0f} 分钟后开盘再刷新…",
                      flush=True)
                time.sleep(min(wait, 1800))
                continue
            try:
                board = self.tick()
                print(f"\n—— {now_cn():%Y-%m-%d %H:%M:%S} ——")
                if not board.empty:
                    print(board.to_string(index=False))
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("本轮刷新失败: %s", e)
                print(f"[{now_cn():%H:%M:%S}] 刷新失败: {e}")
            if once:
                return
            time.sleep(self.interval)
