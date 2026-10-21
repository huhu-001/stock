from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

BUY, SELL, HOLD = 1, -1, 0


@dataclass
class Signal:
    code: str
    strategy: str
    direction: int          # 1 买入 / -1 卖出
    date: pd.Timestamp
    price: float
    reason: str
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "买入" if self.direction == BUY else "卖出"


class Strategy:
    """时序（单只股票）策略基类。

    子类实现 compute(df) -> df，需增加：
      signal 列（1/-1/0）
      reason 列（信号说明，可选）
    以及任意指标列（会用于图表）。
    """

    name = "base"
    label = "基础策略"
    description = ""
    defaults: dict = {}
    # 图表：叠加在K线上的指标列 / 单独副图的指标列
    overlay_cols: list[str] = []
    panel_cols: list[str] = []
    min_bars = 30

    def __init__(self, **params):
        self.params = {**self.defaults, **{k: v for k, v in params.items() if v is not None}}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < self.min_bars:
            return pd.DataFrame()
        out = self.compute(df.copy())
        if "reason" not in out:
            out["reason"] = ""
        out["signal"] = out["signal"].fillna(0).astype(int)
        return out

    def state(self, row) -> str:
        """当前指标状态的简短描述（用于监控面板）。"""
        return ""

    def latest_signals(self, code: str, df: pd.DataFrame, lookback: int = 1) -> list[Signal]:
        """最近 lookback 根K线内出现的信号。"""
        res = self.run(df)
        if res.empty:
            return []
        tail = res.iloc[-lookback:]
        sigs = []
        for ts, row in tail[tail["signal"] != 0].iterrows():
            sigs.append(Signal(code=code, strategy=self.name, direction=int(row["signal"]),
                               date=ts, price=float(row["close"]), reason=str(row["reason"])))
        return sigs

    def __repr__(self):
        return f"{self.name}({self.params})"
