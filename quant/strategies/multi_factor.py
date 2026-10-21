"""多因子打分选股（横截面策略）。

对全市场实时快照中的因子做去极值 + 标准化，按权重与方向合成总分，取排名靠前的股票。
默认因子：60日动量、年初至今涨幅、动态PE、PB、换手率、总市值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FACTORS = {
    "chg_60d": {"weight": 0.25, "direction": 1},
    "chg_ytd": {"weight": 0.10, "direction": 1},
    "pe": {"weight": 0.25, "direction": -1},
    "pb": {"weight": 0.15, "direction": -1},
    "turnover": {"weight": 0.10, "direction": 1},
    "total_mv": {"weight": 0.15, "direction": -1},
}

FACTOR_LABELS = {"chg_60d": "60日涨幅%", "chg_ytd": "年初至今%", "pe": "PE(动)", "pb": "PB",
                 "turnover": "换手率%", "total_mv": "总市值"}


def _winsorize_zscore(s: pd.Series, k: float = 3.0) -> pd.Series:
    """MAD 去极值后 z-score 标准化。"""
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad and not np.isnan(mad):
        s = s.clip(med - k * mad, med + k * mad)
    std = s.std()
    return (s - s.mean()) / std if std else s * 0


class MultiFactorStrategy:
    name = "multi_factor"
    label = "多因子打分选股"
    description = "动量 + 估值 + 活跃度 + 市值多因子加权打分，全市场排名取前 N。"
    cross_sectional = True

    def __init__(self, factors: dict | None = None, max_pe: float = 100, exclude_loss: bool = True,
                 **_):
        self.factors = factors or DEFAULT_FACTORS
        self.max_pe = max_pe
        self.exclude_loss = exclude_loss

    def score(self, spot: pd.DataFrame) -> pd.DataFrame:
        df = spot.copy()
        if self.exclude_loss and "pe" in df:
            df = df[df["pe"] > 0]
        if self.max_pe and "pe" in df:
            df = df[df["pe"] <= self.max_pe]
        if "pb" in df:
            df = df[df["pb"] > 0]
        used = [f for f in self.factors if f in df.columns]
        df = df.dropna(subset=used)
        if df.empty:
            return df
        total_w = sum(abs(self.factors[f]["weight"]) for f in used) or 1
        df["score"] = 0.0
        for f in used:
            cfg = self.factors[f]
            x = df[f]
            if f in ("total_mv", "float_mv"):
                x = np.log(x)  # 市值取对数
            z = _winsorize_zscore(x) * cfg.get("direction", 1)
            df[f"z_{f}"] = z
            df["score"] += z * cfg["weight"] / total_w
        df["score"] = df["score"].round(4)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df
