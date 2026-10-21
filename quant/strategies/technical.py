"""技术面时序策略：双均线、MACD、布林带+RSI 均值回归。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import BUY, SELL, Strategy


class MACrossStrategy(Strategy):
    name = "ma_cross"
    label = "双均线交叉"
    description = "短期均线上穿长期均线买入，下穿卖出；可选成交量放大确认。"
    defaults = {"fast": 5, "slow": 20, "volume_confirm": True}

    @property
    def overlay_cols(self):
        return [f"ma{self.params['fast']}", f"ma{self.params['slow']}"]

    @property
    def min_bars(self):
        return int(self.params["slow"]) + 5

    def compute(self, df):
        f, s = int(self.params["fast"]), int(self.params["slow"])
        fa, sl = f"ma{f}", f"ma{s}"
        df[fa] = ind.sma(df["close"], f)
        df[sl] = ind.sma(df["close"], s)
        df["vol_ma20"] = ind.sma(df["volume"], 20)
        up = ind.cross_up(df[fa], df[sl])
        dn = ind.cross_down(df[fa], df[sl])
        if self.params["volume_confirm"]:
            up &= df["volume"] > df["vol_ma20"]
        df["signal"] = np.select([up, dn], [BUY, SELL], 0)
        df["reason"] = np.select(
            [up, dn],
            [f"MA{f} 上穿 MA{s}" + ("（放量）" if self.params["volume_confirm"] else ""),
             f"MA{f} 下穿 MA{s}"], "")
        return df

    def state(self, row) -> str:
        f, s = f"ma{self.params['fast']}", f"ma{self.params['slow']}"
        if pd.isna(row.get(s)):
            return "-"
        return "均线多头" if row[f] > row[s] else "均线空头"


class MACDStrategy(Strategy):
    name = "macd"
    label = "MACD 金叉死叉"
    description = "DIF 上穿 DEA 买入，下穿卖出；可限定只在零轴上方的金叉。"
    defaults = {"fast": 12, "slow": 26, "signal": 9, "above_zero": False}
    panel_cols = ["dif", "dea", "macd"]
    min_bars = 40

    def compute(self, df):
        p = self.params
        df["dif"], df["dea"], df["macd"] = ind.macd(df["close"], int(p["fast"]), int(p["slow"]),
                                                    int(p["signal"]))
        up = ind.cross_up(df["dif"], df["dea"])
        dn = ind.cross_down(df["dif"], df["dea"])
        if p["above_zero"]:
            up &= df["dif"] > 0
        df["signal"] = np.select([up, dn], [BUY, SELL], 0)
        zero = np.where(df["dif"] > 0, "零轴上方", "零轴下方")
        df["reason"] = np.where(up, pd.Series("MACD 金叉·", index=df.index) + zero,
                                np.where(dn, pd.Series("MACD 死叉·", index=df.index) + zero, ""))
        return df

    def state(self, row) -> str:
        if pd.isna(row.get("dif")):
            return "-"
        return f"{'DIF>DEA' if row['dif'] > row['dea'] else 'DIF<DEA'} 柱{row['macd']:+.2f}"


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    label = "布林带+RSI 均值回归"
    description = ("收盘跌破布林下轨且 RSI 超卖后重新站回下轨时买入；"
                   "突破上轨且 RSI 超买后跌回上轨时卖出。")
    defaults = {"boll_period": 20, "boll_std": 2.0, "rsi_period": 14, "rsi_low": 30, "rsi_high": 70}
    overlay_cols = ["boll_mid", "boll_up", "boll_low"]
    panel_cols = ["rsi"]
    min_bars = 30

    def compute(self, df):
        p = self.params
        df["boll_mid"], df["boll_up"], df["boll_low"] = ind.bollinger(
            df["close"], int(p["boll_period"]), float(p["boll_std"]))
        df["rsi"] = ind.rsi(df["close"], int(p["rsi_period"]))
        # 近 3 根内出现过超卖/超买
        oversold = ((df["close"] < df["boll_low"]) & (df["rsi"] < p["rsi_low"])) \
            .rolling(3, min_periods=1).max().shift().fillna(0).astype(bool)
        overbought = ((df["close"] > df["boll_up"]) & (df["rsi"] > p["rsi_high"])) \
            .rolling(3, min_periods=1).max().shift().fillna(0).astype(bool)
        buy = oversold & ind.cross_up(df["close"], df["boll_low"])
        sell = overbought & ind.cross_down(df["close"], df["boll_up"])
        df["signal"] = np.select([buy, sell], [BUY, SELL], 0)
        r = df["rsi"].round(1).astype(str)
        df["reason"] = np.where(buy, "超卖后站回布林下轨 RSI=" + r,
                                np.where(sell, "超买后跌回布林上轨 RSI=" + r, ""))
        return df

    def state(self, row) -> str:
        if pd.isna(row.get("rsi")) or pd.isna(row.get("boll_up")):
            return "-"
        pos = (row["close"] - row["boll_low"]) / (row["boll_up"] - row["boll_low"]) * 100 \
            if row["boll_up"] != row["boll_low"] else 50
        return f"RSI {row['rsi']:.0f} 带内{pos:.0f}%"
