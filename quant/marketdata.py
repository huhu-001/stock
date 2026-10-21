"""扩展数据层：尾盘分时、逐笔大单、资金流、涨停/炸板/跌停池、龙虎榜、指数、股指期货、板块、公告事件。

每个接口优先用东方财富，失败时尽量换新浪/腾讯；拿不到就返回 None，调用方据此降级（不中断主流程）。
每天取到的数据都会存一份到 data/archive/YYYYMMDD/，积累下来之后才能做回测。
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data import bypass_proxy_for_cn_sources, market_prefix, norm_code
from .timeutil import now_cn

log = logging.getLogger(__name__)

INDEXES = {"sh000001": "上证指数", "sz399001": "深证成指", "sz399006": "创业板指",
           "sh000300": "沪深300", "sh000905": "中证500", "sh000852": "中证1000"}
FUTURES = {"IF0": ("sh000300", "沪深300期货"), "IC0": ("sh000905", "中证500期货"),
           "IM0": ("sh000852", "中证1000期货")}


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


class MarketData:
    def __init__(self, archive_dir: str | Path, ak=None, retries: int = 2,
                 big_order_amount: float = 200000):
        if ak is None:
            bypass_proxy_for_cn_sources()
            import akshare as ak  # noqa: PLC0415
        self.ak = ak
        self.archive_root = Path(archive_dir)
        self.retries = retries
        self.big_order_amount = big_order_amount     # 大单阈值（元）
        self.status: dict[str, str] = {}             # 各数据项来源 / 失败原因，用于报告

    # ------------------------------------------------------------------ 工具
    def _call(self, name: str, *args, **kwargs):
        fn = getattr(self.ak, name, None)
        if fn is None:
            raise RuntimeError(f"当前 akshare 版本没有 {name}，请 pip3 install -U akshare")
        err = None
        for i in range(self.retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(0.8 * (i + 1))
        raise RuntimeError(f"{name}: {str(err)[:120]}")

    def _try(self, key: str, attempts: list):
        """attempts: [(来源名, 无参函数)]，依次尝试，返回第一个非空结果。"""
        errs = []
        for src, fn in attempts:
            try:
                df = fn()
                if df is not None and len(df):
                    self.status[key] = src
                    return df
                errs.append(f"{src}:空")
            except Exception as e:  # noqa: BLE001
                errs.append(f"{src}:{str(e)[:60]}")
        self.status[key] = "不可用（" + "；".join(errs) + "）"
        return None

    def archive(self, name: str, df: pd.DataFrame | None, day: str | None = None):
        if df is None or (hasattr(df, "empty") and df.empty):
            return
        d = self.archive_root / (day or now_cn().strftime("%Y%m%d"))
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f"{name}.csv", index=False, encoding="utf-8-sig")

    def load_archive(self, name: str, day: str) -> pd.DataFrame | None:
        p = self.archive_root / day / f"{name}.csv"
        if p.exists():
            return pd.read_csv(p, dtype={"代码": str, "code": str})
        return None

    # ------------------------------------------------------------ 全市场快照
    def spot_sina(self) -> pd.DataFrame:
        """新浪全市场快照（东方财富不可用时的备用，约 1 分钟）。统一为英文列，成交量=手。"""
        df = self._call("stock_zh_a_spot")
        df = df.rename(columns={"代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_chg",
                                "今开": "open", "最高": "high", "最低": "low", "昨收": "prev_close",
                                "成交量": "volume", "成交额": "amount"})
        df["code"] = df["code"].map(norm_code)
        df = _num(df, ["price", "pct_chg", "open", "high", "low", "prev_close", "volume", "amount"])
        df["volume"] = df["volume"] / 100
        return df[[c for c in ("code", "name", "price", "pct_chg", "open", "high", "low",
                               "prev_close", "volume", "amount") if c in df.columns]]

    # ---------------------------------------------------------- 1. 尾盘分时
    def minute_bars(self, code: str, day: date | None = None) -> pd.DataFrame | None:
        """当日 1 分钟K线：time, open, close, high, low, volume(手), amount(元), vwap。"""
        code = norm_code(code)
        day = day or now_cn().date()
        ds = day.strftime("%Y-%m-%d")

        def em():
            df = self._call("stock_zh_a_hist_min_em", symbol=code, start_date=f"{ds} 09:30:00",
                            end_date=f"{ds} 15:00:00", period="1", adjust="")
            df = df.rename(columns={"时间": "time", "开盘": "open", "收盘": "close", "最高": "high",
                                    "最低": "low", "成交量": "volume", "成交额": "amount",
                                    "均价": "vwap"})
            return df

        def sina():
            df = self._call("stock_zh_a_minute", symbol=market_prefix(code) + code, period="1")
            df = df.rename(columns={"day": "time"})
            df = df[df["time"].astype(str).str.startswith(ds)].copy()
            df = _num(df, ["volume"])
            df["volume"] = df["volume"] / 100       # 股 → 手
            return df

        df = self._try(f"分时:{code}", [("东方财富", em), ("新浪", sina)])
        if df is None:
            return None
        df = _num(df.copy(), ["open", "close", "high", "low", "volume", "amount", "vwap"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        if "amount" not in df or df["amount"].isna().all():
            df["amount"] = df["close"] * df["volume"] * 100
        cum_amt, cum_vol = df["amount"].cumsum(), (df["volume"] * 100).cumsum()
        if "vwap" not in df or df["vwap"].isna().all() or (df["vwap"] <= 0).all():
            df["vwap"] = cum_amt / cum_vol.replace(0, np.nan)
        return df

    # ---------------------------------------------------------- 逐笔大单
    def ticks(self, code: str) -> pd.DataFrame | None:
        """当日逐笔：time, price, lots(手), amount(元), side(买盘/卖盘/中性盘)。"""
        code = norm_code(code)

        def em():
            df = self._call("stock_intraday_em", symbol=code)
            df = df.rename(columns={"时间": "time", "成交价": "price", "手数": "lots",
                                    "买卖盘性质": "side"})
            df = _num(df, ["price", "lots"])
            df["amount"] = df["price"] * df["lots"] * 100
            return df

        def tx():
            df = self._call("stock_zh_a_tick_tx_js", symbol=market_prefix(code) + code)
            df = df.rename(columns={"成交时间": "time", "成交价格": "price", "成交量": "lots",
                                    "成交金额": "amount", "性质": "side"})
            return _num(df, ["price", "lots", "amount"])

        return self._try(f"逐笔:{code}", [("东方财富", em), ("腾讯", tx)])

    def big_orders(self, code: str) -> dict | None:
        """大单（单笔成交额 ≥ 阈值）主动买卖统计。"""
        t = self.ticks(code)
        if t is None or "amount" not in t:
            return None
        big = t[t["amount"] >= self.big_order_amount]
        buy = big.loc[big["side"].astype(str).str.contains("买"), "amount"].sum()
        sell = big.loc[big["side"].astype(str).str.contains("卖"), "amount"].sum()
        total = t["amount"].sum()
        return {"大单买入": buy, "大单卖出": sell, "大单净额": buy - sell,
                "大单主动买占比": buy / (buy + sell) if buy + sell else np.nan,
                "大单成交占比": big["amount"].sum() / total if total else np.nan}

    # ---------------------------------------------------------- 资金流向
    def fund_flow(self, code: str) -> dict | None:
        code = norm_code(code)
        mk = market_prefix(code)

        def em():
            return self._call("stock_individual_fund_flow", stock=code, market=mk)

        df = self._try(f"资金流:{code}", [("东方财富", em)])
        if df is None:
            return None
        r = df.iloc[-1]
        g = lambda k: pd.to_numeric(r.get(k), errors="coerce")  # noqa: E731
        return {"主力净流入": g("主力净流入-净额"), "主力净占比%": g("主力净流入-净占比"),
                "超大单净流入": g("超大单净流入-净额"), "大单净流入": g("大单净流入-净额"),
                "日期": str(r.get("日期", ""))}

    # ---------------------------------------------------------- 3. 情绪：涨停池等
    def zt_pool(self, day: str) -> pd.DataFrame | None:
        return self._try("涨停池", [("东方财富", lambda: self._call("stock_zt_pool_em", date=day))])

    def zt_prev_pool(self, day: str) -> pd.DataFrame | None:
        """昨日涨停股今日表现。"""
        return self._try("昨日涨停", [("东方财富",
                                      lambda: self._call("stock_zt_pool_previous_em", date=day))])

    def zb_pool(self, day: str) -> pd.DataFrame | None:
        return self._try("炸板池", [("东方财富", lambda: self._call("stock_zt_pool_zbgc_em", date=day))])

    def dt_pool(self, day: str) -> pd.DataFrame | None:
        return self._try("跌停池", [("东方财富", lambda: self._call("stock_zt_pool_dtgc_em", date=day))])

    # ---------------------------------------------------------- 4. 龙虎榜
    def lhb(self, day: str) -> pd.DataFrame | None:
        """当日龙虎榜：代码 名称 涨跌幅 龙虎榜净买额 上榜原因 …"""
        def em():
            df = self._call("stock_lhb_detail_em", start_date=day, end_date=day)
            df["代码"] = df["代码"].map(norm_code)
            return df

        def sina():
            df = self._call("stock_lhb_detail_daily_sina", date=day)
            df = df.rename(columns={"股票代码": "代码", "股票名称": "名称", "指标": "上榜原因"})
            df["代码"] = df["代码"].map(norm_code)
            return df

        return self._try("龙虎榜", [("东方财富", em), ("新浪", sina)])

    def lhb_seats(self, code: str, day: str) -> pd.DataFrame | None:
        """个股龙虎榜席位（买入前五 + 卖出前五），列：方向 营业部 买入 卖出 净额。"""
        code = norm_code(code)

        def em():
            out = []
            for flag in ("买入", "卖出"):
                df = self._call("stock_lhb_stock_detail_em", symbol=code, date=day, flag=flag)
                df = df.rename(columns={"交易营业部名称": "营业部", "买入金额": "买入",
                                        "卖出金额": "卖出"})
                df["方向"] = flag
                out.append(df)
            return pd.concat(out, ignore_index=True)

        df = self._try(f"龙虎榜席位:{code}", [("东方财富", em)])
        if df is not None:
            df = _num(df, ["买入", "卖出", "净额"])
        return df

    # ---------------------------------------------------------- 5. 指数 / 期货 / 板块
    def index_spot(self) -> pd.DataFrame | None:
        def sina():
            df = self._call("stock_zh_index_spot_sina")
            df = df[df["代码"].isin(INDEXES)].copy()
            return _num(df, ["最新价", "涨跌幅", "昨收", "今开", "最高", "最低"])
        return self._try("指数", [("新浪", sina)])

    def index_minute(self, symbol: str = "sh000905") -> pd.DataFrame | None:
        def sina():
            df = self._call("stock_zh_a_minute", symbol=symbol, period="1")
            df = df.rename(columns={"day": "time"})
            df["time"] = pd.to_datetime(df["time"])
            last_day = df["time"].dt.date.max()
            return _num(df[df["time"].dt.date == last_day].copy(), ["close", "volume"])
        return self._try(f"指数分时:{symbol}", [("新浪", sina)])

    def futures_basis(self, idx: pd.DataFrame | None) -> pd.DataFrame | None:
        """股指期货主力合约相对现货指数的升贴水（%）。"""
        def sina():
            df = self._call("futures_zh_spot", symbol=",".join(FUTURES), market="FF", adjust="0")
            return _num(df, ["current_price"])
        fut = self._try("股指期货", [("新浪", sina)])
        if fut is None or idx is None:
            return None
        rows = []
        prices = fut["current_price"].tolist()
        for (fcode, (icode, label)), fp in zip(FUTURES.items(), prices):
            ip = idx.loc[idx["代码"] == icode, "最新价"]
            if len(ip) and pd.notna(fp) and fp > 0:
                rows.append({"品种": label, "期货": fp, "现货": float(ip.iloc[0]),
                             "升贴水%": round((fp / float(ip.iloc[0]) - 1) * 100, 2)})
        return pd.DataFrame(rows) if rows else None

    def sectors(self) -> pd.DataFrame | None:
        """行业板块涨跌：板块 涨跌幅 上涨家数 下跌家数 领涨股票。"""
        def em():
            df = self._call("stock_board_industry_name_em")
            df = df.rename(columns={"板块名称": "板块"})
            return _num(df, ["涨跌幅", "上涨家数", "下跌家数", "换手率"])

        def sina():
            df = self._call("stock_sector_spot", indicator="新浪行业")
            df = df.rename(columns={"股票名称": "领涨股票"})
            return _num(df, ["涨跌幅", "公司家数"])
        return self._try("板块", [("东方财富", em), ("新浪", sina)])

    def industry(self, code: str) -> str:
        """个股所属行业（东方财富个股资料）。"""
        code = norm_code(code)

        def em():
            df = self._call("stock_individual_info_em", symbol=code)
            v = df.loc[df["item"] == "行业", "value"]
            return pd.DataFrame({"行业": [str(v.iloc[0])]}) if len(v) else None
        df = self._try(f"行业:{code}", [("东方财富", em)])
        return "" if df is None else df["行业"].iloc[0]

    # ---------------------------------------------------------- 7. 消息与事件
    def notices(self, day: str) -> pd.DataFrame | None:
        def em():
            df = self._call("stock_notice_report", symbol="全部", date=day)
            df["代码"] = df["代码"].map(norm_code)
            return df
        return self._try("公告", [("东方财富", em)])

    def forecasts(self) -> pd.DataFrame | None:
        """最近一期业绩预告。"""
        today = now_cn().date()
        q = [date(today.year, 3, 31), date(today.year, 6, 30), date(today.year, 9, 30),
             date(today.year, 12, 31), date(today.year - 1, 12, 31)]
        cands = sorted([d for d in q if d <= today + timedelta(days=100)], reverse=True)[:2]

        def em(d):
            df = self._call("stock_yjyg_em", date=d.strftime("%Y%m%d"))
            df["代码"] = df["股票代码"].map(norm_code)
            return df
        return self._try("业绩预告", [(f"东方财富{d:%Y%m%d}", lambda d=d: em(d)) for d in cands])

    def unlocks(self, days: int = 30) -> pd.DataFrame | None:
        """未来 N 天限售解禁。"""
        today = now_cn().date()

        def em():
            df = self._call("stock_restricted_release_detail_em",
                            start_date=today.strftime("%Y%m%d"),
                            end_date=(today + timedelta(days=days)).strftime("%Y%m%d"))
            df["代码"] = df["股票代码"].map(norm_code)
            return _num(df, ["占解禁前流通市值比例", "实际解禁市值"])
        return self._try("解禁", [("东方财富", em)])

    def reductions(self, days: int = 30) -> pd.DataFrame | None:
        """近 N 天股东减持公告。"""
        def em():
            df = self._call("stock_ggcg_em", symbol="股东减持")
            df["代码"] = df["代码"].map(norm_code)
            df["公告日"] = pd.to_datetime(df["公告日"], errors="coerce")
            cutoff = pd.Timestamp(now_cn().date() - timedelta(days=days))
            return df[df["公告日"] >= cutoff]
        return self._try("减持", [("东方财富", em)])

    def status_table(self) -> pd.DataFrame:
        rows = [{"数据": k, "来源/状态": v} for k, v in self.status.items() if ":" not in k]
        per_stock = {}
        for k, v in self.status.items():
            if ":" in k:
                kind = k.split(":")[0]
                ok = not v.startswith("不可用")
                a, b = per_stock.get(kind, (0, 0))
                per_stock[kind] = (a + ok, b + 1)
        rows += [{"数据": k, "来源/状态": f"成功 {a}/{b} 只"} for k, (a, b) in per_stock.items()]
        return pd.DataFrame(rows)


def last_trade_day(day: date, archive_root: Path | None = None) -> date:
    """上一个交易日（周末跳过；节假日用归档目录或日线判断不了时近似）。"""
    d = day - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def today_str() -> str:
    return now_cn().strftime("%Y%m%d")


def parse_day(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()
