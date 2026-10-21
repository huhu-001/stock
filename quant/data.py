"""数据层：AKShare 行情获取 + 本地缓存；另提供离线演示用的 MockProvider。

所有 provider 输出统一的英文列名：
  日线 daily():  index=date, open high low close volume amount pct_chg turnover
  实时 spot():   code name price pct_chg open high low prev_close volume amount
                 turnover pe pb total_mv float_mv chg_60d chg_ytd
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .timeutil import last_close_time, now_cn

log = logging.getLogger(__name__)

DAILY_COLS = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg", "换手率": "turnover",
}
SPOT_COLS = {
    "代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_chg",
    "今开": "open", "最高": "high", "最低": "low", "昨收": "prev_close",
    "成交量": "volume", "成交额": "amount", "换手率": "turnover",
    "市盈率-动态": "pe", "市净率": "pb", "总市值": "total_mv", "流通市值": "float_mv",
    "60日涨跌幅": "chg_60d", "年初至今涨跌幅": "chg_ytd", "量比": "volume_ratio",
}
SPOT_NUMERIC = ["price", "pct_chg", "open", "high", "low", "prev_close", "volume", "amount",
                "turnover", "pe", "pb", "total_mv", "float_mv", "chg_60d", "chg_ytd",
                "volume_ratio"]


# 国内行情源域名：强制直连，不走系统代理/VPN（代理常导致 ProxyError）
DIRECT_DOMAINS = ["eastmoney.com", "sina.com.cn", "sinajs.cn", "sina.cn", "qq.com", "gtimg.cn", "csindex.com.cn", "sse.com.cn",
                  "szse.cn", "bse.cn", "10jqka.com.cn", "baostock.com", "127.0.0.1", "localhost"]


def bypass_proxy_for_cn_sources():
    cur = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    items = [x for x in cur.split(",") if x.strip()]
    items += [d for d in DIRECT_DOMAINS if d not in items]
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(items)


SOURCE_LABELS = {"eastmoney": "东方财富", "tencent": "腾讯", "sina": "新浪"}


def market_prefix(code: str) -> str:
    if code.startswith(("8", "4", "92")):
        return "bj"
    return "sh" if code.startswith(("6", "9")) else "sz"


def normalize_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """把腾讯/新浪日线统一成东方财富口径：成交量=手，成交额=元，换手率=%。"""
    df = raw.copy()
    if "date" in df.columns:
        df = df.reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] > 0]
    ratio = (df["amount"] / (df["volume"] * df["close"])).replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(ratio) and ratio > 0:
        # 让 成交额 / (成交量×100) ≈ 价格，即成交量单位为“手”；按 10 的整数次幂校正
        k = 10 ** round(np.log10(ratio / 100))
        df["volume"] = df["volume"] * k
    if "turnover" in df.columns and df["turnover"].notna().any():
        if df["turnover"].median() < 0.2:      # 小数口径 → 百分比
            df["turnover"] = df["turnover"] * 100
    else:
        df["turnover"] = np.nan
    df["pct_chg"] = df["close"].pct_change() * 100
    return df[["open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover"]].astype(float)


def norm_code(code) -> str:
    s = str(code).strip().lower()
    for p in ("sh", "sz", "bj", "."):
        s = s.replace(p, "")
    return s.zfill(6)


# ----------------------------------------------------------------------------
class BaseProvider:
    def stock_list(self) -> pd.DataFrame: ...
    def spot(self) -> pd.DataFrame: ...
    def daily(self, code: str, days: int | None = None) -> pd.DataFrame: ...
    def index_constituents(self, index_code: str) -> pd.DataFrame: ...

    def daily_batch(self, codes: Iterable[str], days: int | None = None, max_workers: int = 4,
                    progress: Callable[[int, int], None] | None = None) -> dict[str, pd.DataFrame]:
        codes = list(dict.fromkeys(norm_code(c) for c in codes))
        out: dict[str, pd.DataFrame] = {}
        errors: list[str] = []
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.daily, c, days): c for c in codes}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    df = fut.result()
                    if df is not None and not df.empty:
                        out[c] = df
                except Exception as e:  # noqa: BLE001
                    failed.append(c)
                    log.debug("获取 %s 日线失败: %s", c, e)
                if progress:
                    progress(i, len(codes))
        if failed and len(codes) > 1:
            # 第二轮：放慢速度逐只重试（数据源限流时并发请求容易被断开）
            print(f"  {len(failed)} 只首轮失败，放慢速度逐只重试…", flush=True)
            time.sleep(3)
            for j, c in enumerate(failed, 1):
                try:
                    df = self.daily(c, days)
                    if df is not None and not df.empty:
                        out[c] = df
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
                time.sleep(0.5)
                if j % 50 == 0 or j == len(failed):
                    print(f"  重试 {j}/{len(failed)}，累计成功 {len(out)}/{len(codes)}", flush=True)
        elif failed:
            errors.append("下载失败")
        rep_src = getattr(self, "source_report", lambda: "")()
        if rep_src:
            print(f"  本次下载来源：{rep_src}（其余读本地缓存）")
        if errors:
            print(f"  ⚠ {len(errors)}/{len(codes)} 只下载失败。示例错误: {errors[0][:200]}")
            if any("Proxy" in e for e in errors):
                print("  → 与代理/VPN 有关：请关闭代理，或在代理软件中把 eastmoney.com 设为直连后重试。")
            elif len(errors) > len(codes) * 0.3:
                print("  → 可能被数据源限流：把 config.yaml 中 data.max_workers 调成 1~2，稍后重试。")
        return out


# ----------------------------------------------------------------------------
class AKShareProvider(BaseProvider):
    def __init__(self, cache_dir: str, history_days: int = 400, adjust: str = "qfq",
                 retries: int = 3, bypass_proxy: bool = True,
                 sources: list | None = None):
        if bypass_proxy:
            bypass_proxy_for_cn_sources()
        try:
            import akshare as ak  # noqa: F401
        except ImportError as e:
            raise SystemExit("未安装 akshare，请先执行: pip install -r requirements.txt") from e
        self.ak = ak
        if bypass_proxy:
            bypass_proxy_for_cn_sources()
        self.cache = Path(cache_dir)
        (self.cache / "daily").mkdir(parents=True, exist_ok=True)
        self.history_days = history_days
        self.adjust = adjust or ""
        self.retries = retries
        self.sources = [x for x in (sources or ["eastmoney", "tencent", "sina"])
                        if x in SOURCE_LABELS]
        self._fails: dict = {}     # 各源连续失败次数，超过阈值本次运行不再尝试
        self._used: dict = {}      # 各源成功次数

    # -- helpers -------------------------------------------------------------
    def _call(self, fn, *args, **kwargs):
        err = None
        for i in range(self.retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(1.5 * (i + 1))
        raise RuntimeError(f"{getattr(fn, '__name__', fn)} 调用失败: {err}")

    def _fresh(self, path: Path, max_age: timedelta | None = None) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=now_cn().tzinfo)
        if max_age is not None:
            return now_cn() - mtime < max_age
        return mtime >= last_close_time()

    # -- API -----------------------------------------------------------------
    def stock_list(self) -> pd.DataFrame:
        path = self.cache / "stock_list.csv"
        if self._fresh(path, timedelta(days=1)):
            return pd.read_csv(path, dtype={"code": str})
        df = self._call(self.ak.stock_info_a_code_name)
        df = df.rename(columns={"code": "code", "name": "name"})[["code", "name"]]
        df["code"] = df["code"].map(norm_code)
        df.to_csv(path, index=False)
        return df

    def spot(self) -> pd.DataFrame:
        """全市场实时快照。东方财富失败时改用新浪（新浪较慢，约 1 分钟，且没有换手率/估值字段）。"""
        order = ["eastmoney", "sina"]
        if self.sources and self.sources[0] != "eastmoney":
            order = ["sina", "eastmoney"]
        errs = []
        for src in order:
            try:
                return self._spot_em() if src == "eastmoney" else self._spot_sina()
            except Exception as e:  # noqa: BLE001
                errs.append(f"{SOURCE_LABELS[src]}: {str(e)[:100]}")
        raise RuntimeError("；".join(errs))

    def _spot_sina(self) -> pd.DataFrame:
        print("  获取新浪全市场行情（约 1 分钟）…", flush=True)
        df = self._call(self.ak.stock_zh_a_spot)
        df = df.rename(columns={"代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_chg",
                                "今开": "open", "最高": "high", "最低": "low", "昨收": "prev_close",
                                "成交量": "volume", "成交额": "amount"})
        df["code"] = df["code"].map(norm_code)
        for c in SPOT_NUMERIC:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = df["volume"] / 100          # 股 → 手
        keep = [c for c in SPOT_COLS.values() if c in df.columns]
        return df[keep].reset_index(drop=True)

    def _spot_em(self) -> pd.DataFrame:
        df = self._call(self.ak.stock_zh_a_spot_em)
        df = df.rename(columns=SPOT_COLS)
        keep = [c for c in SPOT_COLS.values() if c in df.columns]
        df = df[keep].copy()
        df["code"] = df["code"].map(norm_code)
        for c in SPOT_NUMERIC:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.reset_index(drop=True)

    def daily(self, code: str, days: int | None = None) -> pd.DataFrame:
        code = norm_code(code)
        days = days or self.history_days
        path = self.cache / "daily" / f"{code}_{self.adjust or 'none'}.csv"
        if self._fresh(path):
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            if len(df) and df.index[0] <= pd.Timestamp(date.today() - timedelta(days=days - 30)):
                return df
        end = date.today()
        start = end - timedelta(days=days)
        errors = []
        for src in self.sources:
            if self._fails.get(src, 0) >= 8:       # 该源已连续失败多次，跳过
                continue
            try:
                df = self._fetch(src, code, start, end)
            except Exception as e:  # noqa: BLE001
                self._fails[src] = self._fails.get(src, 0) + 1
                errors.append(f"{SOURCE_LABELS[src]}: {str(e)[:120]}")
                continue
            self._fails[src] = 0
            if df is None or df.empty:
                errors.append(f"{SOURCE_LABELS[src]}: 无数据")
                continue
            self._used[src] = self._used.get(src, 0) + 1
            df.to_csv(path)
            return df
        raise RuntimeError("；".join(errors) or "所有数据源都不可用")

    def _fetch(self, src: str, code: str, start, end) -> pd.DataFrame:
        if src == "eastmoney":
            raw = self._call(self.ak.stock_zh_a_hist, symbol=code, period="daily",
                             start_date=start.strftime("%Y%m%d"),
                             end_date=end.strftime("%Y%m%d"), adjust=self.adjust)
            if raw is None or raw.empty:
                return pd.DataFrame()
            df = raw.rename(columns=DAILY_COLS)
            df = df[[c for c in DAILY_COLS.values() if c in df.columns]].copy()
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date").sort_index().astype(float)
        sym = market_prefix(code) + code
        if src == "tencent":
            raw = self._call(self.ak.stock_zh_a_hist_tx, symbol=sym,
                             start_date=start.strftime("%Y-%m-%d"),
                             end_date=end.strftime("%Y-%m-%d"), adjust=self.adjust)
        else:  # sina
            raw = self._call(self.ak.stock_zh_a_daily, symbol=sym,
                             start_date=start.strftime("%Y%m%d"),
                             end_date=end.strftime("%Y%m%d"), adjust=self.adjust)
        if raw is None or raw.empty:
            return pd.DataFrame()
        return normalize_daily(raw)

    def source_report(self) -> str:
        return "、".join(f"{SOURCE_LABELS[k]} {v} 只" for k, v in self._used.items())

    def index_constituents(self, index_code: str) -> pd.DataFrame:
        """指数成分股，如 000300 沪深300 / 000905 中证500 / 000852 中证1000。"""
        path = self.cache / f"index_{index_code}.csv"
        if self._fresh(path, timedelta(days=7)):
            return pd.read_csv(path, dtype={"code": str})
        try:
            df = self._call(self.ak.index_stock_cons_csindex, symbol=index_code)
            df = df.rename(columns={"成分券代码": "code", "成分券名称": "name"})
        except Exception:  # noqa: BLE001
            df = self._call(self.ak.index_stock_cons, symbol=index_code)
            df = df.rename(columns={"品种代码": "code", "品种名称": "name"})
        df = df[["code", "name"]].copy()
        df["code"] = df["code"].map(norm_code)
        df.to_csv(path, index=False)
        return df


# ----------------------------------------------------------------------------
class MockProvider(BaseProvider):
    """离线演示 / 单元测试用：生成确定性的随机游走行情，无需联网。"""

    N = 80

    def __init__(self, history_days: int = 400, seed: int = 7, **_):
        self.history_days = history_days
        self.seed = seed
        codes = [f"{600000 + i * 7:06d}" if i % 2 == 0 else f"{i * 13:06d}" for i in range(self.N)]
        codes += ["600519", "000858", "300750", "601318", "000001"]
        self._list = pd.DataFrame({"code": codes, "name": [f"演示股份{c[-3:]}" for c in codes]})
        self._list.loc[3, "name"] = "*ST演示"  # 用于测试 ST 过滤

    def _rng(self, code: str) -> np.random.Generator:
        h = int(hashlib.md5(f"{code}{self.seed}".encode()).hexdigest()[:8], 16)
        return np.random.default_rng(h)

    def stock_list(self) -> pd.DataFrame:
        return self._list.copy()

    def daily(self, code: str, days: int | None = None) -> pd.DataFrame:
        code = norm_code(code)
        n = int((days or self.history_days) * 250 / 365)
        rng = self._rng(code)
        idx = pd.bdate_range(end=pd.Timestamp(date.today()), periods=n)
        drift = rng.normal(0.0003, 0.0008)
        vol = rng.uniform(0.012, 0.03)
        # 加入周期成分，让均线/MACD 能产生交叉
        cyc = 0.01 * np.sin(np.arange(n) / rng.uniform(8, 25))
        ret = rng.normal(drift, vol, n) + np.diff(np.concatenate([[0], cyc]))
        close = rng.uniform(8, 120) * np.exp(np.cumsum(ret))
        open_ = close * (1 + rng.normal(0, 0.005, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
        volume = rng.uniform(5e4, 5e5) * (1 + np.abs(rng.normal(0, 0.5, n)))
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                           "volume": volume}, index=idx)
        df.index.name = "date"
        df["amount"] = df["volume"] * df["close"] * 100
        df["pct_chg"] = df["close"].pct_change().fillna(0) * 100
        df["turnover"] = rng.uniform(0.5, 6, n)
        return df.round(3)

    def spot(self) -> pd.DataFrame:
        rows = []
        for code, name in self._list.itertuples(index=False):
            d = self.daily(code)
            last, prev = d.iloc[-1], d.iloc[-2]
            rng = self._rng(code + "spot")
            rows.append({
                "code": code, "name": name, "price": last.close,
                "pct_chg": (last.close / prev.close - 1) * 100,
                "open": last.open, "high": last.high, "low": last.low, "prev_close": prev.close,
                "volume": last.volume, "amount": last.amount, "turnover": last.turnover,
                "pe": rng.uniform(-20, 80), "pb": rng.uniform(0.5, 8),
                "total_mv": rng.uniform(2e9, 5e11), "float_mv": rng.uniform(3e9, 4e10),
                "chg_60d": (last.close / d.close.iloc[-60] - 1) * 100,
                "chg_ytd": rng.uniform(-30, 60),
                "volume_ratio": last.volume / d.volume.iloc[-6:-1].mean(),
            })
        return pd.DataFrame(rows)

    def index_constituents(self, index_code: str) -> pd.DataFrame:
        return self._list.iloc[:40].copy()


def make_provider(cfg: dict, demo: bool = False) -> BaseProvider:
    d = cfg["data"]
    if demo:
        return MockProvider(history_days=d["history_days"])
    return AKShareProvider(cache_dir=d["cache_dir"], history_days=d["history_days"],
                           adjust=d["adjust"], retries=d["retries"],
                           bypass_proxy=d.get("bypass_proxy", True),
                           sources=d.get("sources"))
