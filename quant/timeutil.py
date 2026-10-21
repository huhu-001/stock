"""A股交易时间工具（北京时间）。节假日未内置，节假日当天仅会多一次无效刷新。"""
from __future__ import annotations

from datetime import datetime, time, timedelta

try:
    from zoneinfo import ZoneInfo
    CN_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # noqa: BLE001  (极老的 Python 环境)
    from datetime import timezone
    CN_TZ = timezone(timedelta(hours=8))

SESSIONS = [(time(9, 30), time(11, 30)), (time(13, 0), time(15, 0))]


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def is_trading_day(dt: datetime) -> bool:
    return dt.weekday() < 5


def is_trading_time(dt: datetime | None = None) -> bool:
    dt = dt or now_cn()
    if not is_trading_day(dt):
        return False
    t = dt.time()
    return any(s <= t <= e for s, e in SESSIONS)


def last_close_time(dt: datetime | None = None) -> datetime:
    """最近一次收盘（15:00）时间点，用于判断日线缓存是否过期。"""
    dt = dt or now_cn()
    d = dt
    if not (is_trading_day(d) and d.time() >= time(15, 5)):
        d = d - timedelta(days=1)
        while not is_trading_day(d):
            d -= timedelta(days=1)
    return d.replace(hour=15, minute=5, second=0, microsecond=0)


def seconds_to_next_session(dt: datetime | None = None) -> float:
    dt = dt or now_cn()
    for add in range(0, 8):
        day = dt + timedelta(days=add)
        if not is_trading_day(day):
            continue
        for s, _ in SESSIONS:
            start = day.replace(hour=s.hour, minute=s.minute, second=0, microsecond=0)
            if start > dt:
                return (start - dt).total_seconds()
    return 3600.0
