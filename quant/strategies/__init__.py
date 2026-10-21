from __future__ import annotations

from .base import BUY, HOLD, SELL, Signal, Strategy
from .multi_factor import MultiFactorStrategy
from .technical import MACDStrategy, MACrossStrategy, MeanReversionStrategy

TECHNICAL = {s.name: s for s in (MACrossStrategy, MACDStrategy, MeanReversionStrategy)}
REGISTRY = {**TECHNICAL, MultiFactorStrategy.name: MultiFactorStrategy}


def build(name: str, cfg: dict | None = None):
    """按名称 + config.yaml 中的参数实例化策略。"""
    if name not in REGISTRY:
        raise KeyError(f"未知策略 {name}，可选: {', '.join(REGISTRY)}")
    params = ((cfg or {}).get("strategies") or {}).get(name) or {}
    return REGISTRY[name](**params)


__all__ = ["BUY", "SELL", "HOLD", "Signal", "Strategy", "REGISTRY", "TECHNICAL", "build",
           "MACrossStrategy", "MACDStrategy", "MeanReversionStrategy", "MultiFactorStrategy"]
