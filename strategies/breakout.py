"""
突破策略 — 价格突破20日新高 → signal = 1
"""
import pandas as pd
from utils import fillna_df


def generate_signal(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    df = fillna_df(df)
    close = df["close"]
    high_20 = close.rolling(lookback).max().fillna(0)
    # 当日收盘 > 前一日20日最高（突破确认）
    signal = (close > high_20.shift(1)).astype(int)
    signal = signal.fillna(0)
    return signal
