"""
趋势策略 — MA20 > MA60 → signal = 1
"""
import pandas as pd
from utils import fillna_df


def generate_signal(df: pd.DataFrame) -> pd.Series:
    df = fillna_df(df)
    close = df["close"]
    ma20 = close.rolling(20).mean().fillna(0)
    ma60 = close.rolling(60).mean().fillna(0)
    signal = (ma20 > ma60).astype(int)
    signal = signal.fillna(0)
    return signal
