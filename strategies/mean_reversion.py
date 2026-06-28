"""
均值回归策略 — RSI < 30 → signal = 1
"""
import pandas as pd
from utils import fillna_df
from strategies.technical import compute_rsi


def generate_signal(df: pd.DataFrame, rsi_length: int = 14) -> pd.Series:
    df = fillna_df(df)
    close = df["close"]
    rsi = compute_rsi(close, length=rsi_length)
    rsi = rsi.fillna(50)
    signal = (rsi < 30).astype(int)
    signal = signal.fillna(0)
    return signal
