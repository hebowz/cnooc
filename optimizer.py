"""
参数优化 — Grid Search MA 参数
"""
import pandas as pd
import numpy as np
from backtest import run_backtest
from utils import fillna_df


def optimize_ma(df: pd.DataFrame) -> dict:
    """
    Grid Search 寻找最优 MA 参数
    MA short: 10, 20, 30
    MA long: 50, 60, 80
    """
    df = fillna_df(df)
    short_options = [10, 20, 30]
    long_options = [50, 60, 80]
    best_sharpe = -999.0
    best_params = {}
    results = []
    for short in short_options:
        for long in long_options:
            if short >= long:
                continue
            close = df["close"]
            ma_short = close.rolling(short).mean().fillna(0)
            ma_long = close.rolling(long).mean().fillna(0)
            signal = (ma_short > ma_long).astype(int).fillna(0)
            bt = run_backtest(df, signal)
            sharpe = bt["sharpe_ratio"]
            results.append({"ma_short": short, "ma_long": long, "sharpe": sharpe})
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {"ma_short": short, "ma_long": long}
    return {
        "best_params": best_params,
        "best_sharpe": best_sharpe,
        "all_results": results,
    }
