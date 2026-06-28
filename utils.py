"""
工具模块 — 通用函数
"""
import numpy as np
import pandas as pd


def generate_fallback_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成 random walk 模拟数据作为 fallback"""
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
    price = 100.0
    prices = [price]
    for _ in range(1, n):
        price *= 1 + np.random.normal(0.0002, 0.015)
        prices.append(price)
    df = pd.DataFrame({"date": dates, "close": prices})
    df["open"] = df["close"] * (1 + np.random.normal(0, 0.005, n))
    df["high"] = df["close"] * (1 + np.abs(np.random.normal(0, 0.01, n)))
    df["low"] = df["close"] * (1 - np.abs(np.random.normal(0, 0.01, n)))
    df["volume"] = np.random.randint(10_000_000, 50_000_000, n)
    df.set_index("date", inplace=True)
    df = df.fillna(0)
    return df


def safe_divide(a: float, b: float) -> float:
    """安全除法，分母为0返回0"""
    return a / b if b != 0 else 0.0


def fillna_df(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 统一 fillna(0)"""
    return df.fillna(0)
