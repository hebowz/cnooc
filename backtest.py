"""
回测系统 — Sharpe/Sortino/盈亏比 + 有/无融资对比
"""
import pandas as pd
import numpy as np
from utils import fillna_df


def run_enhanced_backtest(df: pd.DataFrame, signals: pd.Series,
                          margin_pct: float = 0.0, rate: float = 0.0355) -> dict:
    """
    signals: 仓位权重 Series (0.0~1.0)，index 对齐 df
    margin_pct: 融资比例 (0.0 = 无融资, 0.5 = 50%融资)
    rate: 融资年利率
    """
    df = fillna_df(df)
    close = df["close"].fillna(0)
    returns = close.pct_change().fillna(0)

    signals = signals.reindex(returns.index).fillna(0)

    # 策略收益 = 前一日信号 × 当日收益 × (1 + margin_pct)
    leverage = 1 + margin_pct
    strategy_returns = signals.shift(1).fillna(0) * returns * leverage

    # 融资利息日化扣除
    if margin_pct > 0:
        daily_rate = rate / 252
        strategy_returns = strategy_returns - margin_pct * daily_rate

    strategy_returns = strategy_returns.fillna(0)

    cumulative = (1 + strategy_returns).cumprod()
    cumulative_return = (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0.0

    # 年化收益
    years = len(strategy_returns) / 252
    ann_return = (1 + cumulative_return / 100) ** (1 / max(1, years)) - 1 if years > 0 else 0

    # Sharpe ratio
    ann_ret = strategy_returns.mean() * 252
    ann_vol = strategy_returns.std() * np.sqrt(252)
    sharpe = round(float(ann_ret / ann_vol), 3) if ann_vol > 0 else 0.0

    # Sortino ratio
    downside = strategy_returns[strategy_returns < 0]
    ann_downside = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = round(float(ann_ret / ann_downside), 3) if ann_downside > 0 else 0.0

    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = round(float(drawdown.min() * 100), 2) if len(drawdown) > 0 else 0.0

    # 胜率
    winning_days = (strategy_returns > 0).sum()
    total_days = max(1, (strategy_returns != 0).sum())
    win_rate = round(float(winning_days / total_days * 100), 1)

    # 盈亏比
    avg_win = strategy_returns[strategy_returns > 0].mean() if winning_days > 0 else 0
    avg_loss = abs(strategy_returns[strategy_returns < 0].mean()) if (strategy_returns < 0).sum() > 0 else 1e-10
    profit_factor = round(float(avg_win / avg_loss), 2) if avg_loss > 0 else float("inf")

    # 净值曲线
    nav_series = cumulative

    return {
        "cumulative_return_pct": round(float(cumulative_return), 2),
        "annual_return_pct": round(float(ann_return * 100), 2),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_drawdown,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "total_days": len(strategy_returns),
        "nav_curve": nav_series,
        "margin_pct": margin_pct,
        "margin_rate": rate,
    }


def compare_margin_vs_nomargin(df: pd.DataFrame, signals: pd.Series) -> dict:
    """有/无融资对比"""
    no_margin = run_enhanced_backtest(df, signals, margin_pct=0.0)
    with_margin = run_enhanced_backtest(df, signals, margin_pct=0.5)

    return {
        "no_margin": no_margin,
        "with_margin": with_margin,
        "diff_return": round(with_margin["cumulative_return_pct"] - no_margin["cumulative_return_pct"], 2),
        "diff_sharpe": round(with_margin["sharpe_ratio"] - no_margin["sharpe_ratio"], 3),
        "diff_drawdown": round(with_margin["max_drawdown_pct"] - no_margin["max_drawdown_pct"], 2),
        "verdict": "融资增强收益" if with_margin["cumulative_return_pct"] > no_margin["cumulative_return_pct"] else "融资未增强收益",
    }


def run_backtest(df: pd.DataFrame, signal: pd.Series) -> dict:
    """向后兼容旧接口"""
    return run_enhanced_backtest(df, signal, margin_pct=0.0)
