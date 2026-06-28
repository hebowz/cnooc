"""
技术指标库 — MACD, KDJ, ATR, Bollinger, EMA, BOS, CHOCH, FVG, Order Block, 均线共振
+ ADX, OBV, 换手率, ICT/SMC (Liquidity Sweep, Breaker Block, Premium-Discount, Equal High-Low)
+ 多时间框架 (周线resample, 共识评分)
"""
import pandas as pd
import numpy as np
from utils import fillna_df


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """RSI 相对强弱指标 (EMA method)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD 指标"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return {"DIF": dif, "DEA": dea, "MACD_hist": macd_hist}


def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3) -> dict:
    """KDJ 指标"""
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return {"K": k, "D": d, "J": j}


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR 平均真实波幅"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def calc_ema(close: pd.Series, periods: list = None) -> dict:
    """多周期 EMA"""
    if periods is None:
        periods = [12, 26, 50, 200]
    result = {}
    for p in periods:
        result[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()
    return result


def calc_bollinger(close: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
    """布林带"""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bandwidth = (upper - lower) / mid * 100
    return {"upper": upper, "mid": mid, "lower": lower, "bandwidth": bandwidth}


# ═══════════════════════════════════════════════════════════
#  BOS / CHOCH / FVG / Order Block
# ═══════════════════════════════════════════════════════════

def detect_bos(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 10) -> pd.Series:
    """Break of Structure (BOS): 价格突破前高或跌破前低"""
    highest = high.rolling(lookback).max().shift(1)
    lowest = low.rolling(lookback).min().shift(1)
    bos = pd.Series(0, index=close.index)
    bos[close > highest] = 1
    bos[close < lowest] = -1
    return bos


def detect_choch(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 10) -> pd.Series:
    """Change of Character (CHOCH): 突破了前一段的趋势结构"""
    bos = detect_bos(high, low, close, lookback)
    choch = pd.Series(0, index=close.index)
    prev_bos = bos.shift(1).fillna(0)
    choch[(bos == 1) & (prev_bos == -1)] = 1
    choch[(bos == -1) & (prev_bos == 1)] = -1
    return choch


def detect_fvg(high: pd.Series, low: pd.Series, close: pd.Series, threshold: float = 0.005) -> pd.Series:
    """Fair Value Gap (FVG): 价格缺口"""
    fvg = pd.Series(0, index=close.index)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    next_low = low.shift(-1)
    next_high = high.shift(-1)
    bullish_fvg = next_low > prev_high * (1 + threshold)
    bearish_fvg = next_high < prev_low * (1 - threshold)
    fvg[bullish_fvg] = 1
    fvg[bearish_fvg] = -1
    return fvg


def detect_order_block(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Order Block: 放量大阳/大阴区域"""
    returns = close.pct_change()
    vol_avg = volume.rolling(lookback).mean()
    vol_surge = volume > vol_avg * 1.5
    ob = pd.Series(0, index=close.index)
    ob[vol_surge & (returns > 0.02)] = 1
    ob[vol_surge & (returns < -0.02)] = -1
    return ob


def detect_ma_resonance(close: pd.Series, periods: list = None) -> dict:
    """多周期均线共振判断"""
    if periods is None:
        periods = [5, 10, 20, 60, 120]
    ema_dict = calc_ema(close, periods)
    current = close.iloc[-1]
    resonance = {"bullish": 0, "bearish": 0, "total": len(periods)}
    for p, ema_s in ema_dict.items():
        if current > ema_s.iloc[-1]:
            resonance["bullish"] += 1
        else:
            resonance["bearish"] += 1
    ratio = resonance["bullish"] / resonance["total"]
    if ratio >= 0.8:
        resonance["signal"] = "强多头共振"
    elif ratio >= 0.6:
        resonance["signal"] = "偏多共振"
    elif ratio >= 0.4:
        resonance["signal"] = "震荡"
    elif ratio >= 0.2:
        resonance["signal"] = "偏空共振"
    else:
        resonance["signal"] = "强空头共振"
    return resonance


# ═══════════════════════════════════════════════════════════
#  新增 — ADX / OBV / 换手率
# ═══════════════════════════════════════════════════════════

def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX 平均趋向指数 — 衡量趋势强度（非方向）"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=close.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm = pd.Series(0.0, index=close.index)
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV 能量潮 — 累积量价关系"""
    direction = pd.Series(0, index=close.index)
    direction[close > close.shift(1)] = 1
    direction[close < close.shift(1)] = -1
    obv = (direction * volume).cumsum()
    return obv


def calc_turnover_rate(volume: pd.Series, shares_outstanding: float = None) -> pd.Series:
    """换手率 — 日成交/流通股本"""
    if shares_outstanding is None or shares_outstanding <= 0:
        shares_outstanding = 30_000_000_000  # 默认约 300亿股
    return volume / shares_outstanding * 100


# ═══════════════════════════════════════════════════════════
#  ICT/SMC 高级指标
# ═══════════════════════════════════════════════════════════

def detect_liquidity_sweep(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 20) -> pd.Series:
    """Liquidity Sweep: 价格短暂突破前高/前低后快速回撤 → 流动性猎杀"""
    highest = high.rolling(lookback).max().shift(1)
    lowest = low.rolling(lookback).min().shift(1)
    sweep = pd.Series(0, index=close.index)
    for i in range(lookback + 1, len(high) - 1):
        if high.iloc[i] > highest.iloc[i] and close.iloc[i] < close.iloc[i - 1]:
            sweep.iloc[i] = -1  # 向上假突破 → 空头猎杀
        elif low.iloc[i] < lowest.iloc[i] and close.iloc[i] > close.iloc[i - 1]:
            sweep.iloc[i] = 1   # 向下假跌破 → 多头猎杀
    return sweep


def detect_breaker_block(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Breaker Block: Order Block 被突破后形成的支撑/阻力区域"""
    ob = detect_order_block(high, low, close, volume, lookback)
    breaker = pd.Series(0, index=close.index)
    for i in range(lookback + 5, len(close)):
        prev_ob = ob.iloc[i - 5:i]
        if (prev_ob == 1).any() and close.iloc[i] < low.iloc[i - 2]:
            breaker.iloc[i] = -1  # 多头 OB 被跌破
        elif (prev_ob == -1).any() and close.iloc[i] > high.iloc[i - 2]:
            breaker.iloc[i] = 1   # 空头 OB 被突破
    return breaker


def detect_premium_discount(close: pd.Series) -> pd.Series:
    """Premium/Discount 区域: 相对 EMA200 的溢价/折价"""
    ema200 = close.ewm(span=200, adjust=False).mean()
    pd_zone = pd.Series(0.0, index=close.index)
    ratio = (close - ema200) / ema200
    pd_zone[ratio > 0.10] = 1.0   # Premium (溢价)
    pd_zone[ratio < -0.10] = -1.0  # Discount (折价)
    # 中间区域保持0 → 均衡区
    return pd_zone


def detect_equal_high_low(high: pd.Series, low: pd.Series, threshold: float = 0.003) -> pd.Series:
    """Equal High / Equal Low: 相等高/低点 → 流动性汇聚"""
    eql = pd.Series(0, index=high.index)
    for i in range(3, len(high)):
        h_close = (high.iloc[i] - high.iloc[i - 1:i]) / high.iloc[i]
        if (abs(h_close) < threshold).any():
            eql.iloc[i] = 1  # Equal High
        l_close = (low.iloc[i] - low.iloc[i - 1:i]) / low.iloc[i]
        if (abs(l_close) < threshold).any():
            eql.iloc[i] = -1  # Equal Low
    return eql


# ═══════════════════════════════════════════════════════════
#  多时间框架
# ═══════════════════════════════════════════════════════════

def resample_ohlcv(df: pd.DataFrame, timeframe: str = 'W') -> pd.DataFrame:
    """日线 → 周线/月线 重采样"""
    rules = {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last', 'volume': 'sum'}
    available = {k: v for k, v in rules.items() if k in df.columns}
    if timeframe == 'W':
        return df.resample('W').apply(available)
    elif timeframe == 'M':
        return df.resample('M').apply(available)
    else:
        raise ValueError(f"不支持的时间框架: {timeframe}")


def compute_mtf_signals(df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> dict:
    """多时间框架共识评分
    返回: consensus (日+周的一致度), daily_trend, weekly_trend, details
    """
    close_d = df_daily["close"].dropna()
    close_w = df_weekly["close"].dropna()

    # 日线趋势
    ma20_d = float(close_d.rolling(20).mean().iloc[-1]) if len(close_d) >= 20 else 0
    ma60_d = float(close_d.rolling(60).mean().iloc[-1]) if len(close_d) >= 60 else 0
    current_d = float(close_d.iloc[-1])
    daily_trend = "bullish" if current_d > ma20_d > ma60_d else ("bearish" if current_d < ma20_d < ma60_d else "neutral")

    # 周线趋势
    ma20_w = float(close_w.rolling(20).mean().iloc[-1]) if len(close_w) >= 20 else 0
    ma40_w = float(close_w.rolling(40).mean().iloc[-1]) if len(close_w) >= 40 else 0
    current_w = float(close_w.iloc[-1])
    weekly_trend = "bullish" if current_w > ma20_w > ma40_w else ("bearish" if current_w < ma20_w < ma40_w else "neutral")

    # 共识
    if daily_trend == weekly_trend == "bullish":
        consensus_score = 85
        signal = "日/周线多头共振，趋势强烈看多"
    elif daily_trend == "bullish" and weekly_trend == "neutral":
        consensus_score = 65
        signal = "日线偏多，周线中性，短线做多"
    elif daily_trend == "bullish" and weekly_trend == "bearish":
        consensus_score = 45
        signal = "日线偏多但周线偏空，短线反弹对待"
    elif daily_trend == "bearish" and weekly_trend == "bearish":
        consensus_score = 15
        signal = "日/周线空头共振，趋势强烈看空"
    elif daily_trend == "bearish" and weekly_trend == "bullish":
        consensus_score = 55
        signal = "日线偏空但周线偏多，中期趋势未坏"
    else:
        consensus_score = 50
        signal = "日/周线信号不一，等待方向明确"

    return {"consensus_score": consensus_score, "consensus_signal": signal,
            "daily_trend": daily_trend, "weekly_trend": weekly_trend,
            "daily_current": round(current_d, 2), "weekly_current": round(current_w, 2)}


# ═══════════════════════════════════════════════════════════
#  汇总函数
# ═══════════════════════════════════════════════════════════

def compute_all_technical(df: pd.DataFrame) -> dict:
    """一次性计算所有基础技术指标，返回字典"""
    df = fillna_df(df)
    close = df["close"].dropna()
    high = df["high"].dropna()
    low = df["low"].dropna()
    volume = df["volume"].dropna() if "volume" in df.columns else pd.Series(1, index=close.index)

    macd = calc_macd(close)
    kdj = calc_kdj(high, low, close)
    atr = calc_atr(high, low, close)
    ema_dict = calc_ema(close)
    bb = calc_bollinger(close)
    bos = detect_bos(high, low, close)
    choch = detect_choch(high, low, close)
    fvg = detect_fvg(high, low, close)
    ob = detect_order_block(high, low, close, volume)
    resonance = detect_ma_resonance(close)

    def last_val(s):
        return float(s.dropna().iloc[-1]) if len(s.dropna()) > 0 else 0.0

    def last_signal(s):
        return int(s.dropna().iloc[-1]) if len(s.dropna()) > 0 else 0

    return {
        "macd": {"DIF": last_val(macd["DIF"]), "DEA": last_val(macd["DEA"]), "hist": last_val(macd["MACD_hist"])},
        "kdj": {"K": last_val(kdj["K"]), "D": last_val(kdj["D"]), "J": last_val(kdj["J"])},
        "atr": last_val(atr),
        "ema": {k: last_val(v) for k, v in ema_dict.items()},
        "bollinger": {"upper": last_val(bb["upper"]), "mid": last_val(bb["mid"]), "lower": last_val(bb["lower"]), "bandwidth": last_val(bb["bandwidth"])},
        "bos": last_signal(bos),
        "choch": last_signal(choch),
        "fvg": last_signal(fvg),
        "order_block": last_signal(ob),
        "ma_resonance": resonance,
    }


def compute_all_technical_extended(df: pd.DataFrame) -> dict:
    """计算所有扩展技术指标（含 ICT/SMC + 多时间框架）"""
    df = fillna_df(df)
    close = df["close"].dropna()
    high = df["high"].dropna()
    low = df["low"].dropna()
    volume = df["volume"].dropna() if "volume" in df.columns else pd.Series(1, index=close.index)

    adx = calc_adx(high, low, close)
    obv = calc_obv(close, volume)
    liq_sweep = detect_liquidity_sweep(high, low, close)
    breaker = detect_breaker_block(high, low, close, volume)
    pd_zone = detect_premium_discount(close)
    eql = detect_equal_high_low(high, low)

    df_weekly = resample_ohlcv(df, 'W')
    mtf = compute_mtf_signals(df, df_weekly)

    def last_val(s):
        return float(s.dropna().iloc[-1]) if len(s.dropna()) > 0 else 0.0

    def last_signal(s):
        return int(s.dropna().iloc[-1]) if len(s.dropna()) > 0 else 0

    return {
        "adx": {"value": last_val(adx),
                "desc": "强趋势" if last_val(adx) > 40 else ("趋势有效" if last_val(adx) > 25 else ("偏弱" if last_val(adx) > 20 else "盘整"))},
        "obv": {"value": last_val(obv),
                "trend_5d": "上升" if len(obv.dropna()) >= 5 and last_val(obv) > float(obv.dropna().iloc[-5]) else "下降"},
        "liquidity_sweep": {"signal": last_signal(liq_sweep),
                           "desc": "多头猎杀(假跌破)" if last_signal(liq_sweep) == 1 else ("空头猎杀(假突破)" if last_signal(liq_sweep) == -1 else "无")},
        "breaker_block": {"signal": last_signal(breaker),
                         "desc": "空头OB被突破(偏多)" if last_signal(breaker) == 1 else ("多头OB被跌破(偏空)" if last_signal(breaker) == -1 else "无")},
        "premium_discount": {"zone": last_val(pd_zone),
                            "desc": "溢价区(偏高)" if last_val(pd_zone) > 0 else ("折价区(偏低)" if last_val(pd_zone) < 0 else "均衡区")},
        "equal_high_low": {"signal": last_signal(eql),
                          "desc": "Equal High(阻力汇聚)" if last_signal(eql) == 1 else ("Equal Low(支撑汇聚)" if last_signal(eql) == -1 else "无")},
        "mtf": mtf,
    }
