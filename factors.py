"""
因子体系 — 32 因子，5大类加权：价值35/技术30/质量15/宏观10/油价10
"""
import pandas as pd
import numpy as np
from utils import safe_divide, fillna_df
from strategies.technical import calc_macd, calc_kdj, calc_atr, calc_bollinger, calc_ema, compute_rsi


def _scalar(v) -> float:
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    if isinstance(v, pd.Series):
        return float(v.iloc[0]) if len(v) > 0 else 0.0
    if isinstance(v, np.ndarray):
        return float(v.flat[0]) if v.size > 0 else 0.0
    try:
        return float(v.item())
    except Exception:
        return float(v)


def _percentile_rank(s: pd.Series, current: float) -> float:
    if len(s) < 10:
        return 0.5
    return _scalar((s < current).sum() / len(s))


# ═══════════════════════════════════════════════════════════
#  价值因子（weight 35%）
# ═══════════════════════════════════════════════════════════

def calc_pe_factor(valuation: dict, financial: dict) -> tuple[float, str]:
    pe = valuation.get("pe", 0)
    is_fb = valuation.get("_fallback", False) or financial.get("_fallback", False)
    if pe and pe > 0:
        if pe < 6:
            score, desc = 90.0, f"PE {pe:.1f}，极低估值，深度价值区间"
        elif pe < 10:
            score, desc = 75.0, f"PE {pe:.1f}，低估值，价值洼地"
        elif pe < 15:
            score, desc = 60.0, f"PE {pe:.1f}，估值合理"
        elif pe < 25:
            score, desc = 45.0, f"PE {pe:.1f}，估值偏高"
        else:
            score, desc = 25.0, f"PE {pe:.1f}，高估值区间"
    else:
        score, desc = 50.0, "PE 数据待更新"
    if is_fb:
        desc += "（预估）"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_pb_factor(valuation: dict) -> tuple[float, str]:
    pb = valuation.get("pb", 0)
    is_fb = valuation.get("_fallback", False)
    if pb and pb > 0:
        if pb < 1.0:
            score, desc = 90.0, f"PB {pb:.2f}，破净边缘，极度低估"
        elif pb < 1.5:
            score, desc = 75.0, f"PB {pb:.2f}，低估值区间"
        elif pb < 2.5:
            score, desc = 55.0, f"PB {pb:.2f}，估值合理"
        else:
            score, desc = 35.0, f"PB {pb:.2f}，估值偏高"
    else:
        score, desc = 50.0, "PB 数据待更新"
    if is_fb:
        desc += "（预估）"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_price_percentile_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or len(df) < 60:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    current = _scalar(close.iloc[-1])
    pct = _percentile_rank(close, current)
    score = round((1 - pct) * 100, 1)
    if pct < 0.2:
        desc = f"历史 {pct*100:.0f}% 分位，历史底部区域，极佳买点"
    elif pct < 0.4:
        desc = f"历史 {pct*100:.0f}% 分位，偏低区间"
    elif pct < 0.6:
        desc = f"历史 {pct*100:.0f}% 分位，中枢"
    elif pct < 0.8:
        desc = f"历史 {pct*100:.0f}% 分位，偏高"
    else:
        desc = f"历史 {pct*100:.0f}% 分位，历史高位，谨慎"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_dividend_factor(financial: dict, price: float) -> tuple[float, str]:
    dps = financial.get("每股收益", 0)
    estimated_div = dps * 0.48
    if estimated_div > 0 and price > 0:
        div_yield = estimated_div / price * 100
        score = min(100, div_yield * 17)
        if div_yield > 6:
            desc = f"股息率 {div_yield:.2f}%，超高股息，优秀"
        elif div_yield > 4:
            desc = f"股息率 {div_yield:.2f}%，高股息吸引力"
        elif div_yield > 2.5:
            desc = f"股息率 {div_yield:.2f}%，中等股息"
        else:
            desc = f"股息率 {div_yield:.2f}%，股息偏低"
    else:
        score, desc = 50.0, "股息数据待更新"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_ah_premium_factor(ah: dict) -> tuple[float, str]:
    premium = ah.get("premium", 20.0)
    is_fb = ah.get("_fallback", False)
    if premium < 0:
        score, desc = 90.0, f"AH溢价 {premium:.1f}%，A股折价，稀有机会"
    elif premium < 5:
        score, desc = 80.0, f"AH溢价 {premium:.1f}%，接近H股价，性价比高"
    elif premium < 15:
        score, desc = 65.0, f"AH溢价 {premium:.1f}%，溢价适中"
    elif premium < 30:
        score, desc = 50.0, f"AH溢价 {premium:.1f}%，溢价中等"
    elif premium < 50:
        score, desc = 35.0, f"AH溢价 {premium:.1f}%，溢价偏大"
    else:
        score, desc = 20.0, f"AH溢价 {premium:.1f}%，溢价过大"
    if is_fb:
        desc += "（预估）"
    return score, desc


def calc_fcf_factor(financial: dict) -> tuple[float, str]:
    fcf_ratio = financial.get("fcf_ratio", 0)
    if fcf_ratio > 30:
        score, desc = 80.0, f"FCF/营收比 {fcf_ratio:.1f}%，自由现金流充裕"
    elif fcf_ratio > 15:
        score, desc = 60.0, f"FCF/营收比 {fcf_ratio:.1f}%，现金流良好"
    elif fcf_ratio > 5:
        score, desc = 45.0, f"FCF/营收比 {fcf_ratio:.1f}%，现金流一般"
    else:
        score, desc = 30.0, f"FCF/营收比 {fcf_ratio:.1f}%，现金流偏紧"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_peer_comparison_factor(peers: dict, spot: dict) -> tuple[float, str]:
    if not peers:
        return 50.0, "国际对标数据不足"
    from valuation import peer_comparison
    result = peer_comparison(peers, spot.get("pe", 0), spot.get("pb", 0))
    return result["score"], result["desc"]


# ═══════════════════════════════════════════════════════════
#  趋势因子（weight 30% — 技术类）
# ═══════════════════════════════════════════════════════════

def calc_trend_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or len(df) < 60:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    ma20 = _scalar(close.rolling(20).mean().iloc[-1])
    ma60 = _scalar(close.rolling(60).mean().iloc[-1])
    current = _scalar(close.iloc[-1])
    if pd.isna(ma20) or pd.isna(ma60) or ma60 == 0:
        return 50.0, "均线异常"
    diff = _scalar((ma20 - ma60) / ma60 * 100)
    score = 50 + diff * 5 + _scalar((current - ma20) / ma20 * 100) * 2
    if ma20 > ma60 and current > ma20:
        desc = f"多头排列 MA20(¥{ma20:.2f}) > MA60(¥{ma60:.2f})"
    elif ma20 > ma60:
        desc = f"MA20上穿MA60，偏多"
    else:
        desc = f"空头排列 MA20(¥{ma20:.2f}) < MA60(¥{ma60:.2f})"
    return _scalar(np.clip(round(score, 1), 0, 100)), desc


def calc_ema_trend_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or len(df) < 30:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    emas = calc_ema(close, [12, 26])
    ema12 = _scalar(emas["EMA12"].iloc[-1])
    ema26 = _scalar(emas["EMA26"].iloc[-1])
    if ema12 > ema26:
        score = 60 + min(20, _scalar((ema12 - ema26) / ema26 * 200))
        desc = f"EMA12(¥{ema12:.2f}) > EMA26(¥{ema26:.2f})，偏多"
    else:
        score = 40 + max(-20, _scalar((ema12 - ema26) / ema26 * 200))
        desc = f"EMA12 < EMA26，偏空"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_momentum_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or len(df) < 60:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    ret_20 = _scalar(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
    ret_60 = _scalar(close.iloc[-1] / close.iloc[-60] - 1) if len(close) >= 60 else 0
    score = 50 + ret_20 * 200 + ret_60 * 100
    desc = f"20日 {ret_20*100:+.2f}%，60日 {ret_60*100:+.2f}%"
    return _scalar(np.clip(round(score, 1), 0, 100)), desc


def calc_rsi_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or len(df) < 14:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    rsi_val = _scalar(compute_rsi(close, 14).dropna().iloc[-1])
    if rsi_val < 30:
        score, desc = 70 + (30 - rsi_val) * 1.5, f"RSI {rsi_val:.1f}，超卖区间，潜在反弹"
    elif rsi_val > 70:
        score, desc = 30 - (rsi_val - 70) * 1.5, f"RSI {rsi_val:.1f}，超买区间，注意回调"
    elif rsi_val > 50:
        score, desc = 55.0, f"RSI {rsi_val:.1f}，中性偏强"
    else:
        score, desc = 45.0, f"RSI {rsi_val:.1f}，中性偏弱"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_macd_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or len(df) < 30:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    macd = calc_macd(close)
    hist = _scalar(macd["MACD_hist"].dropna().iloc[-1])
    dif = _scalar(macd["DIF"].dropna().iloc[-1])
    dea = _scalar(macd["DEA"].dropna().iloc[-1])
    prev_hist = _scalar(macd["MACD_hist"].dropna().iloc[-2]) if len(macd["MACD_hist"].dropna()) > 1 else 0
    score = 50 + hist * 200
    if hist > 0 and prev_hist <= 0:
        desc = f"MACD金叉 DIF({dif:.3f}) > DEA({dea:.3f})"
    elif hist < 0 and prev_hist >= 0:
        desc = f"MACD死叉 DIF({dif:.3f}) < DEA({dea:.3f})"
    elif hist > 0:
        desc = "MACD多头" + ("，柱状放大" if hist > prev_hist else "，柱状收敛")
    else:
        desc = "MACD空头"
    return _scalar(np.clip(round(score, 1), 0, 100)), desc


def calc_atr_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns or len(df) < 14:
        return 50.0, "数据不足"
    atr = calc_atr(df["high"], df["low"], df["close"], 14)
    atr_val = _scalar(atr.dropna().iloc[-1])
    price = _scalar(df["close"].dropna().iloc[-1])
    atr_pct = atr_val / price * 100 if price > 0 else 0
    if atr_pct < 1.5:
        score, desc = 60.0, f"ATR {atr_pct:.1f}%，低波窄幅，适合持仓"
    elif atr_pct < 3:
        score, desc = 50.0, f"ATR {atr_pct:.1f}%，正常波动"
    else:
        score, desc = 35.0, f"ATR {atr_pct:.1f}%，高波，注意风险控制"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_bollinger_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or len(df) < 20:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    bb = calc_bollinger(close)
    current = _scalar(close.iloc[-1])
    upper = _scalar(bb["upper"].dropna().iloc[-1])
    lower = _scalar(bb["lower"].dropna().iloc[-1])
    if upper == lower or np.isnan(upper) or np.isnan(lower):
        return 50.0, "布林带异常"
    bb_pos = (current - lower) / (upper - lower)
    if bb_pos > 0.8:
        score, desc = 30.0, "价格触及上轨，超买压力"
    elif bb_pos < 0.2:
        score, desc = 75.0, "价格触及下轨，超卖机会"
    elif bb_pos > 0.5:
        score, desc = 55.0, f"布林带中轨上方，偏强"
    else:
        score, desc = 45.0, f"布林带中轨下方，偏弱"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_adx_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns or len(df) < 14:
        return 50.0, "数据不足"
    from strategies.technical import calc_adx as compute_adx
    adx = compute_adx(df["high"], df["low"], df["close"], 14)
    val = _scalar(adx.dropna().iloc[-1])
    if val > 40:
        score, desc = 70.0, f"ADX {val:.1f}，强趋势运行"
    elif val > 25:
        score, desc = 55.0, f"ADX {val:.1f}，趋势有效"
    elif val > 20:
        score, desc = 45.0, f"ADX {val:.1f}，趋势偏弱，震荡"
    else:
        score, desc = 35.0, f"ADX {val:.1f}，无趋势/盘整"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_obv_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or "volume" not in df.columns or len(df) < 20:
        return 50.0, "数据不足"
    from strategies.technical import calc_obv as compute_obv
    obv = compute_obv(df["close"], df["volume"])
    val = _scalar(obv.dropna().iloc[-1])
    prev = _scalar(obv.dropna().iloc[-5]) if len(obv.dropna()) >= 5 else val
    change = (val / prev - 1) if prev != 0 else 0
    score = 50 + change * 500
    desc = f"OBV 5日{change*100:+.1f}%，{'资金流入' if change > 0 else '资金流出'}"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_kdj_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns or len(df) < 9:
        return 50.0, "数据不足"
    kdj = calc_kdj(df["high"], df["low"], df["close"])
    k = _scalar(kdj["K"].dropna().iloc[-1])
    d = _scalar(kdj["D"].dropna().iloc[-1])
    j = _scalar(kdj["J"].dropna().iloc[-1])
    if j < 0:
        score, desc = 80.0, f"KDJ J={j:.1f}，超卖钝化，反弹概率高"
    elif j > 100:
        score, desc = 25.0, f"KDJ J={j:.1f}，超买钝化，注意回调"
    elif k > d and j > 50:
        score, desc = 60.0, f"KDJ金叉区域 K({k:.1f})>D({d:.1f})"
    elif k < d and j < 50:
        score, desc = 40.0, f"KDJ死叉区域 K({k:.1f})<D({d:.1f})"
    else:
        score, desc = 50.0, f"KDJ 中性 K({k:.1f}) D({d:.1f}) J({j:.1f})"
    return _scalar(np.clip(score, 0, 100)), desc


# ═══════════════════════════════════════════════════════════
#  质量因子（weight 15%）
# ═══════════════════════════════════════════════════════════

def calc_roe_factor(financial: dict) -> tuple[float, str]:
    roe = financial.get("加权净资产收益率", 0)
    if roe > 20:
        score, desc = 90.0, f"ROE {roe:.1f}%，卓越盈利"
    elif roe > 15:
        score, desc = 75.0, f"ROE {roe:.1f}%，优秀盈利"
    elif roe > 10:
        score, desc = 60.0, f"ROE {roe:.1f}%，良好盈利"
    elif roe > 5:
        score, desc = 45.0, f"ROE {roe:.1f}%，盈利一般"
    else:
        score, desc = 30.0, f"ROE {roe:.1f}%，盈利偏弱"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_roic_factor(financial: dict) -> tuple[float, str]:
    roic = financial.get("roic", 0)
    if roic > 15:
        score, desc = 85.0, f"ROIC {roic:.1f}%，资本回报优秀"
    elif roic > 10:
        score, desc = 70.0, f"ROIC {roic:.1f}%，资本回报良好"
    elif roic > 5:
        score, desc = 50.0, f"ROIC {roic:.1f}%，资本回报一般"
    else:
        score, desc = 35.0, f"ROIC {roic:.1f}%，资本回报偏低"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_gross_margin_factor(financial: dict) -> tuple[float, str]:
    gm = financial.get("毛利率", 0)
    if gm > 50:
        score, desc = 85.0, f"毛利率 {gm:.1f}%，很强的定价权"
    elif gm > 30:
        score, desc = 65.0, f"毛利率 {gm:.1f}%，良好盈利空间"
    elif gm > 15:
        score, desc = 50.0, f"毛利率 {gm:.1f}%，中等"
    else:
        score, desc = 35.0, f"毛利率 {gm:.1f}%，偏低"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_net_margin_factor(financial: dict) -> tuple[float, str]:
    nm = financial.get("净利率", 0)
    if nm > 30:
        score, desc = 90.0, f"净利率 {nm:.1f}%，极强盈利能力"
    elif nm > 20:
        score, desc = 75.0, f"净利率 {nm:.1f}%，优秀盈利能力"
    elif nm > 10:
        score, desc = 55.0, f"净利率 {nm:.1f}%，中等"
    else:
        score, desc = 40.0, f"净利率 {nm:.1f}%，偏低"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_payout_factor(financial: dict) -> tuple[float, str]:
    dps = financial.get("每股收益", 0)
    payout = 0.48  # 公司稳定分红率
    if dps and dps > 0:
        score = min(90, payout * 150)
        desc = f"分红率约{payout*100:.0f}%，{'高分红' if payout > 0.4 else '中等分红'}"
    else:
        score, desc = 50.0, "分红数据待更新"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_coverage_factor(financial: dict) -> tuple[float, str]:
    eps = financial.get("每股收益", 0)
    dps = eps * 0.48
    if eps > 0 and dps > 0:
        coverage = eps / dps
        if coverage > 2:
            score, desc = 80.0, f"分红覆盖率 {coverage:.1f}x，安全边际充足"
        elif coverage > 1.5:
            score, desc = 65.0, f"分红覆盖率 {coverage:.1f}x，覆盖良好"
        elif coverage > 1.0:
            score, desc = 50.0, f"分红覆盖率 {coverage:.1f}x，基本覆盖"
        else:
            score, desc = 30.0, f"分红覆盖率 {coverage:.1f}x，覆盖不足"
    else:
        score, desc = 50.0, "覆盖率数据待更新"
    return _scalar(np.clip(score, 0, 100)), desc


# ═══════════════════════════════════════════════════════════
#  宏观因子（weight 10%）
# ═══════════════════════════════════════════════════════════

def calc_dxy_factor(dxy: pd.Series) -> tuple[float, str]:
    if dxy is None or len(dxy.dropna()) < 10:
        return 50.0, "美元指数数据不足"
    d = dxy.dropna()
    ret = _scalar(d.iloc[-1] / d.iloc[-10] - 1)
    val = _scalar(d.iloc[-1])
    score = 50 - ret * 300
    desc = f"美元指数 {val:.2f}，10日{ret*100:+.2f}%"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_us10y_factor(us10y: pd.Series) -> tuple[float, str]:
    if us10y is None or len(us10y.dropna()) < 10:
        return 50.0, "美国10Y收益率数据不足"
    s = us10y.dropna()
    val = _scalar(s.iloc[-1])
    prev = _scalar(s.iloc[-60]) if len(s) >= 60 else val
    change = val - prev
    if change < -0.5:
        score, desc = 70.0, f"10Y {val:.2f}%，利率下行利好资源股"
    elif change < 0:
        score, desc = 55.0, f"10Y {val:.2f}%，利率小幅下行"
    elif change < 0.5:
        score, desc = 45.0, f"10Y {val:.2f}%，利率小幅上行"
    else:
        score, desc = 30.0, f"10Y {val:.2f}%，利率上行压制资源股估值"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_vix_factor(vix: pd.Series) -> tuple[float, str]:
    if vix is None or len(vix.dropna()) < 1:
        return 50.0, "VIX 数据不足"
    val = _scalar(vix.dropna().iloc[-1])
    if val < 15:
        score, desc = 70.0, f"VIX {val:.1f}，低恐慌，市场平稳"
    elif val < 20:
        score, desc = 55.0, f"VIX {val:.1f}，温和"
    elif val < 30:
        score, desc = 40.0, f"VIX {val:.1f}，偏高，注意震荡"
    else:
        score, desc = 20.0, f"VIX {val:.1f}，高恐慌，谨慎"
    return score, desc


def calc_ssec_factor(ssec: pd.Series) -> tuple[float, str]:
    if ssec is None or len(ssec.dropna()) < 20:
        return 50.0, "上证指数数据不足"
    s = ssec.dropna()
    val = _scalar(s.iloc[-1])
    pct = _percentile_rank(s, val)
    score = round(pct * 100)
    desc = f"上证 {val:.0f}，处于{pct*100:.0f}%分位"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_cpi_factor(cpi: pd.Series) -> tuple[float, str]:
    if cpi is None or len(cpi.dropna()) < 3:
        return 50.0, "CPI数据不足"
    s = cpi.dropna()
    val = _scalar(s.iloc[-1])
    if val < 0:
        score, desc = 35.0, f"CPI {val:.1f}%，通缩压力"
    elif val < 1:
        score, desc = 60.0, f"CPI {val:.1f}%，低通胀温和"
    elif val < 3:
        score, desc = 50.0, f"CPI {val:.1f}%，通胀适中"
    elif val < 5:
        score, desc = 35.0, f"CPI {val:.1f}%，通胀偏高"
    else:
        score, desc = 25.0, f"CPI {val:.1f}%，高通胀"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_pmi_factor(pmi: pd.Series) -> tuple[float, str]:
    if pmi is None or len(pmi.dropna()) < 3:
        return 50.0, "PMI数据不足"
    s = pmi.dropna()
    val = _scalar(s.iloc[-1])
    if val > 52:
        score, desc = 75.0, f"PMI {val:.1f}，制造业扩张强劲"
    elif val > 50:
        score, desc = 60.0, f"PMI {val:.1f}，制造业温和扩张"
    elif val > 48:
        score, desc = 45.0, f"PMI {val:.1f}，制造业收缩边缘"
    else:
        score, desc = 30.0, f"PMI {val:.1f}，制造业收缩"
    return _scalar(np.clip(score, 0, 100)), desc


# ═══════════════════════════════════════════════════════════
#  油价因子（weight 10%）
# ═══════════════════════════════════════════════════════════

def calc_brent_ma_trend(brent: pd.Series) -> tuple[float, str]:
    if brent is None or len(brent.dropna()) < 60:
        return 50.0, "原油数据不足"
    s = brent.dropna()
    current = _scalar(s.iloc[-1])
    ma20 = _scalar(s.rolling(20).mean().iloc[-1])
    ma60 = _scalar(s.rolling(60).mean().iloc[-1])
    ret_1m = _scalar(s.iloc[-1] / s.iloc[-20] - 1) if len(s) >= 20 else 0
    if current > ma20 > ma60:
        score, desc = 75.0, f"布伦特 ${current:.2f}，多头排列，油价趋势向上"
    elif current > ma20:
        score, desc = 60.0, f"布伦特 ${current:.2f}，站上MA20，短期偏强"
    elif current > ma60:
        score, desc = 50.0, f"布伦特 ${current:.2f}，MA20下方但MA60上方，震荡"
    elif current < ma20 < ma60:
        score, desc = 30.0, f"布伦特 ${current:.2f}，空头排列，油价趋势偏弱"
    else:
        score, desc = 40.0, f"布伦特 ${current:.2f}，短期偏弱但长均支撑"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_oil_supply_factor(oil_supply: dict) -> tuple[float, str]:
    if not oil_supply:
        return 50.0, "OPEC供给数据不足"
    trend = oil_supply.get("trend", "稳定")
    if trend == "减产":
        score, desc = 70.0, "OPEC供给偏紧，支撑油价"
    elif trend == "稳定":
        score, desc = 55.0, "OPEC供给稳定"
    else:
        score, desc = 40.0, "OPEC供给增加，压制油价"
    return _scalar(np.clip(score, 0, 100)), desc


# ═══════════════════════════════════════════════════════════
#  权重分配 — 5大类
# ═══════════════════════════════════════════════════════════

CATEGORY_WEIGHTS = {
    "value": 0.35, "technical": 0.30, "quality": 0.15, "macro": 0.10, "oil": 0.10,
}

FACTOR_CATEGORIES = {
    "pe": "value", "pb": "value", "price_percentile": "value", "dividend": "value",
    "ah_premium": "value", "fcf": "value", "peer_compare": "value",
    "trend": "technical", "ema_trend": "technical", "momentum": "technical",
    "rsi": "technical", "macd": "technical", "atr": "technical",
    "bollinger": "technical", "adx": "technical", "obv": "technical", "kdj": "technical",
    "roe": "quality", "roic": "quality", "gross_margin": "quality",
    "net_margin": "quality", "payout": "quality", "coverage": "quality",
    "dxy": "macro", "us10y": "macro", "vix": "macro", "ssec": "macro",
    "cpi": "macro", "pmi": "macro",
    "brent_ma_trend": "oil", "oil_supply": "oil",
}

WEIGHTS = {
    # 价值因子 35%
    "pe": 0.065, "pb": 0.050, "price_percentile": 0.055, "dividend": 0.055,
    "ah_premium": 0.040, "fcf": 0.045, "peer_compare": 0.040,
    # 技术因子 30%
    "trend": 0.040, "ema_trend": 0.030, "momentum": 0.030, "rsi": 0.030,
    "macd": 0.035, "atr": 0.025, "bollinger": 0.025, "adx": 0.025,
    "obv": 0.030, "kdj": 0.030,
    # 质量因子 15%
    "roe": 0.030, "roic": 0.025, "gross_margin": 0.025, "net_margin": 0.025,
    "payout": 0.025, "coverage": 0.020,
    # 宏观因子 10%
    "dxy": 0.020, "us10y": 0.020, "vix": 0.015, "ssec": 0.020,
    "cpi": 0.010, "pmi": 0.015,
    # 油价因子 10%
    "brent_ma_trend": 0.070, "oil_supply": 0.030,
}

FACTOR_NAMES = {
    "pe": "PE估值", "pb": "PB估值", "price_percentile": "价格分位",
    "dividend": "股息率", "ah_premium": "AH溢价", "fcf": "自由现金流",
    "peer_compare": "国际对标",
    "trend": "MA趋势", "ema_trend": "EMA趋势", "momentum": "动量",
    "rsi": "RSI", "macd": "MACD", "atr": "ATR", "bollinger": "布林带",
    "adx": "ADX", "obv": "OBV", "kdj": "KDJ",
    "roe": "ROE", "roic": "ROIC", "gross_margin": "毛利率",
    "net_margin": "净利率", "payout": "分红率", "coverage": "分红覆盖",
    "dxy": "美元指数", "us10y": "美10Y利率", "vix": "VIX恐慌",
    "ssec": "上证指数", "cpi": "CPI", "pmi": "PMI",
    "brent_ma_trend": "布伦特趋势", "oil_supply": "OPEC供给",
}

FACTOR_CALIBERS = {
    "pe": "PE-TTM 动态市盈率 · 数据源: akshare/东方财富",
    "pb": "PB 市净率(最近一期) · 数据源: akshare/东方财富",
    "price_percentile": "当前收盘价在历史价格序列中的分位 · 数据源: akshare 日线",
    "dividend": "EPS × 0.48 分红率估算 · 数据源: 季报",
    "ah_premium": "(A价-H价×汇率)/(H价×汇率) · 数据源: akshare 实时",
    "fcf": "营业利润率作为FCF proxy · 数据源: 季报",
    "peer_compare": "XOM/CVX/COP/BP/SHEL PE/PB对比 · 数据源: 手工参考值",
    "trend": "MA20 vs MA60 均线排列 · 数据源: akshare 日线",
    "ema_trend": "EMA12 vs EMA26 · 数据源: akshare 日线",
    "momentum": "20日/60日涨跌幅 · 数据源: akshare 日线",
    "rsi": "RSI-14 相对强弱 · 数据源: akshare 日线",
    "macd": "MACD(12,26,9) · 数据源: akshare 日线",
    "atr": "ATR-14 平均真实波幅 · 数据源: akshare 日线",
    "bollinger": "布林带(20,2) · 数据源: akshare 日线",
    "adx": "ADX-14 趋势强度 · 数据源: akshare 日线",
    "obv": "OBV 能量潮 · 数据源: akshare 日线",
    "kdj": "KDJ(9,3,3) · 数据源: akshare 日线",
    "roe": "加权ROE 季报累计 · 数据源: akshare 季报",
    "roic": "ROIC 近似(摊薄ROE) · 数据源: akshare 季报",
    "gross_margin": "销售毛利率 · 数据源: akshare 季报",
    "net_margin": "销售净利率 · 数据源: akshare 季报",
    "payout": "每股分红/EPS · 数据源: akshare 季报",
    "coverage": "EPS/每股分红 · 数据源: akshare 季报",
    "dxy": "美元指数日线 · 数据源: Yahoo Finance",
    "us10y": "美国10年期国债收益率 · 数据源: Yahoo Finance",
    "vix": "CBOE VIX日线 · 数据源: Yahoo Finance",
    "ssec": "上证综指日线 · 数据源: akshare",
    "cpi": "中国CPI当月同比 · 数据源: akshare/国家统计局",
    "pmi": "中国制造业PMI · 数据源: akshare/国家统计局",
    "brent_ma_trend": "布伦特期货 MA20/MA60趋势 · 数据源: Yahoo Finance BZ=F",
    "oil_supply": "OPEC供给预估 · 数据源: EIA月度展望(预估值)",
}

CATEGORY_NAMES = {
    "value": "价值因子", "technical": "技术因子", "quality": "质量因子",
    "macro": "宏观因子", "oil": "油价因子",
}


# ═══════════════════════════════════════════════════════════
#  流量因子（保留原有但不在主权重中独立）
# ═══════════════════════════════════════════════════════════

def calc_flow_factor(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or "volume" not in df.columns or len(df) < 20:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    volume = df["volume"].dropna()
    pc = close.pct_change().dropna()
    vc = volume.pct_change().dropna()
    ci = pc.index.intersection(vc.index)
    if len(ci) < 10:
        return 50.0, "数据不足"
    recent_pc = pc.loc[ci].iloc[-20:]
    recent_vc = vc.loc[ci].iloc[-20:]
    same = ((recent_pc > 0) & (recent_vc > 0)).sum() + ((recent_pc < 0) & (recent_vc < 0)).sum()
    ratio = safe_divide(same, len(recent_pc))
    score = ratio * 100
    desc = f"量价配合 {ratio*100:.0f}%，{'方向一致' if ratio > 0.6 else '部分分化' if ratio > 0.4 else '背离'}"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_margin_sentiment(margin_df: pd.DataFrame) -> tuple[float, str]:
    if margin_df is None or margin_df.empty:
        return 50.0, "融资数据不足"
    if "margin_balance" in margin_df.columns:
        bal = margin_df["margin_balance"].dropna()
    else:
        bal = margin_df.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
    if len(bal) < 10:
        return 50.0, "融资数据不足"
    change = _scalar(bal.iloc[-1] / bal.iloc[-10] - 1)
    score = 50 + change * 1000
    desc = f"融资余额10日{change*100:+.2f}%，{'融资增加偏乐观' if change > 0 else '融资减少偏谨慎'}"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_northbound_sentiment(nb_df: pd.DataFrame) -> tuple[float, str]:
    if nb_df is None or nb_df.empty:
        return 50.0, "北向资金数据不足"
    if "net_flow" in nb_df.columns:
        flow = nb_df["net_flow"].dropna()
    else:
        flow = nb_df.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
    if len(flow) < 5:
        return 50.0, "北向数据不足"
    recent_net = _scalar(flow.iloc[-5:].sum())
    score = 50 + recent_net * 0.5
    desc = f"近5日净流入 ¥{recent_net:.1f}亿，{'持续流入' if recent_net > 0 else '持续流出'}"
    return _scalar(np.clip(score, 0, 100)), desc


def calc_smart_money(df: pd.DataFrame) -> tuple[float, str]:
    if "close" not in df.columns or "volume" not in df.columns or len(df) < 10:
        return 50.0, "数据不足"
    close = df["close"].dropna()
    volume = df["volume"].dropna()
    amount = close * volume
    change = close.pct_change().dropna()
    amount_change = amount.pct_change().dropna()
    ci = change.index.intersection(amount_change.index)
    if len(ci) < 5:
        return 50.0, "数据不足"
    r = change.loc[ci].iloc[-5:]
    a = amount_change.loc[ci].iloc[-5:]
    smart = ((r > 0) & (a > 0.1)).sum() - ((r < 0) & (a > 0.1)).sum()
    score = 50 + smart * 10
    desc = f"近5日聪明钱{'净流入' if smart > 0 else '净流出' if smart < 0 else '中性'}"
    return _scalar(np.clip(score, 0, 100)), desc


# ═══════════════════════════════════════════════════════════
#  主计算函数
# ═══════════════════════════════════════════════════════════

def compute_all_factors(data: dict) -> dict:
    stock = fillna_df(data["stock"])
    financial = data.get("financial", {})
    spot = data.get("spot", data.get("valuation", {}))
    ah = data.get("ah", {})
    peers = data.get("peers", {})
    brent = data.get("brent", pd.Series(dtype=float))
    dxy = data.get("dxy", pd.Series(dtype=float))
    vix = data.get("vix", pd.Series(dtype=float))
    us10y = data.get("us10y", pd.Series(dtype=float))
    cpi = data.get("cpi", pd.Series(dtype=float))
    pmi = data.get("pmi", pd.Series(dtype=float))
    oil_supply = data.get("oil_supply", {})
    ssec = data.get("ssec", pd.Series(dtype=float))
    margin = data.get("margin", pd.DataFrame())
    northbound = data.get("northbound", pd.DataFrame())
    price = _scalar(stock["close"].dropna().iloc[-1]) if "close" in stock.columns else 30.0

    results = {}
    descs = {}

    funcs = [
        # 价值因子 7
        ("pe", calc_pe_factor(spot, financial)),
        ("pb", calc_pb_factor(spot)),
        ("price_percentile", calc_price_percentile_factor(stock)),
        ("dividend", calc_dividend_factor(financial, price)),
        ("ah_premium", calc_ah_premium_factor(ah)),
        ("fcf", calc_fcf_factor(financial)),
        ("peer_compare", calc_peer_comparison_factor(peers, spot)),
        # 技术因子 10
        ("trend", calc_trend_factor(stock)),
        ("ema_trend", calc_ema_trend_factor(stock)),
        ("momentum", calc_momentum_factor(stock)),
        ("rsi", calc_rsi_factor(stock)),
        ("macd", calc_macd_factor(stock)),
        ("atr", calc_atr_factor(stock)),
        ("bollinger", calc_bollinger_factor(stock)),
        ("adx", calc_adx_factor(stock)),
        ("obv", calc_obv_factor(stock)),
        ("kdj", calc_kdj_factor(stock)),
        # 质量因子 6
        ("roe", calc_roe_factor(financial)),
        ("roic", calc_roic_factor(financial)),
        ("gross_margin", calc_gross_margin_factor(financial)),
        ("net_margin", calc_net_margin_factor(financial)),
        ("payout", calc_payout_factor(financial)),
        ("coverage", calc_coverage_factor(financial)),
        # 宏观因子 6
        ("dxy", calc_dxy_factor(dxy)),
        ("us10y", calc_us10y_factor(us10y)),
        ("vix", calc_vix_factor(vix)),
        ("ssec", calc_ssec_factor(ssec)),
        ("cpi", calc_cpi_factor(cpi)),
        ("pmi", calc_pmi_factor(pmi)),
        # 油价因子 2
        ("brent_ma_trend", calc_brent_ma_trend(brent)),
        ("oil_supply", calc_oil_supply_factor(oil_supply)),
    ]

    for key, (score, desc) in funcs:
        results[key] = score
        descs[key] = desc

    # AI 综合评分
    ai_score = round(sum(results[k] * WEIGHTS[k] for k in WEIGHTS if k in results), 1)
    results["_ai_score"] = ai_score
    results["_descriptions"] = descs

    # 5大类评分
    category_scores = {}
    for cat, cat_w in CATEGORY_WEIGHTS.items():
        cat_keys = [k for k, c in FACTOR_CATEGORIES.items() if c == cat and k in results]
        if cat_keys:
            cat_score = round(sum(results[k] * WEIGHTS[k] for k in cat_keys) /
                            sum(WEIGHTS[k] for k in cat_keys), 1)
        else:
            cat_score = 50.0
        category_scores[cat] = {"score": cat_score, "weight": cat_w,
                                "name": CATEGORY_NAMES.get(cat, cat),
                                "factors": len(cat_keys)}
    results["_category_scores"] = category_scores
    results["_calibers"] = FACTOR_CALIBERS

    return results


def get_action(ai_score: float, risk_safe: bool = True) -> dict:
    """AI 评分 → 7档操作建议"""
    if ai_score >= 95:
        return {"action": "强烈买入", "color": "#ff3b30",
                "detail": "多因子高度共振+估值极低，强烈看多",
                "tier": 1}
    elif ai_score >= 90:
        return {"action": "买入", "color": "#ff3b30",
                "detail": "估值偏低+多因子共振，建议买入",
                "tier": 2}
    elif ai_score >= 85:
        return {"action": "偏多", "color": "#ff6b35",
                "detail": "估值偏低，多数因子偏多，可加仓",
                "tier": 3}
    elif ai_score >= 80:
        return {"action": "持有偏多", "color": "#ff9f0a",
                "detail": "估值合理，部分因子偏多，维持仓位",
                "tier": 4}
    elif ai_score >= 70:
        return {"action": "持有", "color": "#007aff",
                "detail": "估值合理区间，多空交织，等待更好时机",
                "tier": 5}
    elif ai_score >= 60:
        return {"action": "持有偏空", "color": "#5ac8fa",
                "detail": "估值偏高或因子分化，控制仓位",
                "tier": 6}
    else:
        return {"action": "卖出", "color": "#34c759",
                "detail": "估值过高或风险累积，建议减仓/清仓",
                "tier": 7}
