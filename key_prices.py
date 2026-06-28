"""
关键价格体系 — 建议买入/减仓/融资买入/止盈/止损价 + 理由
"""
import numpy as np
import pandas as pd


def compute_key_prices(price: float, financial: dict, atr_val: float,
                       bollinger: dict, ma_values: dict, hist_lows: pd.Series = None,
                       ai_score: float = 50.0) -> dict:
    """
    基于当前价格、估值、技术指标、历史低点，计算关键价格体系
    """
    keys = {}
    eps = financial.get("每股收益", 2.5)
    dps = eps * 0.48

    # 1. 建议买入价（基于估值）
    # PE=6x → 6 * EPS
    value_price_pe6 = round(eps * 6, 2)
    # PE=8x → 保守买入
    value_price_pe8 = round(eps * 8, 2)

    boll_mid = bollinger.get("mid", price * 0.95)
    boll_lower = bollinger.get("lower", price * 0.85)

    # 综合买入价：PB低位 + 布林下轨 + PE低估值 取高者（保守）
    suggested_buy = max(value_price_pe8, boll_lower, hist_lows.iloc[-1] if hist_lows is not None and len(hist_lows) > 0 else price * 0.8)
    keys["suggested_buy"] = {
        "price": round(suggested_buy, 2),
        "pct_from_current": round((suggested_buy / price - 1) * 100, 2),
        "reason": f"PE 8x(¥{value_price_pe8}) / 布林下轨(¥{boll_lower:.2f}) / 历史低点 综合保守价",
    }

    # 2. 建议减仓价（PE偏高或技术信号反转）
    suggested_reduce = round(price * 1.20, 2)
    keys["suggested_reduce"] = {
        "price": suggested_reduce,
        "pct_from_current": round((suggested_reduce / price - 1) * 100, 2),
        "reason": "当前价+20%，PE回至15x附近，建议分批减仓锁定利润",
    }

    # 3. 融资买入价（AI高分+低估值时用融资）
    margin_buy = suggested_buy * 0.95  # 融资在更低价格进场
    keys["margin_buy"] = {
        "price": round(margin_buy, 2),
        "pct_from_current": round((margin_buy / price - 1) * 100, 2),
        "reason": f"低于建议买入价5%以下，仅限AI评分≥90时使用融资买入",
    }

    # 4. 止盈价
    tp1 = round(price * 1.15, 2)  # 近端止盈 +15%
    tp2 = round(price * 1.25, 2)  # 远端止盈 +25%
    keys["take_profit_1"] = {
        "price": tp1,
        "pct_from_current": 15.0,
        "reason": "近端止盈，PE回合理区间上沿，减仓1/3",
    }
    keys["take_profit_2"] = {
        "price": tp2,
        "pct_from_current": 25.0,
        "reason": "远端止盈，估值偏高，再减1/3，保留底仓",
    }

    # 5. 止损价
    # ATR-based: price - 2*ATR
    stop_loss_atr = round(price - 2 * atr_val, 2) if atr_val > 0 else round(price * 0.92, 2)
    # 硬止损：-8%
    stop_loss_hard = round(price * 0.92, 2)
    stop_loss = max(stop_loss_atr, stop_loss_hard)
    keys["stop_loss"] = {
        "price": stop_loss,
        "pct_from_current": round((stop_loss / price - 1) * 100, 2),
        "reason": f"2×ATR(¥{stop_loss_atr}) vs 硬止损-8%(¥{stop_loss_hard}) 取高者，跌破即减仓",
    }

    # 6. 股息支撑价（股息率≥5% 对应的价格）
    div_support = round(dps / 0.05, 2) if dps > 0 else 0
    keys["dividend_support"] = {
        "price": div_support,
        "pct_from_current": round((div_support / price - 1) * 100, 2) if div_support > 0 else 0,
        "reason": f"此价格对应股息率5%，股息价值支撑强",
    }

    # 7. 综合操作区间总结
    if price <= suggested_buy:
        zone = "买入区间"
        zone_color = "#ff3b30"
    elif price <= suggested_buy * 1.05:
        zone = "加仓观察区间"
        zone_color = "#ff6b35"
    elif price <= suggested_reduce * 0.85:
        zone = "持有区间"
        zone_color = "#007aff"
    elif price <= suggested_reduce:
        zone = "减仓观察区间"
        zone_color = "#ff9f0a"
    else:
        zone = "减仓区间"
        zone_color = "#34c759"

    keys["_current_zone"] = {"zone": zone, "color": zone_color}
    keys["_current_price"] = price

    return keys
