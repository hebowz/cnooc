"""
仓位管理 — 7档仓位 + 融资利息(3.55%) + 维保三级预警 + 净股息收益
"""
import numpy as np

# (最低AI评分, 仓位权重%, 融资额度¥)
POSITION_TIERS = [
    (95, 1.00, 2_000_000),
    (90, 1.00, 1_000_000),
    (85, 0.90, 500_000),
    (80, 0.80, 0),
    (70, 0.70, 0),
    (60, 0.50, 0),
    (0,  0.20, 0),
]

MARGIN_RATE = 0.0355  # 融资年利率 3.55%


def get_position_tier(ai_score: float) -> dict:
    """AI评分 → 仓位档位"""
    score = float(np.clip(ai_score, 0, 100))
    for min_score, weight, leverage_limit in POSITION_TIERS:
        if score >= min_score:
            return {"ai_score": score, "target_weight": weight,
                    "target_position_pct": weight * 100,
                    "suggested_leverage_limit": leverage_limit,
                    "allow_leverage": leverage_limit > 0}
    return {"ai_score": score, "target_weight": 0.20, "target_position_pct": 20.0,
            "suggested_leverage_limit": 0, "allow_leverage": False}


def calc_position(ai_score: float) -> dict:
    """向后兼容旧接口"""
    return get_position_tier(ai_score)


def calc_margin_metrics(market_value: float, debt: float, rate: float = MARGIN_RATE) -> dict:
    """融资成本计算
    market_value: 当前总市值(price × shares)
    debt: 融资负债总额
    rate: 年利率
    """
    daily_interest = debt * rate / 365 if debt > 0 else 0
    annual_interest = debt * rate if debt > 0 else 0
    margin_ratio = market_value / debt if debt > 0 else float("inf")
    cost_rate = (annual_interest / market_value * 100) if market_value > 0 else 0
    return {
        "market_value": round(market_value, 2),
        "debt": round(debt, 2),
        "annual_interest": round(annual_interest, 2),
        "daily_interest": round(daily_interest, 2),
        "margin_cost_rate_pct": round(cost_rate, 2),
        "margin_ratio": round(margin_ratio, 2) if debt > 0 else float("inf"),
        "margin_rate_annual": rate,
    }


def calc_margin_warnings(market_value: float, debt: float) -> dict:
    """维保比例三级预警
    维保比例 = 总资产 / 总负债 = market_value / debt
    """
    if debt <= 0:
        return {"level": "safe", "ratio": float("inf"), "message": "无融资负债，安全",
                "need_add_funds": 0, "need_reduce_debt": 0}

    ratio = market_value / debt
    if ratio >= 3.5:
        return {"level": "safe", "ratio": round(ratio, 2), "message": "维保充裕，安全",
                "need_add_funds": 0, "need_reduce_debt": 0}
    elif ratio >= 3.2:
        return {"level": "watch", "ratio": round(ratio, 2),
                "message": f"维保{ratio:.1f}x，接近预警线350%，注意监控",
                "need_add_funds": max(0, debt * 3.5 - market_value),
                "need_reduce_debt": 0}
    elif ratio >= 3.0:
        return {"level": "warning", "ratio": round(ratio, 2),
                "message": f"维保{ratio:.1f}x，低于320%建议减仓线",
                "need_add_funds": max(0, debt * 3.2 - market_value),
                "need_reduce_debt": max(0, debt - market_value / 3.2)}
    else:
        return {"level": "danger", "ratio": round(ratio, 2),
                "message": f"维保{ratio:.1f}x，低于300%立即减仓线！",
                "need_add_funds": max(0, debt * 3.0 - market_value),
                "need_reduce_debt": max(0, debt - market_value / 3.0)}


def calc_dividend_net_yield(annual_dividend: float, margin_interest: float, tax_rate: float = 0.10,
                            market_value: float = 1.0) -> dict:
    """净股息收益 = 股息总收入 - 融资利息 - 股息税"""
    net_div = annual_dividend * (1 - tax_rate) - margin_interest
    net_yield = (net_div / market_value * 100) if market_value > 0 else 0
    gross_yield = (annual_dividend / market_value * 100) if market_value > 0 else 0
    return {
        "annual_dividend_gross": round(annual_dividend, 2),
        "dividend_tax": round(annual_dividend * tax_rate, 2),
        "annual_dividend_net": round(annual_dividend * (1 - tax_rate), 2),
        "margin_interest_annual": round(margin_interest, 2),
        "net_income": round(net_div, 2),
        "net_yield_pct": round(net_yield, 2),
        "gross_yield_pct": round(gross_yield, 2),
        "positive_carry": net_div > 0,
    }


def positive_carry_check(dividend_yield_pct: float, margin_rate: float = MARGIN_RATE) -> dict:
    """正利差判断：股息率 > 融资利率 → 正利差(套息)"""
    net_spread = dividend_yield_pct - margin_rate * 100
    return {
        "dividend_yield_pct": round(dividend_yield_pct, 2),
        "margin_rate_pct": round(margin_rate * 100, 2),
        "spread_pct": round(net_spread, 2),
        "is_positive_carry": net_spread > 0,
        "verdict": "正利差，融资持股可覆盖利息" if net_spread > 0 else (
            "负利差，融资成本高于股息收益" if net_spread < 0 else "利差平衡"),
    }


def expected_dividend(shares: int, price: float, financial: dict) -> dict:
    """预期股息收入"""
    eps = financial.get("每股收益", 0)
    if eps <= 0:
        eps = 2.57
    dps = eps * 0.48
    total_div = shares * dps
    div_yield = (dps / price * 100) if price > 0 else 0
    return {
        "dps_estimated": round(dps, 3),
        "total_dividend": round(total_div, 2),
        "div_yield_pct": round(div_yield, 2),
    }


def calc_new_cost(old_cost: float, old_shares: int, new_price: float, new_shares: int) -> float:
    total_cost = old_cost * old_shares + new_price * new_shares
    total_shares = old_shares + new_shares
    return total_cost / total_shares if total_shares > 0 else old_cost
