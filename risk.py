"""
风险控制模块 — 维保比例 = 总资产 / 总负债
扩展: -25%/-30%场景、margin call价格、风险等级
"""
import numpy as np


def stress_test(price: float, shares: int, debt: float) -> dict:
    """
    多场景压力测试: -3%, -5%, -8%, -10%, -15%, -20%, -25%, -30%
    维保比例 = 总资产 / 总负债 = (股价 × 股数) / 负债
    安全阈值: 维保比例 ≥ 3.0x
    """
    scenarios = {}
    total_value = price * shares
    for pct in [-3, -5, -8, -10, -15, -20, -25, -30]:
        new_price = price * (1 + pct / 100)
        new_value = new_price * shares
        loss = total_value - new_value
        loss_pct = (loss / total_value * 100) if total_value > 0 else 0
        margin_ratio = new_value / debt if debt > 0 else float("inf")
        need_margin = 0.0
        if debt > 0 and margin_ratio < 3.0:
            need_margin = max(0, debt * 3.0 - new_value)
        safe = margin_ratio >= 3.0
        scenarios[f"{pct:+d}%"] = {
            "new_price": round(new_price, 2),
            "new_value": round(new_value, 2),
            "loss": round(loss, 2),
            "loss_pct": round(loss_pct, 2),
            "margin_ratio": round(margin_ratio, 2) if debt > 0 else float("inf"),
            "margin_safe": safe,
            "need_margin": round(need_margin, 2),
        }
    return scenarios


def leverage_allowed(ai_score: float, value: float, debt: float) -> bool:
    if ai_score < 90:
        return False
    if debt <= 0:
        return True
    if value > 0 and debt / value > 1.0:
        return False
    return True


def margin_call_price(market_value: float, debt: float, requirement: float = 3.0) -> dict:
    """计算触发 margin call 的股价
    requirement: 最低维保比例要求
    触发价 = debt × requirement / shares
    """
    if debt <= 0:
        return {"trigger_price": 0, "current_price": 0, "margin_call_pct": 0,
                "verdict": "无融资，无margin call风险"}
    # 从 market_value 反推 shares
    # market_value = price * shares, need price from caller
    # 这里返回公式，实际由UI计算
    return {
        "requirement_ratio": requirement,
        "debt": debt,
        "market_value": market_value,
        "formula": f"触发价 = 负债{debt:,.0f} × {requirement}x / 股数",
        "verdict": "需结合股数计算具体触发价",
    }


def calc_margin_call_price(price: float, shares: int, debt: float, requirement: float = 3.0) -> dict:
    """计算具体触发价格"""
    if debt <= 0 or shares <= 0:
        return {"trigger_price": 0, "margin_call_pct": 0,
                "verdict": "无融资或无持仓"}
    trigger = debt * requirement / shares
    pct_from_current = (trigger / price - 1) * 100 if price > 0 else 0
    if trigger < price:
        verdict = f"触发价 ¥{trigger:.2f}，距当前{abs(pct_from_current):.1f}%"
    else:
        verdict = f"已触发！当前价¥{price:.2f} < 触发价¥{trigger:.2f}"
    return {
        "trigger_price": round(trigger, 2),
        "current_price": round(price, 2),
        "margin_call_pct": round(pct_from_current, 2),
        "requirement_ratio": requirement,
        "verdict": verdict,
    }


def risk_level(ai_score: float, margin_ratio: float, stress_ok: bool) -> dict:
    """综合风险等级"""
    if ai_score >= 85 and margin_ratio >= 3.5 and stress_ok:
        return {"level": "低风险", "color": "#34c759",
                "desc": "评分高+维保充裕+压力测试通过，可适度加仓"}
    elif ai_score >= 70 and margin_ratio >= 3.0 and stress_ok:
        return {"level": "中等风险", "color": "#ff9f0a",
                "desc": "评分中等，维保安全，维持仓位"}
    elif margin_ratio < 3.0 or not stress_ok:
        return {"level": "高风险", "color": "#ff3b30",
                "desc": "维保接近预警或压力测试未通过，建议减仓/降杠杆"}
    else:
        return {"level": "极高风险", "color": "#8b0000",
                "desc": "多指标恶化，立即减仓"}


def risk_summary(price: float, shares: int, debt: float, ai_score: float) -> dict:
    total_value = price * shares
    equity = total_value - debt
    leverage_ratio = (debt / equity) if equity > 0 else 0
    st = stress_test(price, shares, debt)
    can_leverage = leverage_allowed(ai_score, total_value, debt)
    all_safe = all(v["margin_safe"] for v in st.values())
    margin_call = calc_margin_call_price(price, shares, debt)
    margin_ratio = total_value / debt if debt > 0 else float("inf")
    rl = risk_level(ai_score, margin_ratio, all_safe)
    return {
        "total_value": round(total_value, 2),
        "equity": round(equity, 2),
        "debt": round(debt, 2),
        "leverage_ratio": round(leverage_ratio, 2),
        "can_leverage": can_leverage,
        "stress_test": st,
        "overall_safe": all_safe,
        "margin_call": margin_call,
        "risk_level": rl,
    }


def scenario_add_position(price: float, shares: int, cost: float, debt: float,
                          add_price: float, add_shares: int, use_debt: bool = False,
                          max_debt: float = 2_000_000.0) -> dict:
    """加仓方案模拟"""
    add_cost = add_price * add_shares
    new_debt = debt + add_cost if use_debt else debt
    if new_debt > max_debt:
        return {"error": f"总负债 ¥{new_debt:,.0f} 超出可用额度 ¥{max_debt:,.0f}"}

    new_total_shares = shares + add_shares
    new_total_cost = cost * shares + add_price * add_shares
    new_avg_cost = new_total_cost / new_total_shares if new_total_shares > 0 else 0
    new_market_value = price * new_total_shares
    new_equity = new_market_value - new_debt
    new_margin_ratio = new_market_value / new_debt if new_debt > 0 else float("inf")
    new_pnl = (price - new_avg_cost) * new_total_shares

    return {
        "add_shares": add_shares,
        "add_price": add_price,
        "add_cost": round(add_cost, 2),
        "use_debt": use_debt,
        "new_total_shares": new_total_shares,
        "new_avg_cost": round(new_avg_cost, 3),
        "new_market_value": round(new_market_value, 2),
        "new_debt": round(new_debt, 2),
        "new_equity": round(new_equity, 2),
        "new_margin_ratio": round(new_margin_ratio, 2),
        "new_pnl": round(new_pnl, 2),
    }
