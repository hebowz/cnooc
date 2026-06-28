"""
交易日志 — JSON文件存储 + 统计 + AI复盘建议
"""
import json
import os
from datetime import datetime

JOURNAL_FILE = os.path.join(os.path.dirname(__file__), "trade_journal.json")


def load_journal() -> list:
    """加载交易日志"""
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_journal(journal: list):
    """保存交易日志"""
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)


def add_trade(action: str, price: float, shares: int, reason: str = "",
              ai_score: float = 0, notes: str = "") -> dict:
    """记录一笔交易
    action: 'buy' / 'sell' / 'add_margin' / 'reduce_margin' / 'adjust'
    """
    journal = load_journal()
    trade = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "price": round(price, 3),
        "shares": shares,
        "amount": round(price * shares, 2),
        "ai_score": round(ai_score, 1),
        "reason": reason,
        "notes": notes,
    }
    journal.append(trade)
    save_journal(journal)
    return trade


def compute_stats(journal: list = None) -> dict:
    """交易统计"""
    if journal is None:
        journal = load_journal()
    if not journal:
        return {"total_trades": 0, "message": "暂无交易记录"}

    buys = [t for t in journal if t["action"] in ("buy", "add_margin")]
    sells = [t for t in journal if t["action"] in ("sell", "reduce_margin")]

    total_buy_amount = sum(t["amount"] for t in buys)
    total_sell_amount = sum(t["amount"] for t in sells)
    total_buy_shares = sum(t["shares"] for t in buys)
    total_sell_shares = sum(t["shares"] for t in sells)

    avg_buy_price = total_buy_amount / total_buy_shares if total_buy_shares > 0 else 0
    avg_sell_price = total_sell_amount / total_sell_shares if total_sell_shares > 0 else 0

    net_pnl = total_sell_amount - total_buy_amount + (total_buy_shares - total_sell_shares) * avg_buy_price

    return {
        "total_trades": len(journal),
        "buy_count": len(buys), "sell_count": len(sells),
        "total_buy_amount": round(total_buy_amount, 2),
        "total_sell_amount": round(total_sell_amount, 2),
        "total_buy_shares": total_buy_shares,
        "total_sell_shares": total_sell_shares,
        "avg_buy_price": round(avg_buy_price, 3),
        "avg_sell_price": round(avg_sell_price, 3),
        "net_pnl_estimate": round(net_pnl, 2),
    }


def generate_ai_review(journal: list = None) -> str:
    """基于交易记录生成AI复盘建议"""
    stats = compute_stats(journal)
    if stats["total_trades"] == 0:
        return "暂无交易记录，无法生成复盘建议。"

    lines = ["=== AI 复盘建议 ===", f"共 {stats['total_trades']} 笔交易"]
    if stats["buy_count"] > stats["sell_count"]:
        lines.append("买入次数多于卖出，当前偏建仓阶段")
    elif stats["sell_count"] > stats["buy_count"]:
        lines.append("卖出次数多于买入，当前偏减仓阶段")
    else:
        lines.append("买卖次数均衡")

    if stats["avg_sell_price"] > stats["avg_buy_price"] > 0:
        lines.append(f"平均卖价 ¥{stats['avg_sell_price']:.3f} > 买价 ¥{stats['avg_buy_price']:.3f}，交易正利差")
    elif stats["avg_buy_price"] > 0:
        lines.append("平均买卖差价偏小或为负，注意交易成本控制")

    if stats["total_trades"] > 10:
        lines.append("交易频繁，建议减少交易频率，降低摩擦成本")
    elif stats["total_trades"] < 3:
        lines.append("交易记录较少，建议持续跟踪并记录每次操作")

    lines.append("建议: 每笔交易记录加入复盘笔记，定期回顾盈亏原因")
    return "\n".join(lines)
