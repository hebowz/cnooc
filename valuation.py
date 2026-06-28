"""
估值分析 — PE/PB/股息率历史分位 + 国际油企对标 + 综合价值评分
"""
import numpy as np
import pandas as pd


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


def pe_percentile_rank(pe_history: pd.Series, current_pe: float) -> dict:
    """PE历史分位 — 越低越好，低位=低估"""
    if pe_history is None or len(pe_history.dropna()) < 30:
        return {"percentile": 0.5, "score": 50.0, "desc": "PE历史数据不足", "current": current_pe}
    s = pe_history.dropna()
    pct = _scalar((s < current_pe).sum() / len(s))
    score = round((1 - pct) * 100, 1)
    if pct < 0.15:
        desc = f"PE处于历史{pct*100:.0f}%分位，极度低估"
    elif pct < 0.30:
        desc = f"PE处于历史{pct*100:.0f}%分位，显著低估"
    elif pct < 0.50:
        desc = f"PE处于历史{pct*100:.0f}%分位，偏低估值"
    elif pct < 0.70:
        desc = f"PE处于历史{pct*100:.0f}%分位，估值中枢"
    elif pct < 0.85:
        desc = f"PE处于历史{pct*100:.0f}%分位，偏高估值"
    else:
        desc = f"PE处于历史{pct*100:.0f}%分位，高估区域"
    return {"percentile": round(pct, 3), "score": _scalar(np.clip(score, 0, 100)),
            "desc": desc, "current": current_pe,
            "min_pe": round(float(s.min()), 2), "max_pe": round(float(s.max()), 2),
            "median_pe": round(float(s.median()), 2)}


def pb_percentile_rank(pe_history: pd.Series, current_pb: float) -> dict:
    """PB历史分位（复用PE历史Series结构，实际使用时传入PB序列）"""
    if pe_history is None or len(pe_history.dropna()) < 30:
        return {"percentile": 0.5, "score": 50.0, "desc": "PB历史数据不足", "current": current_pb}
    s = pe_history.dropna()
    pct = _scalar((s < current_pb).sum() / len(s))
    score = round((1 - pct) * 100, 1)
    if pct < 0.15:
        desc = f"PB处于历史{pct*100:.0f}%分位，极度低估"
    elif pct < 0.30:
        desc = f"PB处于历史{pct*100:.0f}%分位，显著低估"
    elif pct < 0.50:
        desc = f"PB处于历史{pct*100:.0f}%分位，偏低估值"
    elif pct < 0.70:
        desc = f"PB处于历史{pct*100:.0f}%分位，估值中枢"
    elif pct < 0.85:
        desc = f"PB处于历史{pct*100:.0f}%分位，偏高估值"
    else:
        desc = f"PB处于历史{pct*100:.0f}%分位，高估区域"
    return {"percentile": round(pct, 3), "score": _scalar(np.clip(score, 0, 100)),
            "desc": desc, "current": current_pb,
            "min_pb": round(float(s.min()), 2), "max_pb": round(float(s.max()), 2),
            "median_pb": round(float(s.median()), 2)}


def dividend_yield_percentile(div_history: pd.DataFrame, current_yield: float) -> dict:
    """股息率历史分位 — 越高越好"""
    if div_history is None or div_history.empty:
        return {"percentile": 0.5, "score": 50.0, "desc": "股息历史数据不足"}
    # 尝试从 div_history 提取每股分红列
    try:
        div_col = next(c for c in div_history.columns if "分红" in str(c) or "股利" in str(c))
        divs = div_history[div_col].dropna().astype(float)
    except Exception:
        divs = pd.Series([1.0, 1.1, 1.2], dtype=float)  # fallback
    if len(divs) < 3:
        return {"percentile": 0.5, "score": 50.0, "desc": "股息历史记录不足"}
    # 历史股息率(基于固定价格作proxy)
    avg_price = 25.0
    hist_yields = divs / avg_price * 100
    pct = _scalar((hist_yields < current_yield).sum() / len(hist_yields))
    score = round(pct * 100, 1)
    desc = f"股息率处于历史{pct*100:.0f}%分位，{'高股息' if pct > 0.6 else '中等' if pct > 0.3 else '偏低'}"
    return {"percentile": round(pct, 3), "score": _scalar(np.clip(score, 0, 100)),
            "desc": desc, "avg_hist_div": round(float(divs.mean()), 3),
            "max_hist_div": round(float(divs.max()), 3)}


def peer_comparison(peers: dict, current_pe: float, current_pb: float) -> dict:
    """国际油企对标排名 — 越低越优"""
    peer_pe_list = []
    peer_pb_list = []
    details = []
    # 对标参考值 (pe/pb 需手工设定，chart API不提供基本面)
    reference_values = {
        "XOM": {"pe": 13.2, "pb": 2.1},
        "CVX": {"pe": 14.5, "pb": 1.9},
        "COP": {"pe": 12.0, "pb": 2.8},
        "BP": {"pe": 7.5, "pb": 1.3},
        "SHEL": {"pe": 8.3, "pb": 1.4},
    }
    for ticker, info in peers.items():
        if ticker.startswith("_"):
            continue
        if not isinstance(info, dict):
            continue
        ref = reference_values.get(ticker, {})
        pe_val = ref.get("pe", None)
        pb_val = ref.get("pb", None)
        if pe_val:
            peer_pe_list.append(pe_val)
        if pb_val:
            peer_pb_list.append(pb_val)
        details.append({"ticker": ticker, "name": info.get("name", ticker),
                        "pe": pe_val, "pb": pb_val})

    all_pe = peer_pe_list + [current_pe]
    all_pb = peer_pb_list + [current_pb]

    pe_rank = sum(1 for p in sorted(all_pe) if p < current_pe) + 1
    pb_rank = sum(1 for p in sorted(all_pb) if p < current_pb) + 1

    pe_percentile_peer = (pe_rank - 1) / max(1, len(all_pe) - 1)
    pb_percentile_peer = (pb_rank - 1) / max(1, len(all_pb) - 1)

    score = round((1 - pe_percentile_peer) * 50 + (1 - pb_percentile_peer) * 50)
    desc = f"全球油企中PE排第{pe_rank}/{len(all_pe)}，PB排第{pb_rank}/{len(all_pb)}"

    return {"pe_rank": pe_rank, "pb_rank": pb_rank, "total": len(all_pe),
            "score": _scalar(np.clip(score, 0, 100)), "desc": desc,
            "peers": details,
            "median_pe": round(float(np.median(peer_pe_list)), 1) if peer_pe_list else None,
            "median_pb": round(float(np.median(peer_pb_list)), 2) if peer_pb_list else None}


def compute_value_score(spot: dict, financial: dict, pe_history: pd.Series,
                        pb_history: pd.Series, div_history: pd.DataFrame,
                        peers: dict) -> dict:
    """综合价值评分 + 理由汇总"""
    pe = spot.get("pe", 0)
    pb = spot.get("pb", 0)
    price = spot.get("price", 0)

    pe_pct = pe_percentile_rank(pe_history, pe) if pe_history is not None and pe > 0 else None
    pb_pct = pb_percentile_rank(pb_history, pb) if pb_history is not None and pb > 0 else None

    eps = financial.get("每股收益", 0)
    dps = eps * 0.48
    div_yield = (dps / price * 100) if price > 0 and dps > 0 else 0
    div_pct = dividend_yield_percentile(div_history, div_yield) if div_history is not None else None

    peer = peer_comparison(peers, pe, pb) if peers is not None else None

    reasons = []
    if pe_pct:
        reasons.append(pe_pct["desc"])
    if pb_pct:
        reasons.append(pb_pct["desc"])
    if div_pct:
        reasons.append(div_pct["desc"])
    if peer:
        reasons.append(peer["desc"])

    composite = 0
    w = 0
    for item, weight in [(pe_pct, 0.35), (pb_pct, 0.25), (div_pct, 0.20), (peer, 0.20)]:
        if item and item.get("score") is not None:
            composite += item["score"] * weight
            w += weight
    value_score = round(composite / w) if w > 0 else 50.0

    return {"value_score": value_score, "reasons": reasons,
            "pe_percentile": pe_pct, "pb_percentile": pb_pct,
            "dividend_yield_pct": round(div_yield, 2),
            "dividend_percentile": div_pct, "peer_comparison": peer}
