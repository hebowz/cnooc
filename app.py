"""
中国海油 AI 量化交易决策系统 v6 — 12板块单页滚动 · 价值投资导向
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from data import get_all_data
from utils import fillna_df
from factors import (compute_all_factors, FACTOR_NAMES, WEIGHTS, get_action,
                     CATEGORY_WEIGHTS, CATEGORY_NAMES, FACTOR_CATEGORIES, FACTOR_CALIBERS)
from portfolio import (get_position_tier, expected_dividend, calc_new_cost,
                       calc_margin_metrics, calc_margin_warnings, calc_dividend_net_yield,
                       positive_carry_check, MARGIN_RATE, calc_position)
from risk import risk_summary, scenario_add_position, margin_call_price, risk_level, stress_test
from strategies.technical import compute_all_technical, compute_all_technical_extended
from valuation import compute_value_score
from key_prices import compute_key_prices
from arbitrage import fetch_ah_history, compute_ah_percentile, ah_arbitrage_signal
from journal import load_journal, add_trade, compute_stats, generate_ai_review
from backtest import run_enhanced_backtest, compare_margin_vs_nomargin

# ─── 颜色：中国市场红涨绿跌 ─────────────────────────────
RED = "#ff3b30"
GREEN = "#34c759"
BLUE = "#007aff"
GRAY = "#86868b"
ORANGE = "#ff9f0a"
DARK = "#1d1d1f"

# ─── Page config ──────────────────────────────────────────
st.set_page_config(page_title="AI 价值量化 v6", page_icon="⌘", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    .main { background-color: #f5f5f7; }
    .apple-card {
        background: #fff; border-radius: 18px; padding: 18px 20px; margin: 4px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .apple-label {
        font-size: 12px; font-weight: 500; color: #86868b;
        letter-spacing: 0.03em; text-transform: uppercase; margin-bottom: 4px;
    }
    .apple-num {
        font-size: 32px; font-weight: 600; color: #1d1d1f; line-height: 1.1;
        font-feature-settings: "tnum";
    }
    .section-title {
        font-size: 18px; font-weight: 600; color: #1d1d1f; margin: 24px 0 8px 0;
        padding-bottom: 8px; border-bottom: 2px solid #e5e5ea;
    }
    .score-bar-bg { background: #e5e5ea; border-radius: 5px; height: 4px; margin-top: 4px; }
    .score-bar-fill { border-radius: 5px; height: 4px; }
    .conclusion-card {
        background: linear-gradient(135deg, #1d1d1f 0%, #2d2d2f 100%);
        border-radius: 22px; padding: 28px 32px; color: #fff; margin: 8px 0;
    }
    .data-source {
        font-size: 9px; color: #aeaeb2; margin-top: 2px; text-align: center;
    }
    .caliber-text {
        font-size: 10px; color: #aeaeb2; line-height: 1.4;
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stMetricValue"] { font-size: 24px; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 11px; color: #86868b; font-weight: 500; }
    [data-testid="stMetricDelta"] { font-size: 12px; }
    [data-testid="stCaption"] { font-size: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────
for k, v in [("shares", 0), ("cost", 0.0), ("debt", 0.0), ("max_debt", 2_000_000.0),
             ("symbol", "600938")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─── 辅助函数 ────────────────────────────────────────────
def _src(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return getattr(obj, 'attrs', {}).get('source', '')
    if isinstance(obj, dict):
        return obj.get('_source', '')
    return ''

def _updated(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return getattr(obj, 'attrs', {}).get('updated', '')
    if isinstance(obj, dict):
        return obj.get('_updated', '')
    return ''

def _caliber(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return getattr(obj, 'attrs', {}).get('caliber', '')
    if isinstance(obj, dict):
        return obj.get('_caliber', '')
    return ''

def price_color(v):
    """红涨绿跌"""
    return RED if v >= 0 else GREEN


# ═══════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 股票选择")
    SYMBOL = st.text_input("股票代码", value=st.session_state["symbol"],
                           max_chars=6, placeholder="600938").strip() or "600938"
    if SYMBOL != st.session_state["symbol"]:
        st.session_state["symbol"] = SYMBOL
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### 持仓参数")
    st.session_state["shares"] = st.number_input("持股数量", value=st.session_state["shares"], step=1000, format="%d")
    st.session_state["cost"] = st.number_input("持仓成本 ¥", value=st.session_state["cost"], step=0.001, format="%.3f")
    st.session_state["debt"] = st.number_input("总负债 ¥", value=st.session_state["debt"], step=1000.0, format="%.2f",
                                               help="融资余额 + 其他借款")
    st.session_state["max_debt"] = st.number_input("可用融资额度 ¥", value=st.session_state["max_debt"], step=100000.0, format="%.2f")

    st.divider()
    st.markdown("### 数据源")
    if "data" in st.session_state:
        d = st.session_state["data"]
        for label, key in [("股价", "spot"), ("财务", "financial"), ("布伦特", "brent"),
                           ("美元", "dxy"), ("VIX", "vix"), ("美10Y", "us10y"),
                           ("CPI", "cpi"), ("PMI", "pmi"), ("上证", "ssec"),
                           ("AH溢价", "ah")]:
            st.caption(f"{label}: {_src(d.get(key, {}))}")
        st.caption(f"更新: {_updated(d.get('spot')) or datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ═══════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_data(symbol: str):
    return get_all_data(symbol)

with st.spinner(f"加载 {SYMBOL} 数据中..."):
    data = load_data(SYMBOL)
st.session_state["data"] = data

stock = fillna_df(data["stock"])
spot = data.get("spot", {})
fin = data.get("financial", {})
ah = data.get("ah", {})

price = float(spot.get("price", 0)) or float(stock["close"].dropna().iloc[-1])
prev_price = float(stock["close"].dropna().iloc[-2]) if len(stock["close"].dropna()) > 1 else price
change = price - prev_price
change_pct = (change / prev_price * 100) if prev_price else 0
pe = spot.get("pe", 0) or 0
pb = spot.get("pb", 0) or 0
stock_name = spot.get("name", SYMBOL)

factors = compute_all_factors(data)
descs = factors.pop("_descriptions", {})
ai_score = factors.pop("_ai_score", 50.0)
cat_scores = factors.pop("_category_scores", {})
calibers = factors.pop("_calibers", {})

tech = compute_all_technical(stock)
tech_ext = compute_all_technical_extended(stock)

action = get_action(ai_score)

# 估值分析
value_score = compute_value_score(spot, fin, data.get("pe_pb_history"),
                                   data.get("pb_history"), data.get("dividend_history"),
                                   data.get("peers"))

# ─── 持仓计算 ──────────────────────────────────────────
S = int(st.session_state["shares"])
C = float(st.session_state["cost"])
D = float(st.session_state["debt"])
MD = float(st.session_state["max_debt"])
mv = price * S
cost_total = C * S
pnl = mv - cost_total if S > 0 else 0.0
pnl_pct = (price / C - 1) * 100 if C > 0 else 0.0
equity = mv - D
margin_ratio = mv / D if D > 0 else float("inf")
risk = risk_summary(price, S, D, ai_score)
div_info = expected_dividend(S, price, fin)
pos_tier = get_position_tier(ai_score)
margin_m = calc_margin_metrics(mv, D)
margin_warn = calc_margin_warnings(mv, D)

eps = fin.get("每股收益", 0) or 2.57
roe = fin.get("加权净资产收益率", 0) or 15.64
dps_est = eps * 0.48
div_yield = (dps_est / price * 100) if price > 0 else 0

# 宏观快照
brent_s = data.get("brent", pd.Series())
brent_last = float(brent_s.dropna().iloc[-1]) if len(brent_s.dropna()) > 0 else 0
dxy_s = data.get("dxy", pd.Series())
dxy_last = float(dxy_s.dropna().iloc[-1]) if len(dxy_s.dropna()) > 0 else 0
vix_s = data.get("vix", pd.Series())
vix_last = float(vix_s.dropna().iloc[-1]) if len(vix_s.dropna()) > 0 else 0
us10y_s = data.get("us10y", pd.Series())
us10y_last = float(us10y_s.dropna().iloc[-1]) if len(us10y_s.dropna()) > 0 else 0
cpi_s = data.get("cpi", pd.Series())
cpi_last = float(cpi_s.dropna().iloc[-1]) if len(cpi_s.dropna()) > 0 else 0
pmi_s = data.get("pmi", pd.Series())
pmi_last = float(pmi_s.dropna().iloc[-1]) if len(pmi_s.dropna()) > 0 else 0
ssec_s = data.get("ssec", pd.Series())
ssec_last = float(ssec_s.dropna().iloc[-1]) if len(ssec_s.dropna()) > 0 else 0

# 关键价格
key_prices = compute_key_prices(price, fin, tech["atr"], tech["bollinger"],
                                 tech["ema"], ai_score=ai_score)

# A/H套利
ah_hist = fetch_ah_history(SYMBOL, ah.get("h_code", "00883"))
ah_pct = compute_ah_percentile(ah.get("premium", 42), ah_hist)
ah_sig = ah_arbitrage_signal(ah.get("premium", 42), ah_pct["percentile"])

# 回测
bt_signals = pd.Series(0.7, index=stock.index)  # 默认70%仓位
bt = run_enhanced_backtest(stock, bt_signals, margin_pct=0.0)
bt_cmp = compare_margin_vs_nomargin(stock, bt_signals)

# 交易日志
journal = load_journal()
journal_stats = compute_stats(journal)


# ═══════════════════════════════════════════════════════════
#  标题行
# ═══════════════════════════════════════════════════════════

st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
     padding:6px 0 12px 0;border-bottom:1px solid #e5e5ea;margin-bottom:16px;">
    <div style="font-size:20px;font-weight:700;color:#1d1d1f;">
        {stock_name} <span style="font-weight:400;color:#86868b;">AI 价值量化 v6</span>
    </div>
    <div style="font-size:12px;color:#aeaeb2;">
        {datetime.now().strftime("%Y-%m-%d %H:%M")} · {SYMBOL}.SH · 32因子 · 5大类别
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  面板 A — 仪表盘首页
# ═══════════════════════════════════════════════════════════

st.markdown('<div class="section-title">⌘ 仪表盘 · 核心决策</div>', unsafe_allow_html=True)

cc1, cc2, cc3 = st.columns([2.5, 1, 1])
with cc1:
    st.markdown(f"""
    <div class="conclusion-card">
        <div style="font-size:13px;color:#aeaeb2;margin-bottom:4px;">AI 综合评分 · 5大类加权</div>
        <div style="font-size:48px;font-weight:700;color:#fff;">{action['action']}</div>
        <div style="font-size:15px;color:#aeaeb2;margin-top:6px;">{action['detail']}</div>
        <div style="font-size:12px;color:#86868b;margin-top:8px;">
            第{action['tier']}档 · AI评分 {ai_score:.0f}/100
        </div>
    </div>
    """, unsafe_allow_html=True)

with cc2:
    chg_color = RED if change >= 0 else GREEN
    st.markdown(f"""
    <div class="apple-card" style="text-align:center;">
        <div class="apple-label">{SYMBOL} 实时股价</div>
        <div class="apple-num" style="color:{chg_color};">¥ {price:.2f}</div>
        <div style="font-size:16px;font-weight:600;color:{chg_color};">{change:+.2f} ({change_pct:+.2f}%)</div>
        <div class="data-source">数据源: {_src(spot)} · 口径: 交易日盘中实时</div>
    </div>
    """, unsafe_allow_html=True)

with cc3:
    pnl_color = RED if pnl >= 0 else GREEN
    st.markdown(f"""
    <div class="apple-card" style="text-align:center;">
        <div class="apple-label">浮动盈亏</div>
        <div class="apple-num" style="color:{pnl_color};">¥ {pnl:+,.0f}</div>
        <div style="font-size:16px;font-weight:600;color:{pnl_color};">{pnl_pct:+.2f}%</div>
        <div style="margin-top:6px;font-size:11px;color:#86868b;">
            成本 ¥{C:.3f} · 股息 ¥{div_info['total_dividend']:,.0f} ({div_info['div_yield_pct']:.2f}%)
        </div>
        <div style="font-size:11px;color:#86868b;">
            维保比 {"{:.1f}x".format(margin_ratio) if margin_ratio != float('inf') else 'N/A'}
            · 仓位 {pos_tier['target_position_pct']:.0f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# 5大类评分条
st.markdown('<div style="margin-top:12px;">', unsafe_allow_html=True)
cat_cols = st.columns(5)
for i, (cat_key, cat_info) in enumerate(cat_scores.items()):
    s = cat_info["score"]
    w = cat_info["weight"]
    bc = RED if s >= 70 else (ORANGE if s >= 50 else GREEN)
    with cat_cols[i]:
        st.markdown(f"""
        <div class="apple-card" style="text-align:center;padding:12px 8px;">
            <div class="apple-label">{cat_info['name']} · {w:.0%}</div>
            <div style="font-size:24px;font-weight:600;color:{bc};">{s:.0f}</div>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:{s}%;background:{bc};"></div></div>
            <div style="font-size:10px;color:#aeaeb2;">{cat_info['factors']}个因子</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.caption(f"AI 评分 = Σ(32因子 × 权重) = **{ai_score:.1f}/100** · 价值35% / 技术30% / 质量15% / 宏观10% / 油价10%")


# ═══════════════════════════════════════════════════════════
#  面板 B — 价值分析
# ═══════════════════════════════════════════════════════════

with st.expander("📊 价值分析 · PE/PB/股息率历史分位 + 国际对标", expanded=True):
    val1, val2, val3, val4, val5, val6 = st.columns(6)
    with val1:
        st.metric("PE（市盈率）", f"{pe:.1f}x" if pe else "—",
                  help="PE-TTM 动态市盈率")
    with val2:
        st.metric("PB（市净率）", f"{pb:.2f}x" if pb else "—",
                  help="市净率 = 股价 / 每股净资产")
    with val3:
        st.metric("每股收益", f"¥{eps:.2f}",
                  help="摊薄每股收益（最近一期季报累计值, 非年化）")
    with val4:
        st.metric("ROE", f"{roe:.1f}%",
                  help="加权净资产收益率（最近一期报告）")
    with val5:
        st.metric("预估股息率", f"{div_yield:.2f}%",
                  help="EPS × 48% 分红率 / 股价")
    with val6:
        st.metric("AH 溢价", f"{ah.get('premium', 0):.1f}%",
                  help="AH溢价 = (A价 - H价×0.92) / (H价×0.92) × 100%")

    # PE分位
    pe_pct = value_score.get("pe_percentile", {})
    pb_pct = value_score.get("pb_percentile", {})
    if pe_pct:
        ph1, ph2 = st.columns(2)
        with ph1:
            st.caption(f"PE分位: {pe_pct.get('desc', '')} | PE范围 [{pe_pct.get('min_pe', '?')}, {pe_pct.get('max_pe', '?')}] | 中位数 {pe_pct.get('median_pe', '?')}")
        with ph2:
            st.caption(f"PB分位: {pb_pct.get('desc', '')} | PB范围 [{pb_pct.get('min_pb', '?')}, {pb_pct.get('max_pb', '?')}] | 中位数 {pb_pct.get('median_pb', '?')}")

    # 国际对标
    peer = value_score.get("peer_comparison", {})
    if peer and peer.get("peers"):
        st.caption(f"国际油企对标: {peer.get('desc', '')} | 同业PE中位数 {peer.get('median_pe', '?')}x | PB中位数 {peer.get('median_pb', '?')}x")
        peer_cols = st.columns(len(peer["peers"]))
        for i, p in enumerate(peer["peers"]):
            with peer_cols[i]:
                st.metric(p["name"], f"PE {p.get('pe', '?')}x", delta=f"PB {p.get('pb', '?')}x")

    st.caption(f"数据口径: PE-TTM · PB(最近一期) · EPS/ROE(季报累计值,非年化) · 股息率(EPS×分红率/股价) · AH溢价(CNY/HKD≈0.92) | 数据源: {_src(spot)} / {_src(fin)} / {_src(ah)} | 更新: {_updated(spot)}")


# ═══════════════════════════════════════════════════════════
#  面板 C — 盈利能力
# ═══════════════════════════════════════════════════════════

with st.expander("💰 盈利能力 · ROE/ROIC/毛利率/净利率/FCF/分红率/覆盖率", expanded=False):
    prof1, prof2, prof3, prof4, prof5, prof6, prof7 = st.columns(7)
    with prof1:
        st.metric("ROE", f"{fin.get('加权净资产收益率', 0):.1f}%", help="加权净资产收益率 · 季报累计")
    with prof2:
        st.metric("ROIC", f"{fin.get('roic', 0):.1f}%", help="ROIC近似(摊薄ROE) · 季报")
    with prof3:
        st.metric("毛利率", f"{fin.get('毛利率', 0):.1f}%", help="销售毛利率 · 季报")
    with prof4:
        st.metric("净利率", f"{fin.get('净利率', 0):.1f}%", help="销售净利率 · 季报")
    with prof5:
        st.metric("FCF/营收", f"{fin.get('fcf_ratio', 0):.1f}%", help="营业利润率作为FCF proxy")
    with prof6:
        st.metric("分红率", "48%", help="近3年稳定分红率约48%")
    with prof7:
        coverage = eps / (eps * 0.48) if eps > 0 else 2.08
        st.metric("分红覆盖", f"{coverage:.1f}x", help="EPS / 每股分红")

    st.caption(f"数据口径: 所有盈利能力指标均为最近一期季报累计值，非年化/TTM | 数据源: {_src(fin)} | 更新: {_updated(fin)}")


# ═══════════════════════════════════════════════════════════
#  面板 D — 油价与宏观
# ═══════════════════════════════════════════════════════════

with st.expander("🌐 油价与宏观 · Brent趋势/DXY/US10Y/VIX/CPI/PMI", expanded=False):
    mac1, mac2, mac3, mac4 = st.columns(4)
    with mac1:
        st.metric("布伦特原油", f"${brent_last:.2f}",
                  delta=f"因子 {factors.get('brent_ma_trend', 50):.0f}/100",
                  help=_caliber(data.get("brent")))
    with mac2:
        st.metric("美元指数", f"{dxy_last:.2f}",
                  delta=f"因子 {factors.get('dxy', 50):.0f}/100",
                  help=_caliber(data.get("dxy")))
    with mac3:
        st.metric("美国10Y", f"{us10y_last:.2f}%",
                  delta=f"因子 {factors.get('us10y', 50):.0f}/100",
                  help=_caliber(data.get("us10y")))
    with mac4:
        st.metric("VIX 恐慌", f"{vix_last:.2f}",
                  delta=f"因子 {factors.get('vix', 50):.0f}/100",
                  help=_caliber(data.get("vix")))

    mac5, mac6, mac7 = st.columns(3)
    with mac5:
        st.metric("上证指数", f"{ssec_last:.0f}",
                  delta=f"因子 {factors.get('ssec', 50):.0f}/100",
                  help=_caliber(data.get("ssec")))
    with mac6:
        st.metric("中国CPI", f"{cpi_last:.1f}%",
                  delta=f"因子 {factors.get('cpi', 50):.0f}/100",
                  help=_caliber(data.get("cpi")))
    with mac7:
        st.metric("制造业PMI", f"{pmi_last:.1f}",
                  delta=f"因子 {factors.get('pmi', 50):.0f}/100",
                  help=_caliber(data.get("pmi")))

    # OPEC供给
    oil_sup = data.get("oil_supply", {})
    if oil_sup:
        st.caption(f"OPEC供给: {oil_sup.get('total', '?')} 万桶/日 ({oil_sup.get('trend', '?')}) | {oil_sup.get('note', '')}")

    st.caption(f"数据源: {_src(data.get('brent'))} / {_src(data.get('dxy'))} / {_src(data.get('us10y'))} / {_src(data.get('vix'))} / {_src(data.get('ssec'))} / {_src(data.get('cpi'))} / {_src(data.get('pmi'))} | 更新: {_updated(data.get('brent'))}")


# ═══════════════════════════════════════════════════════════
#  面板 E — 技术分析
# ═══════════════════════════════════════════════════════════

with st.expander("📈 技术分析 · MACD/KDJ/RSI/ADX/ATR/布林/OBV + ICT/SMC + 多周期", expanded=False):
    st.markdown("##### 经典指标")
    tm = tech["macd"]
    tk = tech["kdj"]
    tb = tech["bollinger"]
    t1, t2, t3, t4, t5, t6, t7 = st.columns(7)
    with t1:
        st.metric("MACD", f"DIF {tm['DIF']:.3f}", delta=f"Hist {tm['hist']:.3f}")
    with t2:
        st.metric("KDJ", f"K {tk['K']:.1f}", delta=f"J {tk['J']:.1f}")
    with t3:
        rsi14 = float(tech["ema"].get("EMA14", 50))
        st.metric("RSI-14", f"{factors.get('rsi', 50):.0f}")
    with t4:
        st.metric("ADX", f"{tech_ext['adx']['value']:.1f}", delta=tech_ext['adx']['desc'])
    with t5:
        st.metric("ATR", f"{tech['atr']:.2f}", delta=f"{tech['atr']/price*100:.1f}%")
    with t6:
        st.metric("布林带", f"¥{tb['mid']:.2f}", delta=f"带宽 {tb['bandwidth']:.1f}%")
    with t7:
        st.metric("OBV", tech_ext['obv']['trend_5d'])

    st.markdown("##### 结构信号")
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    with sm1:
        bos_label = "↑ 突破" if tech["bos"] == 1 else ("↓ 跌破" if tech["bos"] == -1 else "— 无")
        st.metric("BOS", bos_label, help="Break of Structure")
    with sm2:
        choch_label = "↑ 空转多" if tech["choch"] == 1 else ("↓ 多转空" if tech["choch"] == -1 else "— 无")
        st.metric("CHOCH", choch_label, help="Change of Character")
    with sm3:
        fvg_label = "↑ 看多" if tech["fvg"] == 1 else ("↓ 看空" if tech["fvg"] == -1 else "— 无")
        st.metric("FVG", fvg_label, help="Fair Value Gap")
    with sm4:
        ob_label = "↑ 看多" if tech["order_block"] == 1 else ("↓ 看空" if tech["order_block"] == -1 else "— 无")
        st.metric("Order Block", ob_label)
    with sm5:
        st.metric("均线共振", tech["ma_resonance"]["signal"],
                  delta=f"{tech['ma_resonance']['bullish']}/{tech['ma_resonance']['total']}多头")

    # ICT/SMC
    st.caption("---")
    st.markdown("##### ICT/SMC 高级信号")
    ict1, ict2, ict3, ict4, ict5 = st.columns(5)
    with ict1:
        st.metric("Liquidity Sweep", tech_ext['liquidity_sweep']['desc'])
    with ict2:
        st.metric("Breaker Block", tech_ext['breaker_block']['desc'])
    with ict3:
        st.metric("Premium/Discount", tech_ext['premium_discount']['desc'])
    with ict4:
        st.metric("Equal High/Low", tech_ext['equal_high_low']['desc'])
    with ict5:
        mtf = tech_ext['mtf']
        st.metric("多周期共识", mtf['consensus_signal'], delta=f"评分 {mtf['consensus_score']}/100")

    st.caption(f"数据口径: 所有技术指标基于日线OHLCV计算 | 数据源: {_src(stock)} | ICT/SMC信号仅供参考，非独立交易依据")


# ═══════════════════════════════════════════════════════════
#  面板 F — 仓位+融资
# ═══════════════════════════════════════════════════════════

with st.expander("📋 仓位与融资 · 持仓概览/融资成本/维保预警/净股息", expanded=True):
    f1, f2 = st.columns([1, 2])
    with f1:
        st.markdown(f"""
        <div class="apple-card">
            <table style="width:100%;font-size:13px;">
                <tr><td style="color:#86868b;">仓位档位</td><td style="text-align:right;font-weight:600;">第{pos_tier['target_weight']*100:.0f}%档</td></tr>
                <tr><td style="color:#86868b;">建议仓位</td><td style="text-align:right;font-weight:600;">{pos_tier['target_position_pct']:.0f}%</td></tr>
                <tr><td style="color:#86868b;">融资额度</td><td style="text-align:right;font-weight:600;">¥ {pos_tier['suggested_leverage_limit']:,.0f}</td></tr>
                <tr><td style="color:#86868b;">总资产</td><td style="text-align:right;">¥ {risk['total_value']:,.0f}</td></tr>
                <tr><td style="color:#86868b;">总负债</td><td style="text-align:right;">¥ {risk['debt']:,.0f}</td></tr>
                <tr><td style="color:#86868b;">净资产</td><td style="text-align:right;">¥ {risk['equity']:,.0f}</td></tr>
                <tr><td style="color:#86868b;">维保比例</td><td style="text-align:right;font-weight:600;">{margin_ratio:.1f}x</td></tr>
                <tr><td style="color:#86868b;">年利息</td><td style="text-align:right;">¥ {margin_m['annual_interest']:,.0f}</td></tr>
                <tr><td style="color:#86868b;">日利息</td><td style="text-align:right;">¥ {margin_m['daily_interest']:,.2f}</td></tr>
                <tr><td style="color:#86868b;">融资成本率</td><td style="text-align:right;">{margin_m['margin_cost_rate_pct']:.2f}%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        # 维保预警
        warn_color = {"safe": GREEN, "watch": ORANGE, "warning": "#ff6b35", "danger": RED}
        wc = warn_color.get(margin_warn["level"], GRAY)
        st.markdown(f"""
        <div class="apple-card" style="border-left:4px solid {wc};">
            <div class="apple-label">融资维保预警</div>
            <div style="font-size:18px;font-weight:600;color:{wc};">{margin_warn['message']}</div>
            <div style="font-size:11px;color:#86868b;margin-top:4px;">
                需追加资金: ¥{margin_warn['need_add_funds']:,.0f} · 需减负债: ¥{margin_warn['need_reduce_debt']:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 净股息
        annual_div = div_info["total_dividend"]
        margin_interest = margin_m["annual_interest"]
        net_div = calc_dividend_net_yield(annual_div, margin_interest, market_value=mv if mv > 0 else 1)
        carry = positive_carry_check(div_yield)
        nd1, nd2 = st.columns(2)
        with nd1:
            st.metric("净股息收入", f"¥{net_div['net_income']:,.0f}",
                      delta=f"净利率 {net_div['net_yield_pct']:.2f}%")
        with nd2:
            st.metric("正利差判断",
                      "✓ 正利差" if carry["is_positive_carry"] else "✗ 负利差",
                      delta=f"利差 {carry['spread_pct']:.2f}%",
                      delta_color="normal" if carry["is_positive_carry"] else "inverse")

    st.caption(f"融资年利率: {MARGIN_RATE*100:.2f}% · 融资成本率 = 年利息/总市值 · 净股息 = 股息税后 - 融资利息 | 数据源: 持仓参数(用户输入)")


# ═══════════════════════════════════════════════════════════
#  面板 G — 关键价格
# ═══════════════════════════════════════════════════════════

with st.expander("🎯 关键价格 · 买入/减仓/融资买入/止盈/止损", expanded=False):
    zone = key_prices.pop("_current_zone", {})
    cp_val = key_prices.pop("_current_price", price)

    st.markdown(f"""
    <div class="apple-card" style="text-align:center;margin-bottom:12px;">
        <span style="font-size:14px;color:#86868b;">当前区间: </span>
        <span style="font-size:18px;font-weight:700;color:{zone.get('color', BLUE)};">{zone.get('zone', '—')}</span>
        <span style="font-size:12px;color:#aeaeb2;margin-left:8px;">当前价 ¥{cp_val:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    kp_labels = [
        ("suggested_buy", "建议买入", GREEN),
        ("margin_buy", "融资买入", ORANGE),
        ("stop_loss", "止损价", RED),
        ("take_profit_1", "止盈1", RED),
        ("take_profit_2", "止盈2", "#ff6b35"),
        ("suggested_reduce", "建议减仓", BLUE),
        ("dividend_support", "股息支撑", GREEN),
    ]
    kp_cols = st.columns(len(kp_labels))
    for i, (k, label, color) in enumerate(kp_labels):
        info = key_prices.get(k, {})
        with kp_cols[i]:
            st.metric(label, f"¥{info.get('price', 0):.2f}",
                      delta=f"距现价 {info.get('pct_from_current', 0):+.1f}%",
                      delta_color="normal" if info.get('pct_from_current', 0) >= 0 else "inverse")
            st.caption(info.get("reason", ""))

    st.caption("关键价格基于: PE估值 + 布林带 + ATR + 历史低点综合计算 · 仅供参考非交易建议")


# ═══════════════════════════════════════════════════════════
#  面板 H — 风险管理
# ═══════════════════════════════════════════════════════════

with st.expander("⚠ 风险管理 · 压力测试 + 加仓模拟", expanded=False):
    st.markdown("##### 压力测试 — 维保比 = 总资产/总负债，阈值 ≥ 3.0x")
    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        rows_html = []
        for k, v in risk["stress_test"].items():
            safe = "✓" if v["margin_safe"] else "✗"
            sc = GREEN if v["margin_safe"] else RED
            mr_str = f"{v['margin_ratio']:.1f}x" if v["margin_ratio"] != float("inf") else "N/A"
            need = f" 追加 ¥{v['need_margin']:,.0f}" if v["need_margin"] > 0 else ""
            rows_html.append(f"<tr><td>{k}</td><td style='text-align:right;'>¥{v['new_price']:.2f}</td>"
                            f"<td style='text-align:right;'>¥{v['loss']:,.0f}</td>"
                            f"<td style='text-align:right;'>{mr_str}</td>"
                            f"<td style='text-align:right;color:{sc};'>{safe}{need}</td></tr>")
        st.markdown(f"""
        <table style="width:100%;font-size:12px;">
            <tr style="color:#86868b;"><td>场景</td><td style='text-align:right;'>股价</td>
                <td style='text-align:right;'>亏损</td><td style='text-align:right;'>维保比</td>
                <td style='text-align:right;'>安全</td></tr>
            {''.join(rows_html)}
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with h2:
        mc = risk.get("margin_call", {})
        rl = risk.get("risk_level", {})
        st.metric("风险等级", rl.get("level", "—"), delta=rl.get("desc", ""))
        st.caption(mc.get("verdict", ""))

    if risk["overall_safe"]:
        st.success("所有压力测试场景通过 ✓")
    else:
        st.error("部分场景触发追加保证金预警 ✗")

    # 加仓模拟
    st.markdown('<div style="font-weight:500;color:#86868b;margin-top:16px;">加仓方案模拟</div>', unsafe_allow_html=True)
    az1, az2, az3, az4 = st.columns(4)
    with az1:
        add_price = st.number_input("加仓价格 ¥", value=round(price * 0.95, 2) if price else 25.0, step=0.01, key="add_price")
    with az2:
        add_shares = st.number_input("加仓数量", value=10000, step=1000, key="add_shares")
    with az3:
        use_debt = st.checkbox("使用融资", value=False, key="use_debt")
    with az4:
        if st.button("计算加仓", width="stretch"):
            scenario = scenario_add_position(price, S, C, D, add_price, add_shares, use_debt, MD)
            if "error" in scenario:
                st.error(scenario["error"])
            else:
                st.session_state["scenario"] = scenario
                st.rerun()

    if "scenario" in st.session_state:
        sc = st.session_state["scenario"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("新平均成本", f"¥{sc['new_avg_cost']:.3f}")
        with sc2:
            st.metric("新总股数", f"{sc['new_total_shares']:,}")
        with sc3:
            st.metric("新总负债", f"¥{sc['new_debt']:,.0f}")
        with sc4:
            st.metric("新维保比例", f"{sc['new_margin_ratio']:.1f}x",
                      delta="融资加仓" if sc['use_debt'] else "自有资金")


# ═══════════════════════════════════════════════════════════
#  面板 I — 回测
# ═══════════════════════════════════════════════════════════

with st.expander("📉 回测 · 年化/Sharpe/Sortino/回撤/胜率 + 有/无融资对比", expanded=False):
    i1, i2, i3, i4, i5 = st.columns(5)
    with i1:
        st.metric("累计收益", f"{bt['cumulative_return_pct']:+.2f}%",
                  help="策略累计收益(buy&hold × 信号权重)")
    with i2:
        st.metric("年化收益", f"{bt['annual_return_pct']:+.2f}%")
    with i3:
        st.metric("Sharpe", f"{bt['sharpe_ratio']:.3f}",
                  delta="优秀" if bt['sharpe_ratio'] > 1 else ("良好" if bt['sharpe_ratio'] > 0.5 else "偏低"))
    with i4:
        st.metric("Sortino", f"{bt['sortino_ratio']:.3f}")
    with i5:
        st.metric("最大回撤", f"{bt['max_drawdown_pct']:.2f}%")

    i6, i7, i8 = st.columns(3)
    with i6:
        st.metric("胜率", f"{bt['win_rate_pct']:.1f}%")
    with i7:
        pf = bt.get('profit_factor', 0)
        st.metric("盈亏比", f"{pf:.2f}x" if pf != float("inf") else "∞")
    with i8:
        st.metric("回测天数", f"{bt['total_days']}天")

    # 融资对比
    st.markdown("##### 融资 vs 无融资对比")
    cmp1, cmp2, cmp3, cmp4 = st.columns(4)
    with cmp1:
        st.metric("无融资收益", f"{bt_cmp['no_margin']['cumulative_return_pct']:+.2f}%")
    with cmp2:
        st.metric("50%融资收益", f"{bt_cmp['with_margin']['cumulative_return_pct']:+.2f}%")
    with cmp3:
        st.metric("收益差", f"{bt_cmp['diff_return']:+.2f}%",
                  delta=bt_cmp['verdict'])
    with cmp4:
        st.metric("回撤差", f"{bt_cmp['diff_drawdown']:+.2f}%",
                  help="负值→融资增加回撤")

    st.caption("回测假设: 默认70%仓位 · 融资利率3.55% · 信号延迟1日 · 不含交易成本 · 仅供参考")


# ═══════════════════════════════════════════════════════════
#  面板 J — A/H套利
# ═══════════════════════════════════════════════════════════

with st.expander("🔄 A/H套利 · 实时溢价 + 历史分位 + 套利信号", expanded=False):
    j1, j2, j3 = st.columns([1, 1, 2])
    with j1:
        st.metric("A股价格", f"¥{ah.get('a_price', '?')}")
        st.metric("H股价格", f"HK${ah.get('h_price', '?')}")
    with j2:
        st.metric("AH溢价", f"{ah.get('premium', 0):.1f}%")
        st.metric("历史分位", f"{ah_pct.get('percentile', 0.5)*100:.0f}%",
                  delta=f"范围 [{ah_pct.get('min_premium', '?')}%, {ah_pct.get('max_premium', '?')}%]")
    with j3:
        sig_color = ah_sig.get("color", GRAY)
        st.markdown(f"""
        <div class="apple-card" style="border-left:4px solid {sig_color};">
            <div class="apple-label">套利信号</div>
            <div style="font-size:24px;font-weight:600;color:{sig_color};">{ah_sig.get('signal', '—')}</div>
            <div style="font-size:13px;color:#86868b;margin-top:4px;">{ah_sig.get('action', '')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"AH溢价 = (A价 - H价×0.92) / (H价×0.92) × 100% | 历史分位基于日频数据 | 数据源: {_src(ah)} / 模拟历史(待接入akshare AH历史接口)")


# ═══════════════════════════════════════════════════════════
#  面板 K — 交易日志
# ═══════════════════════════════════════════════════════════

with st.expander("📝 交易日志 · 记录交易 + 历史列表 + AI复盘建议", expanded=False):
    st.markdown("##### 记录新交易")
    tx1, tx2, tx3, tx4 = st.columns(4)
    with tx1:
        tx_action = st.selectbox("操作", ["buy", "sell", "add_margin", "reduce_margin", "adjust"], key="tx_action")
    with tx2:
        tx_price = st.number_input("价格 ¥", value=price, step=0.01, key="tx_price")
    with tx3:
        tx_shares = st.number_input("数量", value=1000, step=100, key="tx_shares")
    with tx4:
        tx_reason = st.text_input("理由", placeholder="如: AI评分85→买入", key="tx_reason")

    if st.button("记录交易", width="stretch"):
        tx = add_trade(tx_action, tx_price, tx_shares, tx_reason, ai_score)
        st.success(f"已记录: {tx['date']} {tx['action']} {tx['shares']}股 @ ¥{tx['price']}")
        st.rerun()

    # 交易历史
    if journal:
        st.markdown("##### 交易历史")
        rows = []
        for t in reversed(journal[-20:]):  # 最近20条
            rows.append(f"<tr><td>{t['date']}</td><td>{t['action']}</td>"
                       f"<td style='text-align:right;'>{t['shares']:,}</td>"
                       f"<td style='text-align:right;'>¥{t['price']:.3f}</td>"
                       f"<td style='text-align:right;'>¥{t['amount']:,.0f}</td>"
                       f"<td>{t.get('reason', '')}</td></tr>")
        st.markdown(f"""
        <table style="width:100%;font-size:12px;">
            <tr style="color:#86868b;"><td>日期</td><td>操作</td><td style='text-align:right;'>数量</td>
                <td style='text-align:right;'>价格</td><td style='text-align:right;'>金额</td><td>理由</td></tr>
            {''.join(rows)}
        </table>
        """, unsafe_allow_html=True)

        # 统计
        st.markdown("##### 交易统计")
        js1, js2, js3, js4 = st.columns(4)
        with js1:
            st.metric("总交易", journal_stats["total_trades"])
        with js2:
            st.metric("买入/卖出", f"{journal_stats['buy_count']}/{journal_stats['sell_count']}")
        with js3:
            st.metric("均价(买/卖)", f"¥{journal_stats['avg_buy_price']:.3f}/¥{journal_stats['avg_sell_price']:.3f}")
        with js4:
            st.metric("净P&L(估)", f"¥{journal_stats.get('net_pnl_estimate', 0):,.0f}")
    else:
        st.info("暂无交易记录")

    # AI复盘
    st.markdown("##### AI复盘建议")
    review = generate_ai_review(journal)
    st.text(review)


# ═══════════════════════════════════════════════════════════
#  多因子评分详情（所有因子一览）
# ═══════════════════════════════════════════════════════════

with st.expander("🔬 32因子评分详情", expanded=False):
    for cat_key in CATEGORY_WEIGHTS:
        cat_keys = [k for k, c in FACTOR_CATEGORIES.items() if c == cat_key and k in factors]
        if not cat_keys:
            continue
        st.markdown(f"**{CATEGORY_NAMES.get(cat_key, cat_key)}** (权重 {CATEGORY_WEIGHTS[cat_key]:.0%})")
        cols = st.columns(min(6, len(cat_keys)))
        for i, key in enumerate(cat_keys):
            s = factors.get(key, 50)
            d = descs.get(key, "")
            w = WEIGHTS.get(key, 0)
            bc = RED if s >= 70 else (ORANGE if s >= 50 else (BLUE if s >= 40 else GREEN))
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div class="apple-card" style="text-align:center;padding:10px 8px;">
                    <div class="apple-label">{FACTOR_NAMES.get(key, key)} · {w:.2%}</div>
                    <div style="font-size:20px;font-weight:600;color:{bc};">{s:.0f}</div>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{s}%;background:{bc};"></div></div>
                    <div style="font-size:9px;color:#86868b;margin-top:3px;line-height:1.2;">{d}</div>
                    <div style="font-size:8px;color:#aeaeb2;">{FACTOR_CALIBERS.get(key, '')[:40]}...</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════

st.divider()
st.markdown(f"""
<div style="text-align:center;color:#aeaeb2;font-size:10px;padding:4px 0 24px 0;">
    本系统仅供学习参考，不构成投资建议。投资有风险，入市需谨慎。<br>
    数据更新时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · {SYMBOL}.SH · v6 32因子5大类
</div>
""", unsafe_allow_html=True)
