"""
数据层 — 多股票行情 + 估值 + 财务 + 宏观/资金
直接调用 Yahoo Finance API（绕过 yfinance 限流）
"""
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from utils import fillna_df


# ═══════════════════════════════════════════════════════════
#  Yahoo Finance 直连
# ═══════════════════════════════════════════════════════════

_YF_SESSION = None

def _get_yf_session():
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = requests.Session()
        _YF_SESSION.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
    return _YF_SESSION


def _fetch_yahoo(symbol: str) -> pd.Series:
    encoded = symbol.replace('=', '%3D').replace('^', '%5E')
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{encoded}'
    params = {'range': '5y', 'interval': '1d', 'includePrePost': 'false'}
    resp = _get_yf_session().get(url, params=params, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Yahoo API HTTP {resp.status_code}")
    data = resp.json()
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quotes = result['indicators']['quote'][0]
    s = pd.Series(quotes['close'], index=pd.to_datetime(timestamps, unit='s'), name=symbol).dropna()
    if len(s) < 10:
        raise ValueError(f"{symbol} 数据不足")
    return s


def _fix_gbk_cols(cols):
    fixed = []
    for c in cols:
        try:
            fixed.append(c.encode('latin-1').decode('gbk'))
        except Exception:
            fixed.append(c)
    return fixed


# ═══════════════════════════════════════════════════════════
#  股票行情 (akshare)
# ═══════════════════════════════════════════════════════════

def fetch_stock(symbol: str = "600938") -> pd.DataFrame:
    """A股日线 OHLCV · 数据源: akshare stock_zh_a_hist · 口径: 前复权日线"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20220101",
                                end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
        df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                           "最低": "low", "成交量": "volume", "成交额": "amount"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df[[c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]].fillna(0)
        df.attrs["source"] = "akshare (前复权日线)"
        df.attrs["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        if len(df) < 50:
            raise ValueError("数据不足")
        return df
    except Exception:
        from utils import generate_fallback_data
        df = generate_fallback_data(500, seed=42)
        df.attrs["source"] = "fallback (模拟)"
        df.attrs["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return df


def fetch_stock_spot(symbol: str = "600938") -> dict:
    """实时行情 · 数据源: akshare stock_zh_a_spot_em · 口径: 交易日盘中实时"""
    result = {"_source": "akshare (东方财富)", "_updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == symbol]
        if row.empty:
            raise ValueError(f"未找到 {symbol}")
        r = row.iloc[0]
        result["name"] = str(r.get("名称", ""))
        result["price"] = float(r.get("最新价", 0) or 0)
        result["change_pct"] = float(r.get("涨跌幅", 0) or 0)
        result["volume"] = float(r.get("成交量", 0) or 0)
        result["amount"] = float(r.get("成交额", 0) or 0)
        result["pe"] = float(r.get("市盈率-动态", 0) or 0)
        result["pb"] = float(r.get("市净率", 0) or 0)
        result["total_market_cap"] = float(r.get("总市值", 0) or 0)
        return result
    except Exception:
        result.update({"price": 27.0, "pe": 8.5, "pb": 1.3, "change_pct": 0, "_fallback": True,
                       "total_market_cap": 8_000_000_000_000,
                       "_source": "fallback (预估值)"})
        return result


# ═══════════════════════════════════════════════════════════
#  财务指标 (akshare)
# ═══════════════════════════════════════════════════════════

def fetch_financial(symbol: str = "600938") -> dict:
    """财务指标 · 数据源: akshare stock_financial_analysis_indicator
    口径: 季报累计值（非年化/TTM），需注意季节性；列顺序固定
    - col[1]: 摊薄每股收益(元) → 每股收益
    - col[5]: 每股净资产_调整前(元)
    - col[11]: 总资产收益率(%)
    - col[15]: 营业利润率(%)
    - col[29]: 加权净资产收益率(%) → ROE（加权，最近一期）
    扩展映射:
    - col[3]: 净资产收益率-摊薄(%) → ROIC近似
    - col[13]: 销售毛利率(%) → 毛利率
    - col[16]: 销售净利率(%) → 净利率
    """
    result = {"_source": "akshare (财务指标)", "_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "_caliber": "季报累计值 — EPS/ROE 均为最近一期报告数据，非年化/TTM"}
    try:
        import akshare as ak
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2022")
        if df is None or df.empty:
            raise ValueError("财务数据空")
        latest = df.iloc[-1]
        result["每股收益"] = float(latest.iloc[1]) if pd.notna(latest.iloc[1]) else 0.0
        result["每股净资产"] = float(latest.iloc[5]) if pd.notna(latest.iloc[5]) else 0.0
        result["总资产收益率"] = float(latest.iloc[11]) if pd.notna(latest.iloc[11]) else 0.0
        result["营业利润率"] = float(latest.iloc[15]) if pd.notna(latest.iloc[15]) else 0.0
        result["加权净资产收益率"] = float(latest.iloc[29]) if pd.notna(latest.iloc[29]) else 0.0
        # 扩展字段 — col index based on akshare stock_financial_analysis_indicator
        result["roic"] = float(latest.iloc[3]) if len(latest) > 3 and pd.notna(latest.iloc[3]) else 0.0
        result["毛利率"] = float(latest.iloc[13]) if len(latest) > 13 and pd.notna(latest.iloc[13]) else 0.0
        result["净利率"] = float(latest.iloc[16]) if len(latest) > 16 and pd.notna(latest.iloc[16]) else 0.0
        # FCF = 营业利润率近似(自由现金流无直接字段)
        result["fcf_ratio"] = float(latest.iloc[15]) if pd.notna(latest.iloc[15]) else 0.0
        # 保留原始列
        for i, key in enumerate(df.columns):
            try:
                result[str(key)] = float(latest.iloc[i]) if pd.notna(latest.iloc[i]) else 0.0
            except Exception:
                pass
        return result
    except Exception:
        result.update({"每股收益": 2.57, "每股净资产": 10.50, "加权净资产收益率": 15.64,
                       "营业利润率": 30.5, "roic": 12.0, "毛利率": 45.0, "净利率": 28.0,
                       "fcf_ratio": 30.5, "_fallback": True, "_source": "fallback (预估值)"})
        return result


# ═══════════════════════════════════════════════════════════
#  PE/PB 历史 (akshare)
# ═══════════════════════════════════════════════════════════

def fetch_pe_pb_history(symbol: str = "600938") -> tuple:
    """PE/PB历史序列 · 数据源: akshare stock_zh_valuation_ba · 口径: 日频 PE-TTM, PB · 返回 (pe_series, pb_series)"""
    try:
        import akshare as ak
        df_pe = ak.stock_zh_valuation_ba(symbol=symbol, indicator="市盈率")
        df_pb = ak.stock_zh_valuation_ba(symbol=symbol, indicator="市净率")
        if df_pe is None or df_pe.empty:
            raise ValueError("PE历史数据空")
        df_pe["date"] = pd.to_datetime(df_pe.iloc[:, 0])
        df_pe = df_pe.rename(columns={df_pe.columns[1]: "pe_ttm"})
        df_pe.set_index("date", inplace=True)
        pe_s = df_pe["pe_ttm"].dropna().astype(float)
        pe_s.attrs = {"source": "akshare (估值历史)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                      "caliber": "PE-TTM 日频"}
        if df_pb is not None and not df_pb.empty:
            df_pb["date"] = pd.to_datetime(df_pb.iloc[:, 0])
            df_pb = df_pb.rename(columns={df_pb.columns[1]: "pb"})
            df_pb.set_index("date", inplace=True)
            pb_s = df_pb["pb"].dropna().astype(float)
            pb_s.attrs = {"source": "akshare (估值历史)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                          "caliber": "PB 日频"}
        else:
            pb_s = pe_s.copy()
            pb_s.attrs = pe_s.attrs
        return pe_s, pb_s
    except Exception:
        n = 500
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
        pe = np.random.normal(10, 3, n).clip(3, 40)
        pb = np.random.normal(1.8, 0.5, n).clip(0.8, 4.5)
        pe_s = pd.Series(pe, index=dates, name="pe_ttm")
        pe_s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        pb_s = pd.Series(pb, index=dates, name="pb")
        pb_s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return pe_s, pb_s


def fetch_dividend_history(symbol: str = "600938") -> pd.DataFrame:
    """股息历史 · 数据源: akshare stock_history_dividend_detail · 口径: 年度分红记录"""
    try:
        import akshare as ak
        df = ak.stock_history_dividend_detail(symbol=symbol, indicator="分红")
        if df is None or df.empty:
            raise ValueError("股息历史数据空")
        df.attrs = {"source": "akshare (分红历史)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "caliber": "年度分红记录（含送转股）"}
        return df
    except Exception:
        df = pd.DataFrame({"年份": ["2023", "2022", "2021"], "每股分红": [1.23, 1.10, 0.95]})
        df.attrs = {"source": "fallback (预估值)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return df


# ═══════════════════════════════════════════════════════════
#  国际油企对标 (Yahoo Finance)
# ═══════════════════════════════════════════════════════════

OIL_PEERS = {
    "XOM": "埃克森美孚", "CVX": "雪佛龙", "COP": "康菲石油",
    "BP": "BP", "SHEL": "壳牌",
}

def fetch_global_oil_peers() -> dict:
    """国际油企估值对标 · 数据源: Yahoo Finance · 口径: 实时 PE/PB"""
    result = {"_source": "Yahoo Finance", "_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "_caliber": "各公司最新 PE-TTM / PB"}
    for ticker, name in OIL_PEERS.items():
        try:
            s = _fetch_yahoo(ticker)
            price = float(s.iloc[-1])
            result[ticker] = {"name": name, "price": round(price, 2), "pe": None, "pb": None, "ticker": ticker}
        except Exception:
            result[ticker] = {"name": name, "price": None, "pe": None, "pb": None, "ticker": ticker,
                              "_error": "获取失败"}
    # PE/PB 无法从 chart API 直接获取，标记为需人工补充
    return result


# ═══════════════════════════════════════════════════════════
#  布伦特 / 美元 / VIX / US10Y (Yahoo Finance)
# ═══════════════════════════════════════════════════════════

def fetch_brent() -> pd.Series:
    """布伦特原油期货连续合约 · 数据源: Yahoo Finance BZ=F · 口径: 日线收盘价(美元/桶)"""
    try:
        s = _fetch_yahoo('BZ=F')
        s.name = "brent"
        s.attrs = {"source": "Yahoo Finance (BZ=F)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "caliber": "布伦特原油期货连续合约日线收盘价 USD/桶"}
        return s
    except Exception:
        np.random.seed(123)
        n = 500
        price = 72.0
        out = [price]
        for _ in range(1, n):
            price *= 1 + np.random.normal(0.0001, 0.02)
            out.append(price)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
        s = pd.Series(out, index=dates, name="brent")
        s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return s


def fetch_dxy() -> pd.Series:
    """美元指数 · 数据源: Yahoo Finance DX-Y.NYB · 口径: 日线收盘价"""
    try:
        s = _fetch_yahoo('DX-Y.NYB')
        s.name = "dxy"
        s.attrs = {"source": "Yahoo Finance (DX-Y.NYB)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "caliber": "美元指数日线收盘价"}
        return s
    except Exception:
        s = pd.Series(100 + np.random.randn(500).cumsum() * 0.3,
                      index=pd.date_range(end=pd.Timestamp.now(), periods=500, freq="B"), name="dxy")
        s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return s


def fetch_vix() -> pd.Series:
    """VIX 恐慌指数 · 数据源: Yahoo Finance ^VIX · 口径: CBOE VIX 日线收盘价"""
    try:
        s = _fetch_yahoo('^VIX')
        s.name = "vix"
        s.attrs = {"source": "Yahoo Finance (^VIX)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "caliber": "CBOE 波动率指数日线收盘价"}
        return s
    except Exception:
        np.random.seed(777)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq="B")
        s = pd.Series(18 + np.random.randn(500).cumsum() * 0.5, index=dates, name="vix").clip(lower=8, upper=45)
        s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return s


def fetch_us10y() -> pd.Series:
    """美国10年期国债收益率 · 数据源: Yahoo Finance ^TNX · 口径: 日线收盘收益率(%)"""
    try:
        s = _fetch_yahoo('^TNX')
        s.name = "us10y"
        s.attrs = {"source": "Yahoo Finance (^TNX)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "caliber": "美国10年期国债收益率 日线收盘 %"}
        return s
    except Exception:
        np.random.seed(456)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq="B")
        s = pd.Series(4.2 + np.random.randn(500).cumsum() * 0.03, index=dates, name="us10y").clip(lower=0.5, upper=7)
        s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return s


# ═══════════════════════════════════════════════════════════
#  上证指数 (akshare)
# ═══════════════════════════════════════════════════════════

def fetch_ssec() -> pd.Series:
    """上证综指 · 数据源: akshare stock_zh_index_daily · 口径: 日线收盘价"""
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol="sh000001")
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        s = df["close"].dropna()
        s.name = "ssec"
        s.attrs = {"source": "akshare (上证综指)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "caliber": "上证综指日线收盘价"}
        return s
    except Exception:
        np.random.seed(888)
        n = 500
        price = 3300.0
        out = [price]
        for _ in range(1, n):
            price *= 1 + np.random.normal(0.0001, 0.012)
            out.append(price)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
        s = pd.Series(out, index=dates, name="ssec")
        s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return s


# ═══════════════════════════════════════════════════════════
#  CPI / PMI (akshare)
# ═══════════════════════════════════════════════════════════

def fetch_cpi() -> pd.Series:
    """中国CPI同比 · 数据源: akshare macro_china_cpi_monthly · 口径: 月度同比(%)"""
    try:
        import akshare as ak
        df = ak.macro_china_cpi_monthly()
        if df is None or df.empty:
            raise ValueError("CPI数据空")
        date_col = df.columns[0]
        val_col = df.columns[1]
        df["date"] = pd.to_datetime(df[date_col].astype(str))
        df.set_index("date", inplace=True)
        s = df[val_col].dropna().astype(float)
        s.name = "cpi"
        s.attrs = {"source": "akshare (国家统计局)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "caliber": "中国CPI当月同比 %"}
        return s
    except Exception:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="MS")
        s = pd.Series(0.4 + np.random.randn(60) * 0.5, index=dates, name="cpi")
        s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return s


def fetch_pmi() -> pd.Series:
    """中国制造业PMI · 数据源: akshare macro_china_pmi · 口径: 月度"""
    try:
        import akshare as ak
        df = ak.macro_china_pmi()
        if df is None or df.empty:
            raise ValueError("PMI数据空")
        date_col = df.columns[0]
        val_col = df.columns[1]
        df["date"] = pd.to_datetime(df[date_col].astype(str))
        df.set_index("date", inplace=True)
        s = df[val_col].dropna().astype(float)
        s.name = "pmi"
        s.attrs = {"source": "akshare (国家统计局)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "caliber": "中国制造业PMI"}
        return s
    except Exception:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="MS")
        s = pd.Series(50.1 + np.random.randn(60) * 1.5, index=dates, name="pmi")
        s.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return s


def fetch_oil_supply() -> dict:
    """OPEC供给预估 · 数据源: 无可靠免费API · 口径: 预估值(万桶/日)"""
    return {"_source": "预估值（无实时API）", "_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "_caliber": "OPEC+ 原油产量预估 万桶/日",
            "opec_output": 28_500, "non_opec_output": 54_000, "total": 82_500,
            "trend": "稳定", "note": "基于EIA月度展望预估值，非实时数据"}


# ═══════════════════════════════════════════════════════════
#  融资余额 / 北向资金
# ═══════════════════════════════════════════════════════════

def fetch_margin() -> pd.DataFrame:
    """沪深两市融资余额 · 数据源: akshare macro_china_market_margin_sh/sz
    口径: 日频，沪市+深市合计融资余额（亿元）"""
    try:
        import akshare as ak
        sh = ak.macro_china_market_margin_sh()
        sz = ak.macro_china_market_margin_sz()
        if sh is not None and not sh.empty and sz is not None and not sz.empty:
            total = sh.iloc[:, 1].astype(float) + sz.iloc[:, 1].astype(float)
            dates = pd.to_datetime(sh.iloc[:, 0])
            df = pd.DataFrame({"margin_balance": total.values}, index=dates).sort_index()
            df.attrs = {"source": "akshare (上交所+深交所)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "caliber": "沪深两市融资余额合计 亿元"}
            return df
        raise ValueError("融资数据空")
    except Exception:
        n = 120
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
        df = pd.DataFrame({"margin_balance": 1_500_000_000_000 + np.random.randn(n).cumsum() * 1e10}, index=dates)
        df.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return df


def fetch_northbound() -> pd.DataFrame:
    """北向资金净流入 · 数据源: akshare stock_hsgt_hist_em · 口径: 日频 亿元"""
    try:
        import akshare as ak
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df is not None and not df.empty:
            cols = _fix_gbk_cols(list(df.columns))
            df.columns = cols
            date_col = next((c for c in cols if "日期" in c), cols[0])
            flow_col = next((c for c in cols if "净流入" in c or "当日成交净买额" in c), None)
            if flow_col is None:
                num_cols = [c for c in cols if df[c].dtype in ('float64', 'int64')]
                flow_col = num_cols[0] if num_cols else cols[2]
            out = pd.DataFrame({"date": pd.to_datetime(df[date_col]),
                                "net_flow": df[flow_col].astype(float)})
            out.set_index("date", inplace=True)
            out = out.sort_index()
            out.attrs = {"source": "akshare (沪深港通)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                         "caliber": "北向资金当日净买入额 亿元"}
            return out
        raise ValueError("北向数据空")
    except Exception:
        n = 120
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
        df = pd.DataFrame({"net_flow": np.random.randn(n) * 50 + 5}, index=dates)
        df.attrs = {"source": "fallback (模拟)", "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
        return df


# ═══════════════════════════════════════════════════════════
#  AH 溢价
# ═══════════════════════════════════════════════════════════

def fetch_ah_premium(symbol: str = "600938") -> dict:
    """AH溢价 · 数据源: akshare stock_zh_ah_spot · 口径: 实时
    公式: 溢价 = (A股价格 - H股价格 × CNY/HKD汇率) / (H股价格 × CNY/HKD汇率) × 100%
    """
    try:
        import akshare as ak
        df = ak.stock_zh_ah_spot()
        df.columns = _fix_gbk_cols(list(df.columns))
        # 查 H 股代码：600938→00883, 601857→00857, 600028→00386 等
        h_map = {"600938": "00883", "601857": "00857", "600028": "00386",
                 "601088": "01088", "601318": "02318"}
        h_code = h_map.get(symbol, "00883")
        row = df[df["代码"].astype(str).str.strip() == h_code]
        if row.empty:
            row = df[df["名称"].str.contains("海油|石油|石化", na=False)]
            if row.empty:
                raise ValueError(f"未找到 H 股 {h_code}")
        r = row.iloc[0]
        h_price = float(r.get("最新价", 0) or 0)
        # 取 A 股价
        try:
            a_df = ak.stock_zh_a_spot_em()
            a_row = a_df[a_df["代码"] == symbol]
            a_price = float(a_row.iloc[0].get("最新价", 0) or 0) if not a_row.empty else h_price * 1.5
        except Exception:
            a_price = h_price * 1.5
        fx = 0.92
        h_cny = h_price * fx
        premium = (a_price - h_cny) / h_cny * 100 if h_cny > 0 else 0.0
        return {"a_code": symbol, "h_code": h_code, "a_price": round(a_price, 2),
                "h_price": round(h_price, 2), "premium": round(premium, 1),
                "_source": "akshare (实时)", "_caliber": "AH溢价 = (A-H×汇率)/(H×汇率) × 100%",
                "_updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
    except Exception:
        return {"a_price": 27.0, "h_price": 6.3, "premium": 45.0, "_fallback": True,
                "_source": "fallback (预估值)", "_updated": datetime.now().strftime("%Y-%m-%d %H:%M")}


# ═══════════════════════════════════════════════════════════
#  统一入口
# ═══════════════════════════════════════════════════════════

def get_all_data(symbol: str = "600938") -> dict:
    brent = fetch_brent()
    dxy = fetch_dxy()
    vix = fetch_vix()
    ssec = fetch_ssec()
    us10y = fetch_us10y()
    cpi = fetch_cpi()
    pmi = fetch_pmi()
    oil_supply = fetch_oil_supply()
    margin = fetch_margin()
    northbound = fetch_northbound()
    # 股票相关 — 依赖 symbol
    stock = fetch_stock(symbol)
    spot = fetch_stock_spot(symbol)
    financial = fetch_financial(symbol)
    ah = fetch_ah_premium(symbol)
    pe_pb_history, pb_history = fetch_pe_pb_history(symbol)
    dividend_history = fetch_dividend_history(symbol)
    peers = fetch_global_oil_peers()

    stock = fillna_df(stock)
    return {
        "symbol": symbol,
        "stock": stock,
        "spot": spot,
        "financial": financial,
        "ah": ah,
        "pe_pb_history": pe_pb_history,
        "pb_history": pb_history,
        "dividend_history": dividend_history,
        "peers": peers,
        "brent": brent,
        "dxy": dxy,
        "vix": vix,
        "us10y": us10y,
        "cpi": cpi,
        "pmi": pmi,
        "oil_supply": oil_supply,
        "ssec": ssec,
        "margin": margin,
        "northbound": northbound,
    }
