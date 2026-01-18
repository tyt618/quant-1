import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import time
import json
import os
import hashlib

# 安全导入 scipy
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 0. 配置持久化管理 (Config Persistence)
# ==========================================
CONFIG_FILE = 'strategy_config.json'

# 默认标的池
DEFAULT_CODES = ["518880", "588000", "513100", "510180"]

DEFAULT_PARAMS = {
    'lookback': 25,
    'smooth': 3,
    'threshold': 0.005,
    'min_holding': 3,
    'allow_cash': True,
    'selected_codes': DEFAULT_CODES
}

def load_config():
    """从本地文件加载配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                config = DEFAULT_PARAMS.copy()
                config.update(saved_config)
                return config
        except Exception as e:
            return DEFAULT_PARAMS.copy()
    return DEFAULT_PARAMS.copy()

def save_config(config):
    """保存配置到本地文件"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception:
        pass

# ==========================================
# 1. 投行级页面配置 & CSS样式 (UI优化版)
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 核心资产轮动策略终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局背景与字体优化 */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    }
    
    /* 侧边栏优化 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* 指标卡片 (Metric Card) - 优化阴影和圆角 */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 12px;
        padding: 20px 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-color: #d0d0d0;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #2c3e50;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #95a5a6;
        margin-top: 6px;
    }

    /* 信号横幅 (Signal Banner) - 渐变优化 */
    .signal-banner {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* 表格样式优化 */
    .dataframe {
        font-size: 13px !important;
        border: 1px solid #eee;
    }
    
    /* 总资产大标题 */
    .total-asset-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 0.2rem;
        font-family: 'Arial', sans-serif;
    }
    .total-asset-sub {
        font-size: 1.1rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

TRANSACTION_COST = 0.0001  # 万分之一

PRESET_ETFS = {
    "518880": "黄金ETF (避险)", "588000": "科创50 (硬科技)", "513100": "纳指100 (海外)",
    "510180": "上证180 (蓝筹)", "159915": "创业板指 (成长)", "510300": "沪深300 (大盘)",
    "510500": "中证500 (中盘)", "512890": "红利低波 (防御)", "513500": "标普500 (美股)",
    "512480": "半导体ETF (行业)", "512880": "证券ETF (Beta)"
}

# 辅助函数：根据名称生成柔和的颜色
def get_color_from_name(name):
    if name == 'Cash':
        return 'rgba(200, 200, 200, 0.2)' # 灰色代表空仓
    
    # 简单的哈希生成颜色
    hash_obj = hashlib.md5(name.encode())
    hex_dig = hash_obj.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    
    # 调整为浅色 (Pastel)
    r = (r + 255) // 2
    g = (g + 255) // 2
    b = (b + 255) // 2
    
    return f'rgba({r}, {g}, {b}, 0.25)' # 透明度0.25

def metric_html(label, value, sub="", color="#2c3e50"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """

# ==========================================
# 2. 数据层 (Data Layer)
# ==========================================

@st.cache_data(ttl=3600*12) 
def get_all_etf_list():
    try:
        df = ak.fund_etf_spot_em()
        df['display'] = df['代码'] + " | " + df['名称']
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def download_market_data(codes_list, end_date_str):
    start_str = '20150101' 
    price_dict = {}
    name_map = {}
    
    etf_list = get_all_etf_list()
    
    for code in codes_list:
        name = code
        if code in PRESET_ETFS:
            name = PRESET_ETFS[code].split(" ")[0]
        elif not etf_list.empty:
            match = etf_list[etf_list['代码'] == code]
            if not match.empty:
                name = match.iloc[0]['名称']
        
        name_map[code] = name
        
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                price_dict[name] = df['收盘'].astype(float)
        except Exception:
            continue

    if not price_dict:
        return None, None

    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    data.dropna(how='all', inplace=True)
    
    if len(data) < 20:
        return None, None
        
    return data, name_map

# ==========================================
# 3. 策略内核 (Strategy Core)
# ==========================================

def calculate_momentum(data, lookback, smooth):
    mom = data.pct_change(lookback)
    if smooth > 1:
        mom = mom.rolling(smooth).mean()
    return mom

def fast_backtest_vectorized(daily_ret, mom_df, threshold, min_holding=1, cost_rate=0.0001, allow_cash=True):
    signal_mom = mom_df.shift(1)
    
    n_days, n_assets = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    
    strategy_ret = np.zeros(n_days)
    curr_idx = -2 # -2: 初始, -1: Cash, 0~N: 资产
    
    trade_count = 0
    days_held = 0 # 记录当前持仓天数
    
    for i in range(n_days):
        # 1. 每日自然持仓时间增加
        if curr_idx != -2:
            days_held += 1
            
        row_mom = p_mom[i]
        
        if np.isnan(row_mom).all(): 
            continue
            
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        
        target_idx = curr_idx
        
        # --- 策略逻辑 ---
        if allow_cash and best_val < 0:
            target_idx = -1 # 建议空仓
        else:
            if curr_idx == -2:
                if best_val > -np.inf: 
                    target_idx = best_idx
            elif curr_idx == -1:
                if best_val > 0 or (not allow_cash):
                    target_idx = best_idx
            else:
                is_stop_loss = (target_idx == -1) 
                
                if is_stop_loss:
                    pass
                else:
                    if days_held >= min_holding:
                        curr_val = clean_mom[curr_idx]
                        if best_idx != curr_idx:
                            if best_val > curr_val + threshold:
                                target_idx = best_idx
                    else:
                        target_idx = curr_idx
        
        # --- 交易执行 ---
        if target_idx != curr_idx:
            if curr_idx != -2:
                strategy_ret[i] -= cost_rate
                trade_count += 1
                days_held = 0
            
            curr_idx = target_idx
            
        if curr_idx >= 0:
            strategy_ret[i] += p_ret[i, curr_idx]
            
    equity_curve = (1 + strategy_ret).cumprod()
    total_ret = equity_curve[-1] - 1
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    
    return total_ret, max_dd, equity_curve, trade_count

# ==========================================
# 4. 分析师工具箱
# ==========================================

def calculate_pro_metrics(equity_curve, benchmark_curve, trade_count):
    if len(equity_curve) < 2: return {}
    s_eq = pd.Series(equity_curve)
    s_bm = pd.Series(benchmark_curve) if len(benchmark_curve) == len(equity_curve) else None
    daily_ret = s_eq.pct_change().fillna(0)
    bm_ret = s_bm.pct_change().fillna(0) if s_bm is not None else None
    days = len(equity_curve)
    
    total_ret = equity_curve[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / days) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    rf = 0.03
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    calmar = ann_ret / (abs(max_dd) + 1e-9)
    
    alpha, beta = 0.0, 0.0
    if HAS_SCIPY and bm_ret is not None and len(bm_ret) > 10:
        try:
            slope, intercept, _, _, _ = stats.linregress(bm_ret.values[1:], daily_ret.values[1:])
            beta = slope
            alpha = intercept * 252
        except: pass
            
    return {
        "Total Return": total_ret, "CAGR": ann_ret, "Volatility": ann_vol,
        "Max Drawdown": max_dd, "Sharpe Ratio": sharpe, "Calmar Ratio": calmar,
        "Alpha": alpha, "Beta": beta, "Trades": trade_count
    }

def optimize_parameters(data, allow_cash, min_holding):
    # === [关键修改] 精细化步长设置 ===
    lookbacks = range(20, 31, 1) # 周期步长 1
    smooths = range(1, 8, 1)     # 平滑步长 1 (扩大范围)
    thresholds = np.arange(0.0, 0.013, 0.001) # 阈值步长 0.001
    
    daily_ret = data.pct_change().fillna(0)
    n_days = len(daily_ret) 
    results = []
    
    total_iters = len(lookbacks) * len(smooths) * len(thresholds)
    my_bar = st.progress(0, text="正在进行高精度全参数扫描 (Loop/Smooth/Th)...")
    
    idx = 0
    for lb in lookbacks:
        for sm in smooths:
            mom = calculate_momentum(data, lb, sm)
            for th in thresholds:
                ret, dd, _, count = fast_backtest_vectorized(
                    daily_ret, mom, th, 
                    min_holding=min_holding,
                    cost_rate=TRANSACTION_COST, 
                    allow_cash=allow_cash
                )
                score = ret / (abs(dd) + 0.05)
                results.append([lb, sm, th, ret, count, dd, score])
                
                idx += 1
                if idx % 100 == 0:
                    my_bar.progress(min(idx / total_iters, 1.0))
                    
    my_bar.empty()
    df_res = pd.DataFrame(results, columns=['周期', '平滑', '阈值', '累计收益', '调仓次数', '最大回撤', '得分'])
    
    df_res['年化收益'] = (1 + df_res['累计收益']) ** (252 / n_days) - 1
    return df_res

# ==========================================
# 5. 主程序 UI
# ==========================================

def main():
    if 'params' not in st.session_state:
        saved_config = load_config()
        st.session_state.params = saved_config

    with st.sidebar:
        st.title("🎛️ 策略控制台")
        
        st.subheader("1. 资产池配置")
        all_etfs = get_all_etf_list()
        options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        current_selection_codes = st.session_state.params.get('selected_codes', DEFAULT_CODES)
        
        default_display = []
        if not all_etfs.empty:
            for code in current_selection_codes:
                match = all_etfs[all_etfs['代码'] == code]
                if not match.empty:
                    default_display.append(match.iloc[0]['display'])
                else:
                    for opt in options:
                        if opt.startswith(code):
                            default_display.append(opt)
                            break
        else:
            default_display = current_selection_codes
            
        valid_defaults = [x for x in default_display if x in options]
        selected_display = st.multiselect("核心标的池", options, default=valid_defaults)
        selected_codes = [x.split(" | ")[0] for x in selected_display]
        
        st.divider()
        st.subheader("2. 资金管理")
        
        date_mode = st.radio("回测区间", ["全历史", "自定义"], index=0)
        start_date = datetime(2018, 1, 1)
        end_date = datetime.now()
        
        if date_mode == "自定义":
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start", datetime(2019, 1, 1))
            end_date = c2.date_input("End", datetime.now())
            start_date = datetime.combine(start_date, datetime.min.time())
            end_date = datetime.combine(end_date, datetime.min.time())

        # 定投模式选择
        invest_mode = st.radio("投资模式", ["一次性投入 (Lump Sum)", "定期定额 (SIP)"], index=0)
        
        initial_capital = 100000.0
        sip_amount = 0.0
        sip_freq = "None"
        
        if invest_mode == "一次性投入 (Lump Sum)":
            initial_capital = st.number_input("初始本金", value=100000.0, step=10000.0)
        else:
            c1, c2 = st.columns(2)
            initial_capital = c1.number_input("初始底仓", value=10000.0, step=1000.0)
            sip_amount = c2.number_input("定投金额", value=2000.0, step=500.0)
            sip_freq = st.selectbox("定投频率", ["每月 (Monthly)", "每周 (Weekly)"], index=0)

        st.divider()
        st.subheader("3. 策略内核参数")
        
        # [修改] 改为数字输入框，避免拖动不准
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            p_lookback = st.number_input("动量周期 (Lookback)", min_value=2, max_value=120, value=st.session_state.params.get('lookback', 25), step=1)
        with c_p2:
            p_smooth = st.number_input("平滑窗口 (Smooth)", min_value=1, max_value=60, value=st.session_state.params.get('smooth', 3), step=1)
            
        p_threshold = st.number_input("换仓阈值 (Threshold)", 0.0, 0.05, st.session_state.params.get('threshold', 0.005), step=0.001, format="%.3f")
        
        st.markdown("---")
        st.markdown("**🛑 交易频率控制**")
        # [修改] 改为数字输入框
        p_min_holding = st.number_input("最小持仓天数 (Min Hold)", min_value=1, max_value=60, value=st.session_state.params.get('min_holding', 3), step=1, help="买入后必须持有的最少交易日数。设置为1即不限制。")
        
        p_allow_cash = st.checkbox("启用绝对动量避险 (Cash Protection)", value=st.session_state.params.get('allow_cash', True))
        
        current_params = {
            'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold,
            'min_holding': p_min_holding, 'allow_cash': p_allow_cash, 'selected_codes': selected_codes
        }
        
        if current_params != st.session_state.params:
            st.session_state.params = current_params
            save_config(current_params)
            
        if st.button("🔄 重置默认"):
            st.session_state.params = DEFAULT_PARAMS.copy()
            save_config(DEFAULT_PARAMS)
            st.rerun()

    st.markdown("## 🚀 核心资产轮动策略终端 (Pro Ver.)")
    
    if not selected_codes:
        st.warning("请选择标的。")
        st.stop()
        
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    end_date_str = beijing_now.strftime('%Y%m%d')

    with st.spinner("正在接入市场数据终端 (Smart-Link)..."):
        raw_data, name_map = download_market_data(selected_codes, end_date_str)
        
    if raw_data is None:
        st.error("数据不足或下载失败。")
        st.stop()

    daily_ret_all = raw_data.pct_change().fillna(0)
    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth)
    
    mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)
    sliced_data = raw_data.loc[mask]
    sliced_mom = mom_all.loc[mask] 
    sliced_ret = daily_ret_all.loc[mask]
    
    if sliced_data.empty:
        st.error("区间内无数据")
        st.stop()

    signal_mom = sliced_mom.shift(1)
    dates = sliced_ret.index
    
    # === 增强型循环回测（含详细日志） ===
    cash = initial_capital
    share_val = 0.0
    curr_hold = None
    days_held = 0
    current_hold_start_val = 0.0 # 用于计算段内收益
    
    holdings_history = []
    total_assets_curve = []
    total_invested_curve = []
    total_invested = initial_capital
    trade_count_real = 0
    
    daily_details = [] # 详细交易日记数据
    
    last_sip_date = dates[0]
    
    def format_market_perf(row, n_map):
        items = []
        sorted_items = row.sort_values(ascending=False)
        for code, val in sorted_items.items():
            name = n_map.get(code, code).split("(")[0]
            items.append(f"{name}: {val:+.2%}")
        return " | ".join(items)

    for i, date in enumerate(dates):
        # 0. 准备当日的全市场表现数据
        r_today = sliced_ret.loc[date]
        market_perf_str = format_market_perf(r_today, name_map)

        # A. 定投逻辑
        if invest_mode == "定期定额 (SIP)" and i > 0:
            is_sip_day = False
            if sip_freq.startswith("每月"):
                if date.month != last_sip_date.month: is_sip_day = True
            elif sip_freq.startswith("每周"):
                if date.weekday() == 0 and last_sip_date.weekday() != 0: is_sip_day = True
            
            if is_sip_day:
                cash += sip_amount
                total_invested += sip_amount
                last_sip_date = date

        # B. 信号与持仓时间
        if curr_hold is not None:
            days_held += 1
            
        row = signal_mom.loc[date]
        
        target = curr_hold
        
        if not row.isna().all():
            clean_row = row.fillna(-np.inf)
            best_asset = clean_row.idxmax()
            best_score = clean_row.max()
            
            if p_allow_cash and best_score < 0:
                target = 'Cash'
            else:
                if curr_hold is None or curr_hold == 'Cash':
                    target = best_asset
                else:
                    if days_held >= p_min_holding:
                        curr_score = clean_row.get(curr_hold, -np.inf)
                        if best_asset != curr_hold:
                            if best_score > curr_score + p_threshold:
                                target = best_asset
                    else:
                        target = curr_hold

        day_return = 0.0
        if curr_hold and curr_hold != 'Cash' and curr_hold in r_today:
            day_return = r_today[curr_hold]
        
        share_val = share_val * (1 + day_return)
        
        # === 修复收益显示逻辑：在换仓前计算旧持仓的最终收益 ===
        temp_segment_ret = 0.0
        if curr_hold and curr_hold != 'Cash' and current_hold_start_val > 0:
            # 计算的是【当前持仓】截止到今天的收益（含当日涨跌）
            temp_segment_ret = (share_val / current_hold_start_val) - 1
            
        # 准备日志变量 (默认是今天结束时的状态，但如果是换仓日，我们希望记录旧持仓的谢幕)
        log_hold = curr_hold
        log_days = days_held
        log_ret = temp_segment_ret
        note = ""

        # 交易执行
        if target != curr_hold:
            if curr_hold is not None:
                total_equity = share_val + cash
                cost = total_equity * TRANSACTION_COST
                if cash >= cost: cash -= cost
                else: share_val -= cost
                trade_count_real += 1
                days_held = 0 # 重置持仓时间
                
                # 记录调仓动作
                old_name = name_map.get(curr_hold, curr_hold) if curr_hold else "Cash"
                new_name = name_map.get(target, target) if target else "Cash"
                note = f"调仓: {old_name} -> {new_name}"
                
            if target == 'Cash':
                cash += share_val
                share_val = 0.0
            else:
                total = share_val + cash
                share_val = total
                cash = 0.0
                current_hold_start_val = total # 记录新持仓的初始价值
                
            curr_hold = target
            
        # 记录持仓历史
        holdings_history.append(target if target else "Cash")
        current_total = share_val + cash
        total_assets_curve.append(current_total)
        total_invested_curve.append(total_invested)
        
        # 记录详细日志
        # 如果发生了换仓，log_hold 还是旧的，log_ret 是旧持仓的最终收益。这正是我们想要的。
        # 如果没换仓，log_hold 是当前持仓，log_ret 是当前浮盈。
        hold_name_display = name_map.get(log_hold, log_hold) if log_hold and log_hold != 'Cash' else 'Cash'
        
        daily_details.append({
            "日期": date.strftime('%Y-%m-%d'),
            "当前持仓": hold_name_display,
            "持仓天数": log_days if log_hold != 'Cash' else 0,
            "段内收益": log_ret if log_hold != 'Cash' else 0.0,
            "操作": note,
            "总资产": current_total,
            "全市场表现": market_perf_str
        })

    # 结果封装
    df_res = pd.DataFrame({
        '总资产': total_assets_curve,
        '投入本金': total_invested_curve,
        '持仓': holdings_history
    }, index=dates)
    
    # 策略净值 (用于指标计算，快速版)
    _, _, nav_series, _ = fast_backtest_vectorized(
        sliced_ret, sliced_mom, p_threshold, 
        min_holding=p_min_holding, 
        cost_rate=TRANSACTION_COST, 
        allow_cash=p_allow_cash
    )
    df_res['策略净值'] = nav_series
    bm_curve = (1 + sliced_ret.mean(axis=1)).cumprod()
    
    # === 信号栏 ===
    latest_mom = mom_all.iloc[-1].dropna().sort_values(ascending=False)
    last_hold = holdings_history[-1]
    
    col_sig1, col_sig2 = st.columns([2, 1])
    with col_sig1:
        hold_name = name_map.get(last_hold, last_hold) if last_hold != 'Cash' else '🛡️ 空仓避险 (Cash)'
        lock_msg = f"(已持仓 {days_held} 天)" if last_hold != 'Cash' else ""
        if days_held < p_min_holding and last_hold != 'Cash':
            lock_msg += " 🔒 **锁定中**"
            
        st.markdown(f"""
        <div class="signal-banner">
            <h3 style="margin:0">📌 当前持仓: {hold_name}</h3>
            <div style="margin-top:10px;">
                最小持仓限制: {p_min_holding} 天 {lock_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_sig2:
        st.markdown("**🏆 实时排名**")
        for i, (asset, score) in enumerate(latest_mom.head(3).items()):
            display_name = name_map.get(asset, asset)
            st.markdown(f"{i+1}. **{display_name}**: `{score:.2%}`")

    # === 优化引擎 (Updated) ===
    with st.expander("🛠️ 策略参数优化引擎 (Smart Optimizer)", expanded=False):
        if st.button("运行参数寻优"):
            opt_df = optimize_parameters(sliced_data, p_allow_cash, p_min_holding)
            best_r = opt_df.loc[opt_df['累计收益'].idxmax()]
            
            c1, c2, c3 = st.columns([1,1,2])
            with c1: 
                # 显示平滑参数
                param_str = f"L{int(best_r['周期'])}/S{int(best_r['平滑'])}/T{best_r['阈值']:.3f}"
                st.metric("最佳年化", f"{best_r['年化收益']:.1%}", f"最佳参数: {param_str}")
            with c2: st.metric("对应回撤", f"{best_r['最大回撤']:.1%}", f"调仓: {int(best_r['调仓次数'])}")
            with c3:
                pivot = opt_df.pivot_table(index='阈值', columns='周期', values='得分')
                fig = px.imshow(pivot, labels=dict(color="Score"), aspect="auto", origin='lower')
                fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)

    # === 报表计算 ===
    account_ret = df_res['总资产'].iloc[-1] / df_res['投入本金'].iloc[-1] - 1
    account_profit = df_res['总资产'].iloc[-1] - df_res['投入本金'].iloc[-1]
    metrics = calculate_pro_metrics(df_res['策略净值'].values, bm_curve.values, trade_count_real)
    
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="total-asset-header">¥{df_res['总资产'].iloc[-1]:,.0f}</div>
        <div class="total-asset-sub">
            投入本金: ¥{df_res['投入本金'].iloc[-1]:,.0f} | 
            <span style="color: {'#d62728' if account_profit > 0 else 'green'}">
                总盈亏: {account_profit:+,.0f} ({account_ret:+.2%})
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 近半年收益
    six_months_ago = df_res.index[-1] - timedelta(days=180)
    idx_6m = df_res.index.searchsorted(six_months_ago)
    if idx_6m < len(df_res):
        ret_6m = df_res['策略净值'].iloc[-1] / df_res['策略净值'].iloc[idx_6m] - 1
        bm_ret_6m = bm_curve.iloc[-1] / bm_curve.iloc[idx_6m] - 1
    else:
        ret_6m = 0.0
        bm_ret_6m = 0.0

    st.markdown("### 📊 策略表现概览")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.markdown(metric_html("累计收益", f"{metrics.get('Total Return',0):.1%}", "", "#c0392b"), unsafe_allow_html=True)
    with m2: st.markdown(metric_html("年化收益", f"{metrics.get('CAGR',0):.1%}", "", "#c0392b"), unsafe_allow_html=True)
    with m3: st.markdown(metric_html("近半年收益", f"{ret_6m:.1%}", f"超额: {ret_6m - bm_ret_6m:+.1%}", "#2980b9"), unsafe_allow_html=True)
    with m4: st.markdown(metric_html("最大回撤", f"{metrics.get('Max Drawdown',0):.1%}", "", "#27ae60"), unsafe_allow_html=True)
    with m5: st.markdown(metric_html("夏普比率", f"{metrics.get('Sharpe Ratio',0):.2f}", "", "#2c3e50"), unsafe_allow_html=True)
    with m6: st.markdown(metric_html("交易次数", f"{trade_count_real}", "", "#2c3e50"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 综合图表", "📅 年度/月度回报", "📝 交易日记"])
    
    with tab1:
        # 综合图表
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['策略净值'], name="策略净值", line=dict(color='#c0392b', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_res.index, y=bm_curve, name="基准", line=dict(color='#95a5a6', dash='dash')), row=1, col=1)
        drawdown_series = (df_res['策略净值'] - df_res['策略净值'].cummax()) / df_res['策略净值'].cummax()
        # 修复：移除 line 字典中不支持的 opacity 属性
        fig.add_trace(go.Scatter(x=df_res.index, y=drawdown_series, name="回撤", fill='tozeroy', line=dict(color='#c0392b', width=1)), row=2, col=1)

        # 持仓背景色
        df_res['持仓名称'] = df_res['持仓'].map(lambda x: name_map.get(x, x))
        df_res['持仓变化'] = df_res['持仓'] != df_res['持仓'].shift(1)
        change_indices = df_res[df_res['持仓变化']].index.tolist()
        if df_res.index[0] not in change_indices: change_indices.insert(0, df_res.index[0])
        change_indices.append(df_res.index[-1] + timedelta(days=1))

        shapes = []
        for i in range(len(change_indices) - 1):
            start_t = change_indices[i]
            end_t = change_indices[i+1]
            try:
                if start_t > df_res.index[-1]: continue
                current_code = df_res.loc[start_t, '持仓']
                current_name = df_res.loc[start_t, '持仓名称']
                color = get_color_from_name(current_code)
                shapes.append(dict(type="rect", xref="x", yref="paper", x0=start_t, x1=end_t, y0=0, y1=1, fillcolor=color, opacity=0.3, layer="below", line_width=0))
                mid_point = start_t + (end_t - start_t) / 2
                if (end_t - start_t).days > 15: 
                    fig.add_annotation(x=mid_point, y=0.05, xref="x", yref="paper", text=current_name.split(' ')[0], showarrow=False, font=dict(size=10, color="gray"), opacity=0.7)
            except Exception: pass

        fig.update_layout(shapes=shapes, height=600, title_text="策略综合分析", hovermode="x unified", xaxis=dict(rangeslider=dict(visible=False), type="date"))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        # 1. 年度表
        res_y = []
        years = df_res.index.year.unique()
        for y in years:
            d_sub = df_res[df_res.index.year == y]
            if d_sub.empty: continue
            y_ret = d_sub['策略净值'].iloc[-1] / d_sub['策略净值'].iloc[0] - 1
            b_ret = bm_curve.loc[d_sub.index[-1]] / bm_curve.loc[d_sub.index[0]] - 1
            res_y.append({"年份": y, "策略收益": y_ret, "基准收益": b_ret, "超额(Alpha)": y_ret - b_ret})
        
        st.caption("📅 年度盈亏")
        st.dataframe(pd.DataFrame(res_y).set_index("年份").style.format("{:+.2%}").background_gradient(subset=["超额(Alpha)"], cmap="RdYlGn", vmin=-0.2, vmax=0.2), use_container_width=True)

        # 2. 月度矩阵 (Heatmap)
        st.caption("🗓️ 月度盈亏矩阵 (Monthly Returns Matrix)")
        
        # 计算月度收益
        df_nav = df_res['策略净值'].resample('ME').last() # 使用 ME 替代 M 以避免 Pandas 警告
        monthly_rets = df_nav.pct_change().fillna(0)
        
        # 构建透视表 (Year x Month)
        monthly_data = []
        for date, val in monthly_rets.items():
            monthly_data.append({'Year': date.year, 'Month': date.month, 'Return': val})
            
        df_month = pd.DataFrame(monthly_data)
        pivot_month = df_month.pivot(index='Year', columns='Month', values='Return')
        # 补全月份列 (1-12)
        for m in range(1, 13):
            if m not in pivot_month.columns: pivot_month[m] = np.nan
        pivot_month = pivot_month.sort_index(ascending=False).sort_index(axis=1) # 年份倒序，月份正序
        
        # 绘制热力图
        fig_m = px.imshow(pivot_month, 
                          labels=dict(x="月份", y="年份", color="收益率"),
                          x=[f"{i}月" for i in range(1, 13)],
                          color_continuous_scale="RdYlGn", 
                          color_continuous_midpoint=0.0,
                          text_auto=".1%")
        fig_m.update_layout(height=400)
        st.plotly_chart(fig_m, use_container_width=True)

    with tab3:
        # 交易日记 (从 daily_details 生成)
        st.markdown("##### 📝 详细交易日记")
        df_details = pd.DataFrame(daily_details)
        # 格式化展示
        st.dataframe(
            df_details.sort_values(by="日期", ascending=False).style.format({
                "总资产": "{:,.2f}",
                "段内收益": "{:+.2%}"
            }), 
            use_container_width=True,
            column_config={
                "持仓天数": st.column_config.NumberColumn("持仓天数", help="当前连续持仓天数"),
                "段内收益": st.column_config.NumberColumn("段内收益", help="本段持仓期间的累计收益率", format="%.2f%%"),
                "操作": st.column_config.TextColumn("调仓操作", width="medium"),
                "全市场表现": st.column_config.TextColumn("当日全市场表现", width="large"),
            }
        )

if __name__ == "__main__":
    main()
