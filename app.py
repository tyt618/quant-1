import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="核心资产轮动策略看板",
    page_icon="📈",
    layout="wide"
)

# 标的池配置 (固定不变)
ASSETS = {
    '510180': {'name': '上证180 (价值)', 'color': '#1f77b4'},
    '159915': {'name': '创业板指 (成长)', 'color': '#2ca02c'},
    '513100': {'name': '纳指100 (海外)', 'color': '#9467bd'},
    '518880': {'name': '黄金ETF (避险)', 'color': '#ff7f0e'}
}

# ==========================================
# 2. 数据获取与缓存
# ==========================================
@st.cache_data(ttl=3600*12)
def load_data():
    """下载全量数据"""
    price_dict = {}
    # 下载足够早的数据以确保2014年初始动量可计算
    start_str = '20130101'
    end_str = datetime.now().strftime('%Y%m%d')
    
    # 进度提示
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    idx = 0
    for code, info in ASSETS.items():
        name = info['name']
        status_text.text(f"正在下载: {name}...")
        try:
            # 使用前复权 (qfq) 保证收益率真实性
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            price_dict[name] = df['收盘'].astype(float)
        except Exception as e:
            st.error(f"{name} 下载失败: {e}")
        
        idx += 1
        progress_bar.progress(idx / len(ASSETS))
    
    status_text.text("数据清洗中...")
    # 对齐数据，前向填充处理停牌
    data = pd.concat(price_dict, axis=1).sort_index().ffill().dropna()
    
    progress_bar.empty()
    status_text.empty()
    
    return data

def calculate_indicators(data, lookback, smooth_window):
    """根据参数动态计算指标"""
    # 1. 每日收益率
    daily_returns = data.pct_change().fillna(0)
    
    # 2. 核心动量: 过去N日的累计涨幅 (Pt / Pt-n - 1)
    raw_mom = data.pct_change(lookback)
    
    # 3. 动量平滑 (如果 smooth_window=1 则相当于不平滑)
    if smooth_window > 1:
        signal_mom = raw_mom.rolling(smooth_window).mean()
    else:
        signal_mom = raw_mom
        
    # 4. 信号偏移: T日的持仓只能基于T-1日的收盘数据
    # 因此将计算出的动量向后移动一天
    signal_mom = signal_mom.shift(1)
    
    return daily_returns, signal_mom

# ==========================================
# 3. 回测引擎
# ==========================================
def run_backtest(start_date, end_date, initial_capital, daily_returns, signal_mom, threshold):
    # 截取时间段
    mask = (daily_returns.index >= pd.to_datetime(start_date)) & (daily_returns.index <= pd.to_datetime(end_date))
    period_ret = daily_returns.loc[mask]
    period_mom = signal_mom.loc[mask]
    
    if period_ret.empty:
        return None, 0

    dates = period_ret.index
    capital = initial_capital
    curve = []
    holdings = []
    
    current_holding = None
    trade_count = 0
    
    for date in dates:
        # 获取当日动量排名
        row = period_mom.loc[date]
        
        # 1. 选出分数最高的
        best_asset = row.idxmax()
        best_score = row.max()
        
        target = current_holding
        
        # 2. 决策逻辑
        if pd.isna(best_asset) or pd.isna(best_score):
            # 数据不足，保持不变
            pass 
        else:
            if current_holding is None:
                # 空仓直接买入第一名
                target = best_asset
            elif current_holding not in row.index:
                 # 持仓标的数据缺失，强制换到第一名
                target = best_asset
            else:
                curr_score = row[current_holding]
                # 3. 换仓判定
                if best_asset != current_holding:
                    # 只有当 [第一名] > [当前持仓] + [阈值] 时才切换
                    # 严格PPT模式下阈值为0，即只要高一点点就换
                    if best_score > curr_score + threshold:
                        target = best_asset
                    else:
                        target = current_holding
        
        # 记录调仓
        if target != current_holding and target is not None:
            trade_count += 1
            
        current_holding = target
        
        # 计算净值
        if current_holding:
            r = period_ret.loc[date, current_holding]
            capital = capital * (1 + r)
            holdings.append(current_holding)
        else:
            # 只有在动量数据完全计算出来之前（回测极早期）才会空仓
            # 正常策略运行中是不会空仓的（符合"无条件选择最高"）
            holdings.append('准备期')
            
        curve.append(capital)
        
    res_df = pd.DataFrame({
        '总资产': curve,
        '持仓': holdings
    }, index=dates)
    
    return res_df, trade_count

# ==========================================
# 4. 主界面逻辑
# ==========================================
def main():
    # --- 侧边栏配置 ---
    with st.sidebar:
        st.header("⚙️ 策略控制台")
        
        # 1. 模式选择
        mode = st.radio(
            "选择策略模式",
            ("PPT严格复刻模式", "降频稳健模式 (优化)"),
            index=0,
            help="严格模式完全遵循25日动量、不平滑、不设门槛；稳健模式增加了平滑和换仓阈值以减少磨损。"
        )
        
        st.divider()
        
        # 2. 参数自动设定
        if mode == "PPT严格复刻模式":
            lookback = 25
            smooth = 1       # 不平滑
            threshold = 0.0  # 无阈值
            st.info("✅ 参数已锁定为PPT原始设定：\n- 周期: 25日\n- 平滑: 无\n- 阈值: 0 (无条件切换)")
        else:
            lookback = st.number_input("动量周期 (日)", value=25)
            smooth = st.number_input("平滑窗口 (日)", value=3, help="取过去N天的平均动量，防止单日假摔")
            threshold = st.number_input("换仓阈值", value=0.005, step=0.001, format="%.3f", help="新标的必须高出多少才切换")
        
        st.divider()
        
        # 3. 资金与日期
        init_cash = st.number_input("初始本金", value=500000, step=10000)
        
        # 加载数据
        data = load_data()
        min_date = data.index[0].date()
        max_date = data.index[-1].date()
        
        # 日期选择器
        default_start = datetime(2014, 1, 1).date()
        col1, col2 = st.columns(2)
        start_date = col1.date_input("开始日期", value=default_start, min_value=min_date, max_value=max_date)
        end_date = col2.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)

    # --- 主区域 ---
    st.title("📊 核心资产轮动策略看板")
    
    # 显示策略逻辑文档
    with st.expander("📖 查看策略原理 (基于资料)", expanded=False):
        st.markdown("""
        **核心逻辑：** 每日开盘全仓轮动，持有过去 **25个交易日** 动量最强的一个标的。
        
        **资产池构建：**
        - 🟦 **上证180 (510180)**：代表国内价值、蓝筹。
        - 🟩 **创业板指 (159915)**：代表国内成长、科技。
        - 🟪 **纳指100 (513100)**：代表海外科技，国内替代。
        - 🟧 **黄金ETF (518880)**：全球避险资产（最后的防线）。
        
        **执行细节：**
        1. **每日9:30前计算**：基于过去25日收盘价计算动量。
        2. **无条件轮动**：对比4个标的，谁分数最高就买谁，不设绝对阈值（即便都是负的，也选跌得最少的，通常是黄金）。
        3. **全仓操作**：单一标的满仓持有。
        """)

    # 计算指标
    daily_returns, signal_mom = calculate_indicators(data, lookback, smooth)
    
    # 运行回测
    df_res, trade_count = run_backtest(start_date, end_date, init_cash, daily_returns, signal_mom, threshold)
    
    if df_res is None:
        st.warning("该区间无数据")
        st.stop()
        
    # --- 结果展示 ---
    final_val = df_res['总资产'].iloc[-1]
    total_ret = (final_val / init_cash) - 1
    days = (df_res.index[-1] - df_res.index[0]).days
    annual_ret = (final_val / init_cash) ** (365.25/days) - 1 if days > 0 else 0
    
    # 估算换手周期
    avg_days = days / trade_count if trade_count > 0 else days

    # 1. 核心指标卡
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("区间收益率", f"{total_ret*100:.2f}%", f"期末: {final_val:,.0f}")
    c2.metric("年化收益率", f"{annual_ret*100:.2f}%")
    c3.metric("调仓次数", f"{trade_count} 次", f"平均 {avg_days:.1f} 天/换")
    
    # 2. 交互图表
    st.subheader("📈 资金曲线与持仓状态")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.85, 0.15])
    
    # 资金曲线
    fig.add_trace(go.Scatter(
        x=df_res.index, y=df_res['总资产'],
        mode='lines', name='策略净值',
        line=dict(color='#d62728', width=2),
        hovertemplate='净值: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # 添加基准 (变淡)
    for code, info in ASSETS.items():
        name = info['name']
        bench = (1 + daily_returns.loc[df_res.index, name]).cumprod()
        bench = bench / bench.iloc[0] * init_cash
        fig.add_trace(go.Scatter(
            x=df_res.index, y=bench,
            name=name, line=dict(width=1, dash='dot'), opacity=0.3
        ), row=1, col=1)

    # 底部持仓色带
    df_res['group'] = (df_res['持仓'] != df_res['持仓'].shift()).cumsum()
    groups = df_res.reset_index().groupby('group').agg({
        '日期': ['first', 'last'],
        '持仓': 'first'
    })
    groups.columns = ['start', 'end', 'asset']
    
    for _, row in groups.iterrows():
        asset = row['asset']
        color = 'gray'
        for _, info in ASSETS.items():
            if info['name'] == asset: color = info['color']
        
        fig.add_trace(go.Scatter(
            x=[row['start'], row['end']], y=[1, 1],
            mode='lines', line=dict(color=color, width=15),
            name=asset, showlegend=False,
            hovertemplate=f"持仓: {asset}<extra></extra>"
        ), row=2, col=1)

    fig.update_layout(height=500, hovermode="x unified", yaxis=dict(title='总资产'), yaxis2=dict(showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. 详细数据表
    with st.expander("查看每日详细数据"):
        detail = df_res.copy()
        detail['日涨跌'] = detail['总资产'].pct_change()
        st.dataframe(detail.sort_index(ascending=False).style.format({'总资产': '{:,.2f}', '日涨跌': '{:.2%}'}))

if __name__ == "__main__":
    main()