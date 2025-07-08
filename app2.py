import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import date, timedelta
import platform
from matplotlib import font_manager, rc

# --------------------------------------------------------------------------
# 🖥️ 기본 설정 및 캐싱
# --------------------------------------------------------------------------
st.set_page_config(page_title="백테스트 비교 분석", page_icon="⚖️", layout="wide")

@st.cache_resource
def set_font():
    if platform.system() == 'Windows':
        path = "c:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)
    elif platform.system() == 'Darwin':
        rc('font', family='AppleGothic')
    else:
        rc('font', family='NanumGothic')
    mpl.rcParams['axes.unicode_minus'] = False

set_font()

@st.cache_data
def get_krx_list():
    return fdr.StockListing('KRX')

# --------------------------------------------------------------------------
# 📉 1. 기존 백테스트 함수 (Look-ahead Bias 존재)
# --------------------------------------------------------------------------
def run_backtest_original(weekly_df):
    df = weekly_df.copy()
    position = False
    entries, exits = [], []
    entry_date_temp = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        if not position and row['BuySignal'] == 1:
            entry_date_temp = row.name
            entry_price = row['Open']
            position = True
        elif position and row['SellSignal'] == 1:
            exit_date = row.name
            exit_price = row['Close']
            entries.append((entry_date_temp, entry_price))
            exits.append((exit_date, exit_price))
            position = False

    if position and entry_date_temp:
        last_row = df.iloc[-1]
        entries.append((entry_date_temp, entry_price))
        exits.append((last_row.name, last_row['Close']))

    bt_df = pd.DataFrame()
    summary = {'total_trades': 0, 'avg_return': 0, 'cum_return': 0, 'win_rate': 0}
    if entries:
        bt_df = pd.DataFrame(entries, columns=['EntryDate', 'EntryPrice'])
        bt_df['ExitDate'], bt_df['ExitPrice'] = zip(*exits)
        bt_df['Return'] = (bt_df['ExitPrice'] - bt_df['EntryPrice']) / bt_df['EntryPrice']
        bt_df['CumulativeReturn'] = (1 + bt_df['Return']).cumprod()
        summary = {
            'total_trades': len(bt_df),
            'avg_return': bt_df['Return'].mean(),
            'cum_return': bt_df['CumulativeReturn'].iloc[-1] - 1,
            'win_rate': (bt_df['Return'] > 0).mean()
        }
    return bt_df, summary

# --------------------------------------------------------------------------
# 📈 2. 수정된 백테스트 함수 (현실적)
# --------------------------------------------------------------------------
def run_backtest_revised(weekly_df):
    df = weekly_df.copy()
    df['BuyTrigger'] = df['BuySignal'].shift(1)
    df['SellTrigger'] = df['SellSignal'].shift(1)
    df['ActualSellSignal'] = 0

    position = False
    entries, exits = [], []
    entry_date_temp = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        if not position and row['BuyTrigger'] == 1:
            entry_date_temp = row.name
            entry_price = row['Open']
            position = True
        elif position and row['SellTrigger'] == 1:
            exit_date = row.name
            exit_price = row['Open']
            entries.append((entry_date_temp, entry_price))
            exits.append((exit_date, exit_price))
            position = False
            df.loc[df.index[i], 'ActualSellSignal'] = 1

    if position and entry_date_temp:
        last_row = df.iloc[-1]
        entries.append((entry_date_temp, entry_price))
        exits.append((last_row.name, last_row['Open']))
        df.loc[df.index[-1], 'ActualSellSignal'] = 1

    bt_df = pd.DataFrame()
    summary = {'total_trades': 0, 'avg_return': 0, 'cum_return': 0, 'win_rate': 0}
    if entries:
        bt_df = pd.DataFrame(entries, columns=['EntryDate', 'EntryPrice'])
        bt_df['ExitDate'], bt_df['ExitPrice'] = zip(*exits)
        bt_df['Return'] = (bt_df['ExitPrice'] - bt_df['EntryPrice']) / bt_df['EntryPrice']
        bt_df['CumulativeReturn'] = (1 + bt_df['Return']).cumprod()
        summary = {
            'total_trades': len(bt_df),
            'avg_return': bt_df['Return'].mean(),
            'cum_return': bt_df['CumulativeReturn'].iloc[-1] - 1,
            'win_rate': (bt_df['Return'] > 0).mean()
        }
    return bt_df, summary, df['ActualSellSignal']

# --------------------------------------------------------------------------
# ⚖️ 메인 분석 및 비교 함수
# --------------------------------------------------------------------------
def run_analysis_and_compare(stock_name, start_date):
    # 데이터 불러오기 및 기본 지표 계산 (공통 과정)
    krx_list = get_krx_list()
    try:
        stock_code = krx_list[krx_list['Name'] == stock_name]['Code'].iloc[0]
    except IndexError:
        st.error("해당 종목을 찾을 수 없습니다."); return

    df = fdr.DataReader(stock_code, start=start_date, end=date.today())
    if df.empty:
        st.error("해당 기간의 데이터가 없습니다."); return
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = df[df[['Open', 'High', 'Low', 'Close']].ne(0).all(axis=1)]
    weekly = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    weekly['Prev_High'] = weekly['High'].shift(1); weekly['Prev_Low'] = weekly['Low'].shift(1)
    weekly['MA10'] = weekly['Close'].rolling(window=10).mean()
    mf_multiplier = ((weekly['Close'] - weekly['Low']) - (weekly['High'] - weekly['Close'])) / (weekly['High'] - weekly['Low'])
    mf_volume = mf_multiplier * weekly['Volume']; weekly['CMF'] = mf_volume.rolling(window=4).sum() / weekly['Volume'].rolling(window=4).sum()
    weekly['BuySignal'] = 0; weekly['SellSignal'] = 0
    weekly.loc[(weekly['High'] > weekly['Prev_High']) & (weekly['Close'] > weekly['MA10']) & (weekly['CMF'] > 0), 'BuySignal'] = 1
    weekly.loc[(weekly['Low'] < weekly['Prev_Low']) & (weekly['Close'] < weekly['MA10']) & (weekly['CMF'] < 0), 'SellSignal'] = 1
    
    # Fear & Greed 지수 계산 (생략 - 기존과 동일)
    weekly['Momentum5'] = (np.log(weekly['Close']) - np.log(weekly['Close'].shift(5))) * 100
    rolling_low = weekly['Close'].rolling(window=52, min_periods=1).min(); rolling_high = weekly['Close'].rolling(window=52, min_periods=1).max()
    weekly['Position52W'] = ((weekly['Close'] - rolling_low) / (rolling_high - rolling_low)).clip(0, 1)
    recent_vol_avg = weekly['Volume'].rolling(window=5, min_periods=1).mean(); past_vol_avg = weekly['Volume'].rolling(window=20, min_periods=1).mean()
    weekly['VolumeSurge'] = (recent_vol_avg / past_vol_avg).clip(0, 3)
    weekly_return = weekly['Close'].pct_change(); recent_vol = weekly_return.rolling(window=5, min_periods=1).std(); past_vol = weekly_return.rolling(window=20, min_periods=1).std()
    weekly['VolatilitySpike'] = (recent_vol / past_vol).clip(0, 3)
    momentum_score = (weekly['Momentum5'].rolling(window=7, min_periods=1).mean() / 10).clip(-1, 1.5)
    position_score = (2 * weekly['Position52W'].rolling(window=7, min_periods=1).mean() - 1).clip(-1, 1.5)
    volume_score = (weekly['VolumeSurge'].rolling(window=10, min_periods=1).mean() - 1).clip(-0.5, 1.2)
    volatility_score = -(weekly['VolatilitySpike'].rolling(window=10, min_periods=1).mean() - 1).clip(-0.5, 1.2)
    weekly['FearGreedScore'] = (0.45 * momentum_score + 0.45 * position_score + 0.05 * volume_score + 0.05 * volatility_score)

    # 각 방식으로 백테스트 실행
    bt_df_orig, summary_orig = run_backtest_original(weekly)
    bt_df_rev, summary_rev, actual_sell_signal = run_backtest_revised(weekly)
    weekly['ActualSellSignal'] = actual_sell_signal
    
    st.info(f"'{stock_name}' (종목코드: {stock_code}) 분석이 완료되었습니다.")
    
    # 그래프 그리기 (수정된 방식의 거래 시점 기준)
    st.subheader(f"📈 {stock_name} 분석 차트")
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(weekly.index, weekly['Close'], label='종가', color='black'); ax1.plot(weekly.index, weekly['MA10'], label='MA10', linestyle='--', color='gray'); ax1.scatter(weekly[weekly['BuySignal'] == 1].index, weekly[weekly['BuySignal'] == 1]['Close'], color='lightcoral', marker='^', s=70, alpha=0.5, label='잠재 매수');
    if not bt_df_rev.empty:
        ax1.scatter(bt_df_rev['EntryDate'], bt_df_rev['EntryPrice'], color='red', marker='^', s=100, label='실제 매수 (현실)')
    ax1.scatter(weekly[weekly['ActualSellSignal'] == 1].index, weekly[weekly['ActualSellSignal'] == 1]['Close'], color='blue', marker='v', s=100, label='실제 매도 (현실)')
    ax1.set_ylabel('종가'); ax1.grid(True);
    ax2 = ax1.twinx()
    ax2.plot(weekly.index, weekly['FearGreedScore'], label='F&G 지수', color='darkorange')
    ax2.set_ylabel('Fear & Greed 지수', color='darkorange')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=5)
    st.pyplot(fig)

    # 결과 비교 출력
    st.divider()
    st.subheader("⚖️ 백테스트 결과 비교")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📉 기존 방식 <small>(비현실적)</small>", unsafe_allow_html=True)
        if summary_orig['total_trades'] > 0:
            st.metric("누적 수익률", f"{summary_orig['cum_return']:.2%}")
            st.metric("승률", f"{summary_orig['win_rate']:.2%}")
            with st.expander("상세 거래 내역 보기"):
                st.dataframe(bt_df_orig, use_container_width=True)
        else:
            st.warning("거래가 발생하지 않았습니다.")

    with col2:
        st.markdown("#### 📈 수정된 방식 <small>(현실적)</small>", unsafe_allow_html=True)
        if summary_rev['total_trades'] > 0:
            st.metric("누적 수익률", f"{summary_rev['cum_return']:.2%}")
            st.metric("승률", f"{summary_rev['win_rate']:.2%}")
            with st.expander("상세 거래 내역 보기"):
                st.dataframe(bt_df_rev, use_container_width=True)
        else:
            st.warning("거래가 발생하지 않았습니다.")

# --------------------------------------------------------------------------
# 🌐 웹사이트 UI 구성 (모바일 최적화)
# --------------------------------------------------------------------------
st.title("📈 주식 전략 비교 분석")
st.caption("모바일 환경에 최적화되었습니다.")

with st.expander("🔍 분석 설정하기", expanded=True):
    krx_list = get_krx_list()
    popular_stocks = ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '현대차', 'NAVER', '카카오', '삼성바이오로직스']
    other_stocks = sorted([s for s in krx_list['Name'] if s not in popular_stocks])
    stock_list = popular_stocks + other_stocks
    
    stock_name_input = st.selectbox("종목을 선택하세요", stock_list)
    start_date_input = st.date_input("분석 시작일", date.today() - timedelta(days=3 * 365))
    
    st.divider()
    if st.button("🚀 분석 실행", use_container_width=True):
        with st.spinner('데이터를 불러오고 두 가지 방식으로 분석하는 중입니다...'):
            # 메인 분석 비교 함수를 호출합니다.
            run_analysis_and_compare(stock_name_input, start_date_input)

st.divider()
st.markdown("<sub>Made with Streamlit.</sub>", unsafe_allow_html=True)
