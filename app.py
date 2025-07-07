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
# 🖥️ 기본 설정 및 캐싱 (모바일 최적화)
# --------------------------------------------------------------------------
# 페이지 기본 설정: 'centered' 레이아웃으로 변경
st.set_page_config(
    page_title="주식 분석 | 모바일",
    page_icon="📱",
    layout="centered",  # 모바일 친화적인 중앙 정렬 레이아웃
)


# 폰트 설정 (한 번만 실행)
@st.cache_resource
def set_font():
    if platform.system() == 'Windows':
        path = "c:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)
    elif platform.system() == 'Darwin':  # Mac OS
        rc('font', family='AppleGothic')
    else:  # Linux
        rc('font', family='NanumGothic')
    mpl.rcParams['axes.unicode_minus'] = False


set_font()


# KRX 종목 리스트 불러오기 (캐싱)
@st.cache_data
def get_krx_list():
    return fdr.StockListing('KRX')


# --------------------------------------------------------------------------
# 📈 메인 분석 함수 (그래프, 결과 출력 부분 수정)
# --------------------------------------------------------------------------
def run_analysis(stock_name, start_date):
    krx_list = get_krx_list()

    # 1) 종목코드 찾기
    try:
        stock_code = krx_list[krx_list['Name'] == stock_name]['Code'].iloc[0]
    except IndexError:
        st.error("해당 종목을 찾을 수 없습니다. 종목명을 정확히 입력해주세요.")
        return

    # 2) 데이터 불러오기 및 전처리
    end_date = date.today()
    df = fdr.DataReader(stock_code, start=start_date, end=end_date)
    if df.empty:
        st.error("해당 기간의 데이터가 없습니다. 시작일을 확인해주세요.")
        return
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = df[df[['Open', 'High', 'Low', 'Close']].ne(0).all(axis=1)]

    # 4) 주간 데이터 및 지표 계산 (기존과 동일)
    weekly = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    weekly['Prev_High'] = weekly['High'].shift(1)
    weekly['Prev_Low'] = weekly['Low'].shift(1)
    weekly['MA10'] = weekly['Close'].rolling(window=10).mean()
    mf_multiplier = ((weekly['Close'] - weekly['Low']) - (weekly['High'] - weekly['Close'])) / (
                weekly['High'] - weekly['Low'])
    mf_volume = mf_multiplier * weekly['Volume']
    weekly['CMF'] = mf_volume.rolling(window=4).sum() / weekly['Volume'].rolling(window=4).sum()
    weekly['BuySignal'] = 0
    weekly['SellSignal'] = 0
    weekly.loc[(weekly['High'] > weekly['Prev_High']) & (weekly['Close'] > weekly['MA10']) & (
                weekly['CMF'] > 0), 'BuySignal'] = 1
    weekly.loc[(weekly['Low'] < weekly['Prev_Low']) & (weekly['Close'] < weekly['MA10']) & (
                weekly['CMF'] < 0), 'SellSignal'] = 1

    # ... (백테스트, Fear & Greed 계산 로직은 기존과 동일하게 유지)
    # 7) 백테스트 로직 (생략 - 기존 코드와 동일)
    position = False;
    entries, exits = [], [];
    entry_date_temp = None
    for i in range(1, len(weekly)):
        row = weekly.iloc[i]
        if not position and row['BuySignal'] == 1:
            entry_date_temp = row.name;
            entry_price = row['Open'];
            position = True
        elif position and row['SellSignal'] == 1:
            exit_date = row.name;
            exit_price = row['Close'];
            entries.append((entry_date_temp, entry_price));
            exits.append((exit_date, exit_price));
            position = False
    if position and entry_date_temp:
        last_row = weekly.iloc[-1];
        entries.append((entry_date_temp, entry_price));
        exits.append((last_row.name, last_row['Close']))
    bt_df = pd.DataFrame()
    if entries:
        bt_df = pd.DataFrame(entries, columns=['EntryDate', 'EntryPrice']);
        bt_df['ExitDate'], bt_df['ExitPrice'] = zip(*exits);
        bt_df['Return'] = (bt_df['ExitPrice'] - bt_df['EntryPrice']) / bt_df['EntryPrice'];
        bt_df['CumulativeReturn'] = (1 + bt_df['Return']).cumprod()

    # 8) Fear & Greed 지수 계산 (생략 - 기존 코드와 동일)
    weekly['Momentum5'] = (np.log(weekly['Close']) - np.log(weekly['Close'].shift(5))) * 100
    rolling_low = weekly['Close'].rolling(window=52, min_periods=1).min();
    rolling_high = weekly['Close'].rolling(window=52, min_periods=1).max()
    weekly['Position52W'] = ((weekly['Close'] - rolling_low) / (rolling_high - rolling_low)).clip(0, 1)
    # ... (나머지 계산은 기존 코드와 동일)
    recent_vol_avg = weekly['Volume'].rolling(window=5, min_periods=1).mean();
    past_vol_avg = weekly['Volume'].rolling(window=20, min_periods=1).mean()
    weekly['VolumeSurge'] = (recent_vol_avg / past_vol_avg).clip(0, 3)
    weekly_return = weekly['Close'].pct_change();
    recent_vol = weekly_return.rolling(window=5, min_periods=1).std();
    past_vol = weekly_return.rolling(window=20, min_periods=1).std()
    weekly['VolatilitySpike'] = (recent_vol / past_vol).clip(0, 3)
    momentum_score = (weekly['Momentum5'].rolling(window=7, min_periods=1).mean() / 10).clip(-1, 1.5)
    position_score = (2 * weekly['Position52W'].rolling(window=7, min_periods=1).mean() - 1).clip(-1, 1.5)
    volume_score = (weekly['VolumeSurge'].rolling(window=10, min_periods=1).mean() - 1).clip(-0.5, 1.2)
    volatility_score = -(weekly['VolatilitySpike'].rolling(window=10, min_periods=1).mean() - 1).clip(-0.5, 1.2)
    weekly['FearGreedScore'] = (
                0.45 * momentum_score + 0.45 * position_score + 0.05 * volume_score + 0.05 * volatility_score)

    # 9) 그래프 그리기
    st.subheader(f"📈 {stock_name} ({stock_code}) 분석 차트")
    # 그래프 크기를 모바일에 맞게 수정 (10, 7)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    # ... (그래프 그리는 나머지 코드는 기존과 동일)
    ax1.plot(weekly.index, weekly['Close'], label='종가', color='black');
    ax1.plot(weekly.index, weekly['MA10'], label='MA10', linestyle='--', color='gray');
    ax1.scatter(weekly[weekly['BuySignal'] == 1].index, weekly[weekly['BuySignal'] == 1]['Close'], color='lightcoral',
                marker='^', s=70, alpha=0.5, label='잠재 매수');
    if not bt_df.empty:
        ax1.scatter(bt_df['EntryDate'] + pd.Timedelta(days=1), bt_df['EntryPrice'], color='red', marker='^', s=100,
                    label='실제 매수')
    weekly['ActualSellSignal'] = 0;
    position = False
    for i in range(1, len(weekly)):
        if not position and weekly.iloc[i]['BuySignal'] == 1:
            position = True
        elif position and weekly.iloc[i]['SellSignal'] == 1:
            weekly.loc[weekly.index[i], 'ActualSellSignal'] = 1; position = False
    ax1.scatter(weekly[weekly['ActualSellSignal'] == 1].index, weekly[weekly['ActualSellSignal'] == 1]['Close'],
                color='blue', marker='v', s=100, label='실제 매도')
    ax1.set_ylabel('종가');
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(weekly.index, weekly['FearGreedScore'], label='F&G 지수', color='darkorange')
    ax2.axhline(0.5, color='r', linestyle='--', linewidth=0.8);
    ax2.axhline(-0.5, color='g', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Fear & Greed 지수', color='darkorange')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=5)  # 범례 위치 하단으로
    st.pyplot(fig)

    # 10) 백테스트 결과 출력
    st.subheader("📊 백테스트 결과 요약")
    if not bt_df.empty:
        # 결과를 2x2 그리드로 표시
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        total_trades = len(bt_df)
        avg_return = bt_df['Return'].mean()
        cum_return = bt_df['CumulativeReturn'].iloc[-1] - 1
        win_rate = (bt_df['Return'] > 0).mean()

        col1.metric("총 트레이드 수", f"{total_trades} 회")
        col2.metric("승률", f"{win_rate:.2%}")
        col3.metric("평균 수익률", f"{avg_return:.2%}")
        col4.metric("누적 수익률", f"{cum_return:.2%}")

        # 상세 내역은 펼치기/접기 메뉴 안에 표시
        with st.expander("상세 거래 내역 보기"):
            st.dataframe(bt_df.style.format({
                'EntryPrice': '{:,.0f}', 'ExitPrice': '{:,.0f}', 'Return': '{:.2%}', 'CumulativeReturn': '{:.2f}'
            }))
    else:
        st.warning("분석 기간 동안 트레이드가 발생하지 않았습니다.")


# --------------------------------------------------------------------------
# 🌐 웹사이트 UI 구성 (모바일 최적화)
# --------------------------------------------------------------------------
st.title("📈 주식 전략 분석")
st.caption("모바일 환경에 최적화되었습니다.")

# 사이드바 대신 expander를 사용해 메인 화면에 설정 메뉴 배치
with st.expander("🔍 분석 설정하기", expanded=True):
    krx_list = get_krx_list()
    # 인기 종목을 상단에 배치하여 선택 용이성 증대
    popular_stocks = ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '현대차', 'NAVER', '카카오', '삼성바이오로직스']
    other_stocks = sorted([s for s in krx_list['Name'] if s not in popular_stocks])
    stock_list = popular_stocks + other_stocks

    stock_name_input = st.selectbox("종목을 선택하세요", stock_list)
    start_date_input = st.date_input("분석 시작일", date.today() - timedelta(days=3 * 365))

    # 분석 버튼을 중앙에 크게 배치
    st.divider()
    if st.button("🚀 분석 실행", use_container_width=True):
        with st.spinner('데이터를 불러오고 분석하는 중입니다...'):
            run_analysis(stock_name_input, start_date_input)

st.divider()
st.markdown("<sub>Made with Streamlit</sub>", unsafe_allow_html=True)