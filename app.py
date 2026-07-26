# %%
# import libs
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from database import load_data

# %%
df_all, df_earn, df_div, df_pred = load_data()

# Test for database.py
# st.write("資料筆數：", len(df_all))
# st.write("股票清單：", sorted(df_all["name"].unique()))
# st.write("財報筆數：", len(df_earn))
# st.write("股利筆數：", len(df_div))

# %%

st.set_page_config(layout="wide", page_title="美股七巨頭分析")
st.sidebar.title("設定面板")
# ticker_list = df_all["name"].unique().tolist()
ticker_list = [
    stock for stock in df_all["name"].unique().tolist()
    if stock not in ["S&P500", "NASDAQ100"]
]
selected_ticker = st.sidebar.selectbox("選擇股票", ticker_list, index=0)

# 技術指標選擇
st.sidebar.write("")
st.sidebar.subheader("[技術指標]")
index_choose = st.sidebar.radio(
    "選擇要比較的大盤指數",
    ("NA", "S&P500", "NASDAQ100"),
    index=0,
)
show_ma = st.sidebar.checkbox("顯示移動平均線 (SMA)", value=True)
ma_window = st.sidebar.slider("SMA 週期", 5, 200, 20)
show_volume = st.sidebar.checkbox("顯示成交量", value=True)
df_stock = (
    df_all[df_all["name"] == selected_ticker].sort_values("date").reset_index(drop=True)
)
df_SP500 = df_all[df_all["name"] == "S&P500"].sort_values("date").reset_index(drop=True)
df_NASDAQ100 = df_all[df_all["name"] == "NASDAQ100"].sort_values("date").reset_index(drop=True)

# 日期區間
min_date = df_all["date"].min().date()
max_date = df_all["date"].max().date()

st.sidebar.write("")
st.sidebar.subheader("[選擇分析區間]")
# 目前選擇第一個日期時會先跑錯誤，選完第二個才正常顯示，因此加上預設值避免錯誤
date_range = st.sidebar.date_input(
    "調整日期範圍",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if not isinstance(date_range, tuple) or len(date_range) != 2: 
    st.info("請選擇起始與結束日期")
    st.stop()    

start_date, end_date = date_range

df_stock_plot = df_stock[
    (df_stock["date"].dt.date >= start_date)
    & (df_stock["date"].dt.date <= end_date)
].copy()

df_SP500_plot = df_SP500[
    (df_SP500["date"].dt.date >= start_date)
    & (df_SP500["date"].dt.date <= end_date)
].copy()

df_NASDAQ100_plot = df_NASDAQ100[
    (df_NASDAQ100["date"].dt.date >= start_date)
    & (df_NASDAQ100["date"].dt.date <= end_date)
].copy()

# 確保日期格式一致 (將 pandas timestamp 轉為 date 進行比較)
mask = (df_all["date"].dt.date >= start_date) & (df_all["date"].dt.date <= end_date)
df_filtered = df_all.loc[mask]

if df_filtered.empty:
    st.warning("⚠️ 所選日期範圍內無資料，請重新選擇。")
    st.stop()

st.subheader(f"{selected_ticker} 股價資訊")

# 計算指標 (SMA)
if show_ma:
    df_stock_plot[f"SMA_{ma_window}"] = df_stock_plot["close"].rolling(window=ma_window).mean()

# 計算漲跌幅 (%)
latest_close = df_stock["close"].iloc[-1]
prev_close = df_stock["close"].iloc[-2]
change = latest_close - prev_close
pct_change = (change / prev_close) * 100

# 事件
df_earn['event_type'] = 'E'
df_earn['label'] = 'E'
df_div['event_type'] = 'D'
df_div['label'] = 'D'


events = pd.concat([
    df_earn[df_earn["ticker"] == selected_ticker][["date", "event_type", "label"]],
    df_div[df_div["ticker"] == selected_ticker][["date", "event_type", "label"]],
], ignore_index=True).sort_values(by="date")

events = events[
    (events["date"].dt.date >= start_date)
    & (events["date"].dt.date <= end_date)
].copy()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label=f"{selected_ticker} 現價",
        value=f"${latest_close:.2f}",
        delta=f"{pct_change:.2f}%",
    )
with col2:
    st.metric(label="兩年內最高", value=f"${df_stock['high'].max():.2f}")
with col3:
    st.metric(label="兩年內最低", value=f"${df_stock['low'].min():.2f}")
with col4:
    st.metric(label="兩年總成交量", value=f"{(df_stock['volume'].sum()/1000000):,.2f}M")

st.write("")
st.write("")
st.write("")

# %%
# 模型預測結果
st.subheader("模型預測結果")

if df_pred.empty:
    st.info("目前尚無模型預測結果，請先執行 predict_pipeline.py")
else:
    df_pred_stock = (
        df_pred[df_pred["symbol"] == selected_ticker]
        .sort_values(["trade_date", "created_at"])
        .tail(1)
    )

    if df_pred_stock.empty:
        st.info(f"{selected_ticker} 目前沒有模型預測結果")
    else:
        pred_row = df_pred_stock.iloc[0]

        prediction = int(pred_row["prediction"]) if pd.notna(pred_row["prediction"]) else None
        probability_up = pred_row["probability_up"]
        trade_date = pred_row["trade_date"]
        model_name = pred_row["model_name"]
        predict_target = pred_row["predict_target"]

        if prediction == 1:
            prediction_text = "未來 5 日上漲"
        elif prediction == 0:
            prediction_text = "未來 5 日未上漲"
        else:
            prediction_text = "無法判斷"

        probability_text = (
            f"{probability_up * 100:.2f}%"
            if pd.notna(probability_up)
            else "N/A"
        )

        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

        with pred_col1:
            st.metric(
                label="模型判斷",
                value=prediction_text,
            )

        with pred_col2:
            st.metric(
                label="上漲機率",
                value=probability_text,
            )

        with pred_col3:
            st.metric(
                label="模型名稱",
                value=model_name,
            )

        with pred_col4:
            st.metric(
                label="預測基準日",
                value=trade_date.strftime("%Y-%m-%d") if pd.notna(trade_date) else "N/A",
            )

        st.caption(
            f"預測目標：{predict_target}。此結果僅為模型訊號展示，不代表投資建議。"
        )

# %%
st.subheader("📈 股價走勢與大盤比較分析")

# 建立子圖表 (上方 K 線，下方成交量)
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("Price", "Volume"),
    row_width=[0.2, 0.7],
    specs=[[{"secondary_y": True}],  # 關鍵：上方圖表啟用第二 Y 軸
           [{"secondary_y": False}]],
)

# K 線圖
fig.add_trace(
    go.Candlestick(
        x=df_stock_plot["date"],
        open=df_stock_plot["open"],
        high=df_stock_plot["high"],
        low=df_stock_plot["low"],
        close=df_stock_plot["close"],
        name="OHLC",
    ),
    row=1,
    col=1,
    secondary_y=False,
)

# S&P500 & NASDAQ100 指數線
if index_choose == "S&P500":
    fig.add_trace(
        go.Scatter(
            x=df_SP500_plot["date"],
            y=df_SP500_plot["close"],
            opacity=0.7,
            line=dict(color="yellow", width=2),
            name="S&P500",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
elif index_choose == "NASDAQ100":
    fig.add_trace(
        go.Scatter(
            x=df_NASDAQ100_plot["date"],
            y=df_NASDAQ100_plot["close"],
            opacity=0.7,
            line=dict(color="lightblue", width=2),
            name="NASDAQ100",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

# 移動平均線
if show_ma:
    fig.add_trace(
        go.Scatter(
            x=df_stock_plot["date"],
            y=df_stock_plot[f"SMA_{ma_window}"],
            opacity=0.7,
            line=dict(color="orange", width=2),
            name=f"SMA {ma_window}",
        ),
        row=1,
        col=1,
    )

# 成交量
if show_volume:
    fig.add_trace(
        go.Bar(
            x=df_stock_plot["date"], y=df_stock_plot["volume"], name="Volume", marker_color="grey"
        ),
        row=2,
        col=1,
    )

# 圖表版面設定
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=600,
    margin=dict(l=20, r=20, t=40, b=20),
    # plot_bgcolor="#161616",
    # paper_bgcolor="#161616",
    font=dict(size=24, color="white"),
)

# 設定 Y 軸標籤
fig.update_yaxes(
    title_text=f"{selected_ticker} 價格",
    title_font=dict(size=20, color="white"),
    secondary_y=False,
    row=1,
    col=1,
)
fig.update_yaxes(
    title_text=f"{index_choose}指數",
    title_font=dict(size=20, color="white"),
    secondary_y=True,
    row=1,
    col=1,
    showgrid=False,  # 隱藏次軸的網格線避免混亂
)
fig.update_yaxes(
    title_text="成交量", title_font=dict(size=20, color="white"), row=2, col=1
)
fig.update_xaxes(
    range=[start_date, end_date],
    row=1,
    col=1,
)

fig.update_xaxes(
    range=[start_date, end_date],
    row=2,
    col=1,
)

for _, event in events.iterrows():
    fig.add_annotation(
        x=event['date'],
        y=0,
        xref="x",
        yref="y domain",
        text=event['label'],
        showarrow=False,
        font=dict(color="black" if event['event_type']=='E' else "blue", size=10),
        bgcolor="lightgray" if event['event_type']=='E' else "lightblue",
        bordercolor="black",
    )

st.plotly_chart(fig, width='stretch')
st.caption("E: 財報, D: 除權息日")
st.caption("區間: 2023-01-01 至 2024-12-31")

# %%
# 漲跌幅排行
st.subheader(f"{start_date} 至 {end_date} 漲跌幅")

# 計算區間漲跌幅 (%)
date_filtered = (df_all["date"].dt.date >= start_date) & (df_all["date"].dt.date <= end_date)

cols = st.columns(4)
count = 0
best_stock = None
for i in ticker_list:
    closes = df_all.loc[date_filtered & (df_all["name"] == i), "close"]
    if closes.empty:
        # 沒有該股票在區間的資料，跳過
        continue
    each_last = closes.iloc[-1]
    each_prev = closes.iloc[0]
    each_change = each_last - each_prev
    each_pct_change = (each_change / each_prev) * 100
    # 初始化或更新最佳股票
    if best_stock is None or each_pct_change > best_stock[1]:
        best_stock = (each_last, each_pct_change, i)
    with cols[count]:
        st.metric(
            label=f"{i}收盤價",
            value=f"${each_last:.2f}",
            delta=f"{each_pct_change:.2f}%",
        )
    count += 1
    if count >= 4:
        count = 0

# 顯示最佳表現股票（放到新的欄位列的第一個位置）
if best_stock is not None:
    with cols[count]:
        st.metric(
            label=f"{best_stock[2]} 表現最佳",
            value=f"${best_stock[0]:.2f}",
            delta=f"{best_stock[1]:.2f}%",
        )
