# import libs
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client


def get_supabase_client():
    """
    建立 Supabase client。
    本機開發時讀取 .env。
    未來部署到 Hugging Face / Streamlit Cloud 時，可改用 secrets。
    """
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        st.error("找不到 SUPABASE_URL 或 SUPABASE_SERVICE_ROLE_KEY，請檢查 .env")
        st.stop()

    return create_client(supabase_url, supabase_key)


def fetch_all_rows(supabase, table_name, select_columns="*", batch_size=1000, order_columns=None):
    """
    從 Supabase 分批讀取資料，避免一次只取回前 1000 筆。
    """
    all_rows = []
    start = 0

    while True:
        end = start + batch_size - 1

        query = (
            supabase
            .table(table_name)
            .select(select_columns)
            .range(start, end)
        )

        if order_columns:
            for col in order_columns:
                query = query.order(col)

        response = query.execute()
        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < batch_size:
            break

        start += batch_size

    return pd.DataFrame(all_rows)


def load_daily_prices(supabase):
    """
    讀取 daily_prices，並轉成原本 dashboard 使用的欄位格式：
    trade_date -> date
    symbol -> name
    """
    df = fetch_all_rows(
        supabase=supabase,
        table_name="daily_prices",
        select_columns="trade_date, symbol, open, high, low, close, volume",
        order_columns=["symbol", "trade_date"],
    )

    if df.empty:
        st.error("daily_prices 沒有資料，請先確認 Supabase 是否已匯入股價資料")
        st.stop()

    df.rename(
        columns={
            "trade_date": "date",
            "symbol": "name",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(by=["name", "date"], inplace=True)

    return df


def load_earnings(supabase):
    """
    讀取 earnings。
    Supabase 欄位包含：
    earnings_date, symbol, eps_reported, eps_estimate, surprise

    轉成 dashboard 使用格式：
    date, ticker, eps_reported, eps_estimate, surprise
    """
    df = fetch_all_rows(
        supabase=supabase,
        table_name="earnings",
        select_columns="*",
        order_columns=["symbol", "earnings_date"],
    )

    if df.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    df.rename(
        columns={
            "earnings_date": "date",
            "symbol": "ticker",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(by=["ticker", "date"], inplace=True)

    return df


def load_dividends(supabase):
    """
    讀取 dividends。
    Supabase 欄位：
    dividend_date, symbol, dividend

    轉成 dashboard 使用格式：
    date, ticker, dividend
    """
    df = fetch_all_rows(
        supabase=supabase,
        table_name="dividends",
        select_columns="*",
        order_columns=["symbol", "dividend_date"],
    )

    if df.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    df.rename(
        columns={
            "dividend_date": "date",
            "symbol": "ticker",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(by=["ticker", "date"], inplace=True)

    return df


def load_prediction_results(supabase):
    """
    讀取 prediction_results。
    用於 Streamlit 顯示最新模型預測結果。
    """
    df = fetch_all_rows(
        supabase=supabase,
        table_name="prediction_results",
        select_columns="*",
        order_columns=["symbol", "trade_date"],
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "model_name",
                "predict_target",
                "prediction",
                "probability_up",
                "created_at",
            ]
        )

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["probability_up"] = pd.to_numeric(df["probability_up"], errors="coerce")

    df.sort_values(by=["symbol", "trade_date"], inplace=True)

    return df


@st.cache_data
def load_data():
    """
    統一給 app.py 呼叫。
    回傳格式維持原本：
    df_all, df_earn, df_div, df_pred
    """
    supabase = get_supabase_client()

    df_all = load_daily_prices(supabase)
    df_earn = load_earnings(supabase)
    df_div = load_dividends(supabase)
    df_pred = load_prediction_results(supabase)

    return df_all, df_earn, df_div, df_pred
