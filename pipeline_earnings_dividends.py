import os
from datetime import datetime
import math

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client


STOCK_SYMBOLS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
]

BATCH_SIZE = 500


def get_supabase_client():
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("找不到 SUPABASE_URL 或 SUPABASE_SERVICE_ROLE_KEY，請檢查 .env")

    return create_client(supabase_url, supabase_key)


def clean_records_for_json(records):
    """
    將 NaN、inf、-inf 轉成 None，避免 Supabase JSON 序列化錯誤。
    """
    clean_records = []

    for record in records:
        clean_record = {}

        for key, value in record.items():
            if pd.isna(value):
                clean_record[key] = None
            elif isinstance(value, float) and not math.isfinite(value):
                clean_record[key] = None
            else:
                clean_record[key] = value

        clean_records.append(clean_record)

    return clean_records


def upsert_records(supabase, table_name, records, on_conflict):
    if not records:
        return 0

    records = clean_records_for_json(records)

    total = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]

        (
            supabase
            .table(table_name)
            .upsert(batch, on_conflict=on_conflict)
            .execute()
        )

        total += len(batch)

    return total


def insert_pipeline_log(supabase, symbol, status, rows_updated=0, error_message=None):
    log_data = {
        "pipeline_name": "earnings_dividends_yfinance",
        "symbol": symbol,
        "status": status,
        "rows_processed": rows_updated,
        "error_message": error_message,
        "start_date": datetime.now().isoformat(),
    }

    try:
        supabase.table("pipeline_logs").insert(log_data).execute()
    except Exception as e:
        print(f"寫入 pipeline_logs 失敗：{e}")


def fetch_earnings(symbol):
    ticker = yf.Ticker(symbol)

    try:
        df = ticker.earnings_dates
    except Exception as e:
        print(f"{symbol} earnings_dates 抓取失敗：{e}")
        return pd.DataFrame()

    if df is None or df.empty:
        print(f"{symbol} 沒有 earnings 資料")
        return pd.DataFrame()

    df = df.reset_index()

    # yFinance 通常會有 Earnings Date, EPS Estimate, Reported EPS, Surprise(%)
    rename_map = {
        "Earnings Date": "earnings_date",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_reported",
        "Surprise(%)": "surprise",
    }

    df.rename(columns=rename_map, inplace=True)

    if "earnings_date" not in df.columns:
        print(f"{symbol} earnings 缺少 earnings_date 欄位")
        return pd.DataFrame()

    df["symbol"] = symbol
    df["earnings_date"] = (
        pd.to_datetime(df["earnings_date"], errors="coerce")
        .dt.date
        .astype(str)
    )

    keep_cols = ["symbol", "earnings_date"]

    for col in ["eps_estimate", "eps_reported", "surprise"]:
        if col in df.columns:
            keep_cols.append(col)
        else:
            df[col] = None
            keep_cols.append(col)

    df = df[keep_cols]
    df = df.dropna(subset=["earnings_date"])

    for col in ["eps_estimate", "eps_reported", "surprise"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.where(pd.notnull(df), None)

    return df


def fetch_dividends(symbol):
    ticker = yf.Ticker(symbol)

    try:
        series = ticker.dividends
    except Exception as e:
        print(f"{symbol} dividends 抓取失敗：{e}")
        return pd.DataFrame()

    if series is None or series.empty:
        print(f"{symbol} 沒有 dividends 資料")
        return pd.DataFrame()

    df = series.to_frame().reset_index()

    rename_map = {
        "Date": "dividend_date",
        "Dividends": "dividend",
    }

    df.rename(columns=rename_map, inplace=True)

    if "dividend_date" not in df.columns:
        print(f"{symbol} dividends 缺少 dividend_date 欄位")
        return pd.DataFrame()

    df["symbol"] = symbol
    df["dividend_date"] = (
        pd.to_datetime(df["dividend_date"], errors="coerce")
        .dt.date
        .astype(str)
    )
    
    df["dividend"] = pd.to_numeric(df["dividend"], errors="coerce")

    df = df[["symbol", "dividend_date", "dividend"]]
    df = df.dropna(subset=["dividend_date"])
    df = df.where(pd.notnull(df), None)

    return df


def update_one_symbol(supabase, symbol):
    try:
        print(f"開始更新 {symbol} earnings / dividends")

        df_earn = fetch_earnings(symbol)
        df_div = fetch_dividends(symbol)

        earnings_rows = upsert_records(
            supabase=supabase,
            table_name="earnings",
            records=df_earn.to_dict(orient="records") if not df_earn.empty else [],
            on_conflict="symbol,earnings_date",
        )

        dividends_rows = upsert_records(
            supabase=supabase,
            table_name="dividends",
            records=df_div.to_dict(orient="records") if not df_div.empty else [],
            on_conflict="symbol,dividend_date",
        )

        total_rows = earnings_rows + dividends_rows

        print(f"{symbol} 更新完成：earnings {earnings_rows} 筆，dividends {dividends_rows} 筆")

        insert_pipeline_log(
            supabase=supabase,
            symbol=symbol,
            status="success",
            rows_updated=total_rows,
            error_message=None,
        )

    except Exception as e:
        print(f"{symbol} 更新失敗：{e}")

        insert_pipeline_log(
            supabase=supabase,
            symbol=symbol,
            status="failed",
            rows_updated=0,
            error_message=str(e),
        )


def run_earnings_dividends_pipeline():
    supabase = get_supabase_client()

    print("開始執行 earnings / dividends pipeline")

    for symbol in STOCK_SYMBOLS:
        update_one_symbol(supabase, symbol)

    print("earnings / dividends pipeline 執行完成")


if __name__ == "__main__":
    run_earnings_dividends_pipeline()
