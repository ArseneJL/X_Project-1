import os
import math
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


BATCH_SIZE = 500


def get_supabase_client():
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("找不到 SUPABASE_URL 或 SUPABASE_SERVICE_ROLE_KEY，請檢查 .env")

    return create_client(supabase_url, supabase_key)


def fetch_all_rows(supabase, table_name, select_columns="*", batch_size=1000, order_columns=None):
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


def clean_records_for_json(records):
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


def load_source_data(supabase):
    df_prices = fetch_all_rows(
        supabase=supabase,
        table_name="daily_prices",
        select_columns="symbol, trade_date, close, volume",
        order_columns=["symbol", "trade_date"],
    )

    df_earnings = fetch_all_rows(
        supabase=supabase,
        table_name="earnings",
        select_columns="symbol, earnings_date",
        order_columns=["symbol", "earnings_date"],
    )

    df_dividends = fetch_all_rows(
        supabase=supabase,
        table_name="dividends",
        select_columns="symbol, dividend_date",
        order_columns=["symbol", "dividend_date"],
    )

    return df_prices, df_earnings, df_dividends


def add_event_flags(df_symbol, df_earnings_symbol, df_dividends_symbol):
    """
    加入事件特徵：
    has_earnings_nearby：前後 3 天內是否有財報
    has_dividend_nearby：前後 3 天內是否有除息
    """
    df_symbol = df_symbol.copy()

    df_symbol["has_earnings_nearby"] = False
    df_symbol["has_dividend_nearby"] = False

    trade_dates = pd.to_datetime(df_symbol["trade_date"])

    if not df_earnings_symbol.empty:
        earnings_dates = pd.to_datetime(df_earnings_symbol["earnings_date"], errors="coerce").dropna()

        for event_date in earnings_dates:
            mask = (trade_dates >= event_date - timedelta(days=3)) & (
                trade_dates <= event_date + timedelta(days=3)
            )
            df_symbol.loc[mask, "has_earnings_nearby"] = True

    if not df_dividends_symbol.empty:
        dividend_dates = pd.to_datetime(df_dividends_symbol["dividend_date"], errors="coerce").dropna()

        for event_date in dividend_dates:
            mask = (trade_dates >= event_date - timedelta(days=3)) & (
                trade_dates <= event_date + timedelta(days=3)
            )
            df_symbol.loc[mask, "has_dividend_nearby"] = True

    return df_symbol


def build_features_for_symbol(df_symbol, df_earnings, df_dividends, symbol):
    df = df_symbol.copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["trade_date", "close"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 報酬率特徵
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # 移動平均
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_60"] = df["close"].rolling(window=60).mean()

    # 價格相對 SMA20
    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]

    # 成交量變化
    df["volume_change_5d"] = df["volume"].pct_change(5)

    # 20 日波動率
    df["volatility_20"] = df["return_1d"].rolling(window=20).std()

    # 未來 5 日報酬與標籤
    df["target_return_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["target_up_5d"] = (df["target_return_5d"] > 0).astype(int)

    df_earnings_symbol = df_earnings[df_earnings["symbol"] == symbol] if not df_earnings.empty else pd.DataFrame()
    df_dividends_symbol = df_dividends[df_dividends["symbol"] == symbol] if not df_dividends.empty else pd.DataFrame()

    df = add_event_flags(df, df_earnings_symbol, df_dividends_symbol)

    df["symbol"] = symbol
    df["trade_date"] = df["trade_date"].dt.date.astype(str)

    keep_cols = [
        "symbol",
        "trade_date",
        "close",
        "volume",
        "return_1d",
        "return_5d",
        "return_20d",
        "sma_5",
        "sma_20",
        "sma_60",
        "price_vs_sma20",
        "volume_change_5d",
        "volatility_20",
        "has_earnings_nearby",
        "has_dividend_nearby",
        "target_return_5d",
        "target_up_5d",
    ]

    df = df[keep_cols]

    # 移除前面 rolling 不足、以及最後 5 天 target 不足的資料
    df = df.dropna(subset=[
        "return_1d",
        "return_5d",
        "return_20d",
        "sma_5",
        "sma_20",
        "sma_60",
        "price_vs_sma20",
        "volatility_20",
        "target_return_5d",
        "target_up_5d",
    ])

    return df


def build_all_features(df_prices, df_earnings, df_dividends):
    all_features = []

    symbols = sorted(df_prices["symbol"].unique())

    for symbol in symbols:
        # 指數先不納入模型訓練資料
        if symbol in ["S&P500", "NASDAQ100"]:
            continue

        print(f"建立 {symbol} model_features")

        df_symbol = df_prices[df_prices["symbol"] == symbol]

        df_features = build_features_for_symbol(
            df_symbol=df_symbol,
            df_earnings=df_earnings,
            df_dividends=df_dividends,
            symbol=symbol,
        )

        all_features.append(df_features)

    if not all_features:
        return pd.DataFrame()

    return pd.concat(all_features, ignore_index=True)


def upsert_model_features(supabase, df_features):
    if df_features.empty:
        return 0

    records = df_features.to_dict(orient="records")
    records = clean_records_for_json(records)

    total = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]

        (
            supabase
            .table("model_features")
            .upsert(batch, on_conflict="symbol,trade_date")
            .execute()
        )

        total += len(batch)

    return total


def insert_pipeline_log(supabase, status, rows_processed=0, error_message=None):
    log_data = {
        "pipeline_name": "model_features_pipeline",
        "symbol": "ALL",
        "status": status,
        "rows_processed": rows_processed,
        "error_message": error_message,
        "start_date": datetime.now().isoformat(),
    }

    try:
        supabase.table("pipeline_logs").insert(log_data).execute()
    except Exception as e:
        print(f"寫入 pipeline_logs 失敗：{e}")


def run_feature_pipeline():
    supabase = get_supabase_client()

    try:
        print("開始執行 model_features pipeline")

        df_prices, df_earnings, df_dividends = load_source_data(supabase)

        if df_prices.empty:
            raise ValueError("daily_prices 沒有資料，無法建立 model_features")

        df_features = build_all_features(
            df_prices=df_prices,
            df_earnings=df_earnings,
            df_dividends=df_dividends,
        )

        rows = upsert_model_features(supabase, df_features)

        print(f"model_features pipeline 完成，共寫入 {rows} 筆")

        insert_pipeline_log(
            supabase=supabase,
            status="success",
            rows_processed=rows,
            error_message=None,
        )

    except Exception as e:
        print(f"model_features pipeline 失敗：{e}")

        insert_pipeline_log(
            supabase=supabase,
            status="failed",
            rows_processed=0,
            error_message=str(e),
        )


if __name__ == "__main__":
    run_feature_pipeline()
