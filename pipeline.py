import os
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client


# =========================================================
# 基本設定
# =========================================================

STOCK_SYMBOLS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
]

# yFinance 指數代號與你 Supabase 裡 symbol 的對應
INDEX_SYMBOL_MAP = {
    "^GSPC": "S&P500",
    "^NDX": "NASDAQ100",
}

ALL_YF_SYMBOLS = STOCK_SYMBOLS + list(INDEX_SYMBOL_MAP.keys())

BATCH_SIZE = 500


# =========================================================
# Supabase 連線
# =========================================================

def get_supabase_client():
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("找不到 SUPABASE_URL 或 SUPABASE_SERVICE_ROLE_KEY，請檢查 .env")

    return create_client(supabase_url, supabase_key)


# =========================================================
# 查詢目前資料庫最後日期
# =========================================================

def get_last_trade_date(supabase, symbol):
    """
    查詢 daily_prices 中指定 symbol 的最後日期。
    如果資料庫尚無資料，回傳 None。
    """
    response = (
        supabase
        .table("daily_prices")
        .select("trade_date")
        .eq("symbol", symbol)
        .order("trade_date", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return pd.to_datetime(response.data[0]["trade_date"]).date()


# =========================================================
# 從 yFinance 下載資料
# =========================================================

def download_price_data(yf_symbol, db_symbol, start_date, end_date):
    """
    從 yFinance 抓取日線資料，轉成 daily_prices 表需要的格式。
    """
    print(f"下載 {yf_symbol} → {db_symbol}: {start_date} ~ {end_date}")

    df = yf.download(
        tickers=yf_symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        print(f"{yf_symbol} 沒有新資料")
        return pd.DataFrame()

    # 如果 yFinance 回傳 MultiIndex 欄位，先攤平
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # yFinance 通常欄位為 Date, Open, High, Low, Close, Adj Close, Volume
    df.rename(
        columns={
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    required_cols = ["trade_date", "open", "high", "low", "close"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{yf_symbol} 缺少欄位：{missing_cols}")

    if "volume" not in df.columns:
        df["volume"] = None

    df["symbol"] = db_symbol
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    df = df[["trade_date", "symbol", "open", "high", "low", "close", "volume"]]

    # 清掉空價格資料
    df = df.dropna(subset=["trade_date", "open", "high", "low", "close"])

    # 型別整理，避免 numpy 型別造成 Supabase JSON 序列化問題
    df["trade_date"] = df["trade_date"].astype(str)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)

    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype(int)

    return df


# =========================================================
# 寫入 Supabase
# =========================================================

def upsert_daily_prices(supabase, df):
    """
    使用 upsert 寫入 daily_prices。
    需確認 Supabase daily_prices 有 unique(symbol, trade_date)。
    """
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    total = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]

        (
            supabase
            .table("daily_prices")
            .upsert(batch, on_conflict="symbol,trade_date")
            .execute()
        )

        total += len(batch)

    return total


def insert_pipeline_log(supabase, symbol, status, rows_updated=0, error_message=None):
    """
    寫入 pipeline_logs。
    如果你的 pipeline_logs 欄位名稱不同，這段需要依實際表格調整。
    """
    log_data = {
        "pipeline_name": "daily_prices_yfinance",
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


# =========================================================
# 主流程
# =========================================================

def update_one_symbol(supabase, yf_symbol):
    """
    更新單一股票或指數。
    """
    db_symbol = INDEX_SYMBOL_MAP.get(yf_symbol, yf_symbol)

    try:
        last_date = get_last_trade_date(supabase, db_symbol)

        if last_date is None:
            # 如果資料庫完全沒有資料，從 2023-01-01 開始抓
            start_date = date(2023, 1, 1)
        else:
            # 從最後一筆資料的隔天開始補
            start_date = last_date + timedelta(days=1)

        # yFinance 的 end 通常是排除日，所以用明天
        end_date = date.today() + timedelta(days=1)

        if start_date >= end_date:
            print(f"{db_symbol} 已是最新資料，略過")
            insert_pipeline_log(
                supabase=supabase,
                symbol=db_symbol,
                status="skipped",
                rows_updated=0,
                error_message=None,
            )
            return

        df = download_price_data(
            yf_symbol=yf_symbol,
            db_symbol=db_symbol,
            start_date=start_date,
            end_date=end_date,
        )

        rows_updated = upsert_daily_prices(supabase, df)

        print(f"{db_symbol} 更新完成，寫入 {rows_updated} 筆")

        insert_pipeline_log(
            supabase=supabase,
            symbol=db_symbol,
            status="success",
            rows_updated=rows_updated,
            error_message=None,
        )

    except Exception as e:
        error_message = str(e)
        print(f"{db_symbol} 更新失敗：{error_message}")

        insert_pipeline_log(
            supabase=supabase,
            symbol=db_symbol,
            status="failed",
            rows_updated=0,
            error_message=error_message,
        )


def run_daily_prices_pipeline():
    supabase = get_supabase_client()

    print("開始執行 yFinance daily_prices pipeline")

    for yf_symbol in ALL_YF_SYMBOLS:
        update_one_symbol(supabase, yf_symbol)

    print("daily_prices pipeline 執行完成")


if __name__ == "__main__":
    run_daily_prices_pipeline()
