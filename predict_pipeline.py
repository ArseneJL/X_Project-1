import os
import math
from datetime import datetime

import joblib
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


MODEL_PATH = os.path.join("models", "stock_direction_model.pkl")
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


def load_model_package():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}，請先執行 train_model.py")

    return joblib.load(MODEL_PATH)


def load_latest_features(supabase):
    df = fetch_all_rows(
        supabase=supabase,
        table_name="model_features",
        select_columns="*",
        order_columns=["symbol", "trade_date"],
    )

    if df.empty:
        raise ValueError("model_features 沒有資料，請先執行 feature_pipeline.py")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    # 每檔股票取最新一筆 feature
    latest_df = (
        df.sort_values(["symbol", "trade_date"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return latest_df


def build_prediction_records(latest_df, model_package):
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]
    target_column = model_package["target_column"]
    model_name = model_package["model_name"]

    df = latest_df.copy()

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_columns)

    if df.empty:
        raise ValueError("最新 model_features 缺少必要特徵，無法預測")

    X = df[feature_columns]

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = [None] * len(df)

    df["prediction"] = predictions
    df["probability_up"] = probabilities

    df["model_name"] = model_name
    df["predict_target"] = target_column
    df["trade_date"] = df["trade_date"].dt.date.astype(str)

    records = df[
        [
            "symbol",
            "trade_date",
            "model_name",
            "predict_target",
            "prediction",
            "probability_up",
        ]
    ].to_dict(orient="records")

    return clean_records_for_json(records)


def upsert_prediction_results(supabase, records):
    if not records:
        return 0

    total = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]

        (
            supabase
            .table("prediction_results")
            .upsert(
                batch,
                on_conflict="symbol,trade_date,model_name,predict_target",
            )
            .execute()
        )

        total += len(batch)

    return total


def insert_pipeline_log(supabase, status, rows_processed=0, error_message=None):
    log_data = {
        "pipeline_name": "predict_pipeline",
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


def run_predict_pipeline():
    supabase = get_supabase_client()

    try:
        print("開始執行 predict_pipeline")

        model_package = load_model_package()
        latest_df = load_latest_features(supabase)
        records = build_prediction_records(latest_df, model_package)
        rows = upsert_prediction_results(supabase, records)

        print(f"predict_pipeline 完成，共寫入 {rows} 筆 prediction_results")

        insert_pipeline_log(
            supabase=supabase,
            status="success",
            rows_processed=rows,
            error_message=None,
        )

    except Exception as e:
        print(f"predict_pipeline 失敗：{e}")

        insert_pipeline_log(
            supabase=supabase,
            status="failed",
            rows_processed=0,
            error_message=str(e),
        )


if __name__ == "__main__":
    run_predict_pipeline()
