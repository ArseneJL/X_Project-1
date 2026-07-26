import os
from datetime import datetime

import joblib
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "stock_direction_model.pkl")


FEATURE_COLUMNS = [
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
]

TARGET_COLUMN = "target_up_5d"


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


def load_training_data():
    supabase = get_supabase_client()

    df = fetch_all_rows(
        supabase=supabase,
        table_name="model_features",
        select_columns="*",
        order_columns=["symbol", "trade_date"],
    )

    if df.empty:
        raise ValueError("model_features 沒有資料，請先執行 feature_pipeline.py")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    return df


def train_model():
    df = load_training_data()

    # 先用時間切分，不用隨機切分，避免偷看未來
    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("模型訓練完成")
    print(f"資料筆數：{len(df)}")
    print(f"訓練資料：{len(train_df)}")
    print(f"測試資料：{len(test_df)}")
    print(f"Accuracy：{acc:.4f}")
    print("=" * 60)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "model_name": "random_forest_v1",
        "trained_at": datetime.now().isoformat(),
        "accuracy": acc,
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"\n模型已儲存：{MODEL_PATH}")


def preview_training_data():
    df = load_training_data()

    print("=" * 60)
    print("load_training_data() 預覽")
    print("=" * 60)

    print("資料型態：", type(df))
    print("資料筆數：", len(df))

    print("\n欄位名稱：")
    print(df.columns.tolist())

    print("\n前 5 筆資料：")
    print(df.head())

    print("\n每檔股票筆數：")
    print(df["symbol"].value_counts())

    print("\n日期範圍：")
    print(df.groupby("symbol")["trade_date"].agg(["min", "max", "count"]))

    print("\ntarget_up_5d 分布：")
    print(df["target_up_5d"].value_counts())

    print("\n特徵欄位缺失值：")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum())


if __name__ == "__main__":
    # train_model()
    preview_training_data()
