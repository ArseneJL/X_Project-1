from datetime import datetime

from pipeline import run_daily_prices_pipeline
from pipeline_earnings_dividends import run_earnings_dividends_pipeline
from feature_pipeline import run_feature_pipeline
from predict_pipeline import run_predict_pipeline


def main():
    start_time = datetime.now()

    print("=" * 60)
    print("開始執行完整資料更新與預測流程")
    print(f"開始時間：{start_time}")
    print("=" * 60)

    print("\n[1/4] 更新 daily_prices")
    run_daily_prices_pipeline()

    print("\n[2/4] 更新 earnings / dividends")
    run_earnings_dividends_pipeline()

    print("\n[3/4] 更新 model_features")
    run_feature_pipeline()

    print("\n[4/4] 更新 prediction_results")
    run_predict_pipeline()

    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print("=" * 60)
    print("完整資料更新與預測流程執行完成")
    print(f"結束時間：{end_time}")
    print(f"花費時間：{elapsed_time}")
    print("=" * 60)


if __name__ == "__main__":
    main()
