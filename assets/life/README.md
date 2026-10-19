# WHO Life Expectancy 線性迴歸分析

先開啟 analysis_report.html 閱讀中文分析與圖表。HTML 圖表內嵌，可離線閱讀。

## 重現
1. Python 3.11+ 建議建立獨立虛擬環境。
2. pip install -r requirements.txt
3. python analysis.py
或開啟 life_expectancy_linear_regression.ipynb，Run All（notebook 工作目錄設為此資料夾）。

## 檔案
- Life-Expectancy-Data-Updated.csv：指定 Kaggle 更新版原檔，CC0。
- life_expectancy_linear_regression.ipynb：含執行結果的教學 Notebook。
- analysis.py：可重跑主程式。
- model_metrics.csv：四個比較模型的訓練／測試分數。
- unseen_country_metrics.csv：獨立國家切分的測試分數。
- group_cv.csv：原訓練集內 GroupKFold 各折分數。
- coefficients.csv / ols_summary.txt：OLS 係數、國家群聚穩健區間與摘要。
- vif.csv / correlations.csv / descriptive_statistics.csv：資料探索與診斷。
- test_predictions.csv：固定測試集全部預測。
- analysis_summary.json：版本、尺度、基準類別與資料雜湊。
- figures/：5 張 PNG 圖表。

## 結果
完整模型隨機測試 R²=0.9822，RMSE=1.214 歲，MAE=0.979 歲。
未見國家測試 RMSE=1.265 歲。
數值是實際執行所得，不是引用範例。資料本身已補值，模型估計同年度關聯，並非因果或未來預測。

## 來源
- https://www.kaggle.com/datasets/lashagoch/life-expectancy-who-updated
- https://github.com/Vipul97/life-expectancy-prediction
下載日期：2026-10-13；Kaggle version 1。
CSV SHA-256：b1d563472a6e1963bdc22b552edc344ec7f0748f34b6d902619a094ebce12123
