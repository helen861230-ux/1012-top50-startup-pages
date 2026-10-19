# Boston Housing：五種特徵選擇

資料：333 筆有標籤資料、173 筆 Kaggle 未標籤資料；13 個原始輸入欄位，ID 不使用。
來源：https://www.kaggle.com/c/boston-housing
本次使用的公開鏡像：https://gist.github.com/Abhinav-Mahapatra/b23f70e2d57ac182a25372b9bb8d4792
Kaggle 直接下載回傳 HTTP 401，鏡像明確標示此競賽。保留原 TSV 並等值轉成 CSV，hash 見 provenance.json。沒有讀取未標籤資料的答案。

## 執行
建議 Python 3.12。
pip install -r requirements.txt
python analysis.py
或在此目錄以 Jupyter 開啟 boston_housing_analysis.ipynb，Run All。Notebook 已執行並保留輸出。

## 方法與評估
以 seed=42 將 333 筆切成 266 訓練／67 保留驗證。五種方法共用訓練內五折（shuffle=True, seed=42），StandardScaler 在每折內 fit。
Exhaustive 搜尋 8191 個非空組合；Forward/Backward 完整逐步路徑；RFECV 每折依標準化 OLS 係數消除；Lasso 搜尋 61 個 alpha（1e-4–10），保留正則化模型。
依訓練平均 CV RMSE 選擇，保留驗證結果不參與挑方法、特徵或 alpha。CV 分數參與選擇，並非 nested CV 無偏分數。
圖表的 per-count 模型亦僅由訓練資料決定。Lasso 缺失特徵數不補造、不跨缺點畫線。

| method     |   n_features | features                                                                              |   cv_rmse |   test_rmse |   test_r_squared |
|:-----------|-------------:|:--------------------------------------------------------------------------------------|----------:|------------:|-----------------:|
| Backward   |           10 | zn | chas | nox | rm | dis | rad | tax | ptratio | black | lstat                      |    5.1033 |      4.7920 |           0.7449 |
| Exhaustive |           10 | zn | chas | nox | rm | dis | rad | tax | ptratio | black | lstat                      |    5.1033 |      4.7920 |           0.7449 |
| Forward    |           10 | zn | chas | nox | rm | dis | rad | tax | ptratio | black | lstat                      |    5.1033 |      4.7920 |           0.7449 |
| Lasso      |           13 | crim | zn | indus | chas | nox | rm | age | dis | rad | tax | ptratio | black | lstat |    5.1539 |      4.8411 |           0.7396 |
| RFECV      |           13 | crim | zn | indus | chas | nox | rm | age | dis | rad | tax | ptratio | black | lstat |    5.1539 |      4.8463 |           0.7390 |

## 提交檔
預設 submission.csv 使用依訓練 CV 排序後的 Backward。三種方法如分數相同，按特徵數及方法名稱穩定排序。保留原先選定欄位／alpha，重新以全部 333 筆擬合，預測 173 筆 Kaggle 測試資料。
results/submission_*.csv 提供五種方法各自的 ID,medv 預測。未自動提交 Kaggle，也未宣稱排行榜分數。
selection_models.json 為 266 筆驗證模型；submission_models.json 為 333 筆提交模型。兩種係數分開保存。

## 資料與解讀限制
medv 是區域自住房價中位數，歷史美元千元；不是每棟房屋的售價。原始資料為 1970 年代，目標在 50 有上限。本流程保留原始 13 欄作方法重現；black/B 是已知有建構問題的種族轉換指標，係數不作因果或現代實務估價用途。
https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html
全特徵 OLS 診斷與五種特徵選擇分開；不因保留驗證殘差而刪除資料。

## 檔案
analysis.py、feature_selection.py：完整分析與五種方法。
boston_housing_analysis.ipynb：已執行教學 Notebook。
train.csv、test.csv、source_*.tsv、provenance.json：資料與追溯。
results/selection_comparison.csv：五種方法的選定結果。
results/selection_cv_history.csv：全部候選與各折 RMSE。
results/selection_feature_curves.csv、selection_curve_predictions.csv：曲線與逐筆預測。
results/selection_protocol.json：切分列號、folds、路徑、RFECV 每折集合、alpha 網格。
results/selection_predictions.csv：保留驗證逐筆預測。
results/ols_*、vif.csv：全特徵 OLS 診斷。
web_data.json：互動網站使用的模型、資料與結果。
