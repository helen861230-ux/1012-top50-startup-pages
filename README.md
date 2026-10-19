# Regression Lab：50 Startups + Life Expectancy

同一網站以側欄「資料集」下拉選單切換兩個分析，保留各自的模型、資料表、預測、診斷及下載。

直接開啟 index.html，或 python -m http.server 8000 後開啟 http://localhost:8000。
無外部 JavaScript CDN，不會上傳預測輸入。Life Expectancy 模型由 Python 訓練，瀏覽器使用原尺度係數，逐筆核對全部 573 筆測試預測，與 Python 相符。

Life Expectancy 的模型比較採用已完成的四個預先指定模型，不冒稱執行 50 Startups 的五種特徵選擇。
固定測試切分：Life random_state=42；Startups random_state=0。CV 的方法與目的在各自頁面標示。

網站：https://helen861230-ux.github.io/1012-top50-startup-pages/
