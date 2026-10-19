# Regression Lab：三個資料集

網站：https://helen861230-ux.github.io/1012-top50-startup-pages/
側欄下拉選單可切換 50 Startups、Life Expectancy、Boston Housing；六個頁面同步更新。
Boston Housing 包含 Exhaustive、Forward、Backward、RFECV、Lasso、互動特徵數曲線、完整搜尋紀錄、資料表、歷史房價預測、OLS 診斷與 Kaggle 提交檔下載。
333 筆標籤資料切成 266/67，僅在 266 筆內五折選擇。173 筆 Kaggle 測試資料無標籤，提交預測與驗證分數明確分開。
網站純 HTML/JS，不使用外部 CDN；瀏覽器依匯出係數預測，不會傳送輸入。
本機可直接開啟 index.html，或以 python -m http.server 預覽。
