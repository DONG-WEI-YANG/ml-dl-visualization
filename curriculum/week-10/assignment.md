# 第 10 週作業：超參數調校實驗與學習曲線分析
# Week 10 Assignment: Hyperparameter Tuning Experiment & Learning Curve Analysis

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 至課程平台

---

## 作業一：Grid Search 實作與分析（20%）

使用 scikit-learn 的 Wine 資料集 (`sklearn.datasets.load_wine`)，對 **SVM 分類器** 進行 Grid Search：

1. 定義以下搜尋空間：
   - `C`: [0.01, 0.1, 1, 10, 100]
   - `gamma`: [0.001, 0.01, 0.1, 1]
   - `kernel`: ['rbf', 'poly']
2. 使用 `GridSearchCV` 搭配 5-fold 交叉驗證
3. 印出最佳參數組合與最佳分數
4. 繪製 C 與 gamma 的**搜尋結果熱力圖 (Heatmap)**（固定 kernel='rbf'）
5. 分析：為什麼某些參數組合表現特別好或特別差？（至少 100 字）

## 作業二：Random Search 與比較（20%）

1. 對同一資料集與模型，使用 `RandomizedSearchCV`：
   - `C`: `loguniform(0.01, 100)`
   - `gamma`: `loguniform(0.001, 1)`
   - `kernel`: ['rbf', 'poly']
   - `n_iter=40`
2. 印出最佳參數與分數
3. 比較 Grid Search 與 Random Search：
   - 最佳分數差異
   - 總評估次數
   - 實際執行時間（使用 `%%time` 或 `time` 模組計時）
4. 撰寫比較分析：在此案例中，哪種方法更有效率？為什麼？（至少 100 字）

## 作業三：學習曲線分析（25%）

選擇**兩個不同複雜度的模型**（例如：簡單的 Decision Tree with max_depth=2 vs 複雜的 Random Forest with n_estimators=200），使用 Wine 資料集：

1. 分別繪製兩個模型的學習曲線 (Learning Curve)
   - 使用 `sklearn.model_selection.learning_curve`
   - 訓練集比例從 10% 到 100%，取 10 個點
   - 包含訓練分數與驗證分數的平均值和標準差帶 (shaded area)
2. 將兩條學習曲線繪製在同一張圖上（或並排的子圖 subplots）
3. 分析每個模型的學習曲線：
   - 是否存在過擬合或欠擬合？
   - 增加資料量是否有幫助？
   - 兩個模型在表現上有何差異？（至少 150 字）

## 作業四：驗證曲線分析（20%）

使用 Random Forest 模型與 Wine 資料集：

1. 繪製 `max_depth` 的驗證曲線 (Validation Curve)
   - `param_range`: 1 到 20 的整數
   - 包含訓練分數與驗證分數
2. 繪製 `n_estimators` 的驗證曲線
   - `param_range`: [10, 25, 50, 100, 200, 300, 500]
3. 根據驗證曲線，回答以下問題：
   - `max_depth` 的最佳值大約是多少？過大會怎樣？
   - `n_estimators` 增加到多少時效益開始遞減？
   - 如果只能選一個超參數調校，你會先調哪一個？為什麼？（至少 100 字）

## 作業五：綜合調校報告（15%）

基於作業一到四的結果，撰寫一份簡短的調校報告，包含：

1. **最終模型選擇：** 你最終選擇哪個模型和哪組超參數？理由是什麼？
2. **調校流程總結：** 你使用了什麼策略來進行調校？（先粗後細？先調哪個參數？）
3. **經驗反思：** 在這次實驗中，你學到了什麼關於超參數調校的經驗？
4. **若有更多時間：** 你還會嘗試什麼？（例如：其他模型、Optuna、更大的搜尋空間）

報告字數：至少 300 字。

---

## 加分題 Bonus（+10%）

使用 Optuna 對 Gradient Boosting Classifier 進行超參數調校：

1. 安裝 Optuna：`pip install optuna`
2. 定義 objective 函數，搜尋以下超參數：
   - `n_estimators`: 50-500
   - `learning_rate`: 0.01-0.3（對數尺度）
   - `max_depth`: 3-10
   - `subsample`: 0.6-1.0
3. 執行 50 次試驗 (trials)
4. 使用 Optuna 內建視覺化繪製：
   - 最佳化歷程圖 (Optimization History)
   - 超參數重要度圖 (Parameter Importances)
5. 比較 Optuna 找到的最佳結果與 Grid/Random Search 的結果

---

## 評分標準 Grading Criteria
- 程式碼正確執行 Code Execution: 40%
- 視覺化品質 Visualization Quality: 25%
- 分析深度與反思 Analysis Depth & Reflection: 25%
- 程式碼風格與註解 Code Style & Comments: 10%
