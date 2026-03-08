# 第 7 週作業：樹模型與集成方法
# Week 7 Assignment: Tree Models & Ensemble Methods

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 至課程平台
**預計所需時間 Estimated Time:** 3-4 小時

---

## 作業一：決策樹建構與視覺化（20%）

使用 scikit-learn 的 Iris 資料集或 Wine 資料集完成以下任務：

1. **建構決策樹分類器** (DecisionTreeClassifier)，分別使用 `criterion='gini'` 和 `criterion='entropy'`，比較兩者的準確率差異。

2. **視覺化決策樹結構：** 使用 `sklearn.tree.plot_tree()` 或 `export_graphviz()` 繪製完整的決策樹圖，標註每個節點的分裂條件、Gini/Entropy 值、樣本數與類別分佈。

3. **分析與回答：**
   - 根節點選擇了哪個特徵進行第一次分裂？為什麼？
   - 比較 Gini 和 Entropy 兩棵樹的結構差異，差異大嗎？
   - 哪些特徵在樹中從未被使用？這代表什麼意義？

## 作業二：剪枝實驗（20%）

1. **預剪枝實驗：** 在 Wine 資料集上，分別嘗試以下 `max_depth` 值：1, 2, 3, 5, 10, None（不限制），記錄訓練集與測試集的準確率。

2. **繪製圖表：**
   - X 軸為 `max_depth`，Y 軸為準確率
   - 同時繪製訓練集與測試集的準確率曲線
   - 標記最佳 `max_depth` 值

3. **後剪枝實驗（加分題 +5%）：**
   - 使用 `DecisionTreeClassifier` 的 `cost_complexity_pruning_path()` 方法取得有效的 `ccp_alpha` 值
   - 繪製不同 `ccp_alpha` 下的訓練/測試準確率曲線
   - 找出最佳 `ccp_alpha`

4. **分析與回答：**
   - 什麼時候訓練準確率和測試準確率的差距最大？這代表什麼？
   - 預剪枝和後剪枝得到的最佳樹結構是否相似？

## 作業三：隨機森林超參數探索（25%）

使用 scikit-learn 的 Wine 資料集或 Breast Cancer Wisconsin 資料集：

1. **n_estimators 影響分析：** 測試 n_estimators = [1, 5, 10, 20, 50, 100, 200, 500]，繪製準確率隨樹數增加的變化曲線。同時記錄 OOB 分數（設定 `oob_score=True`）。

2. **max_features 影響分析：** 固定 n_estimators=200，測試 max_features = ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]，比較準確率。

3. **特徵重要度 (Feature Importance)：** 訓練一個 Random Forest 模型，繪製特徵重要度的水平長條圖（由高到低排序）。

4. **分析與回答：**
   - 大約需要多少棵樹才能讓準確率趨於穩定？
   - OOB 分數和交叉驗證分數的差距大嗎？
   - 最重要的前 3 個特徵是什麼？它們為什麼重要？

## 作業四：GBDT 實驗與比較（25%）

1. **learning_rate 與 n_estimators 交互作用：** 使用 `GradientBoostingClassifier`，測試以下組合：
   - learning_rate = [0.01, 0.05, 0.1, 0.3]
   - n_estimators = [50, 100, 200, 500]

   繪製一個 4x4 的熱力圖 (Heatmap)，顯示每個組合的測試準確率。

2. **三大模型比較：** 在同一資料集上比較以下三個模型的效能：
   - 單棵決策樹 (DecisionTreeClassifier)
   - 隨機森林 (RandomForestClassifier, n_estimators=200)
   - GBDT (GradientBoostingClassifier, n_estimators=200, learning_rate=0.1)

   使用 5-fold 交叉驗證，報告每個模型的平均準確率與標準差。

3. **學習曲線 (Learning Curve)：** 使用 `sklearn.model_selection.learning_curve()`，為上述三個模型繪製學習曲線圖（X 軸為訓練集大小，Y 軸為分數），分析各模型的偏差-變異特性。

4. **分析與回答：**
   - 哪個模型的效能最好？差距大嗎？
   - 從學習曲線中，哪個模型偏差較高？哪個變異較高？
   - 如果你只能選一個模型部署到生產環境，你會選哪個？為什麼？

## 作業五：反思與總結（10%）

撰寫 200-300 字的反思，包含：
1. 本週最大的收穫是什麼？
2. Bagging 和 Boosting 的核心差異，你能用自己的話解釋嗎？
3. 在什麼情境下你會優先選擇 Random Forest 而非 GBDT？
4. 你在實驗中遇到的困難與解決方式

---

## 加分題 Bonus（+10%）

使用 XGBoost 或 LightGBM（需自行安裝）重複作業四的模型比較實驗，與 sklearn 的 `GradientBoostingClassifier` 比較：
- 準確率差異
- 訓練時間差異
- 提供你的觀察與分析

```python
# 安裝指令
# pip install xgboost lightgbm
```

---

## 評分標準 Grading Criteria

| 項目 | 比重 | 說明 |
|------|------|------|
| 程式碼正確執行 | 40% | 程式碼無錯誤且能完整執行 |
| 圖表品質 | 20% | 軸標籤、標題、圖例完整，易於閱讀 |
| 分析與回答 | 30% | 回答有深度、有邏輯支撐 |
| 程式碼風格 | 10% | 註解清楚、結構整齊 |

---

## 提示 Hints

- 使用 `train_test_split(test_size=0.2, random_state=42, stratify=y)` 確保可重現性
- 繪圖時設定 `plt.rcParams['font.sans-serif']` 以支援中文
- GBDT 的超參數網格搜尋可能需要一些時間，建議先用小規模測試
- 善用 `sklearn.model_selection.cross_val_score` 進行交叉驗證
- 如有問題，請先查閱 scikit-learn 官方文件或詢問 AI 助教
