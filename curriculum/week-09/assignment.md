# 第 9 週作業：特徵工程與資料前處理管線
# Week 9 Assignment: Feature Engineering & Preprocessing Pipeline

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 至課程平台

---

## 作業一：Scaler 比較實驗（20%）

使用 scikit-learn 的 `make_classification` 或真實資料集（如 Wine 資料集），完成以下任務：

1. 在原始資料中人為加入離群值 (Outliers)（例如：隨機選取 5% 的樣本，將其某特徵值乘以 10）
2. 分別使用 StandardScaler、MinMaxScaler、RobustScaler 進行縮放
3. 以散佈圖或分布圖 (Distribution Plot) 視覺化比較三種 Scaler 的效果
4. 使用 SVM 分類器，以 5 折交叉驗證 (5-Fold Cross-Validation) 比較不同 Scaler 下的分類準確率
5. 撰寫 1-2 段文字分析：為什麼 RobustScaler 在含離群值的資料中表現較好？

**提示 Hints:**
```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
```

---

## 作業二：缺失值處理與編碼策略（25%）

使用 Titanic 資料集（可從 Kaggle 下載或使用 `seaborn.load_dataset('titanic')`），完成以下任務：

1. **資料探索：**
   - 列出所有包含缺失值的欄位及其缺失比例
   - 判斷缺失類型（MCAR / MAR / MNAR），並說明理由

2. **缺失值處理：**
   - 對數值欄位 `age` 分別使用均值、中位數、KNN 填補，比較填補後的分布
   - 對類別欄位 `embarked` 使用眾數填補
   - 以視覺化呈現填補前後的分布差異

3. **類別編碼：**
   - 對 `sex` 欄位使用 Label Encoding
   - 對 `embarked` 欄位分別使用 One-Hot Encoding 與 Label Encoding
   - 比較兩種編碼方式對 Logistic Regression 分類結果的影響

4. 撰寫結論：哪種缺失值處理 + 編碼組合最適合此資料集？為什麼？

---

## 作業三：完整 Pipeline 建構（30%）

以 Titanic 資料集（目標：預測是否存活 `survived`）為例，建構完整的 sklearn Pipeline：

1. **定義特徵分組：**
   - 數值特徵 (Numerical Features)：`age`, `fare`, `sibsp`, `parch`
   - 類別特徵 (Categorical Features)：`sex`, `embarked`, `pclass`, `class`

2. **建構前處理 Pipeline：**
   - 數值：SimpleImputer(strategy='median') → StandardScaler
   - 類別：SimpleImputer(strategy='most_frequent') → OneHotEncoder(handle_unknown='ignore')
   - 使用 ColumnTransformer 組合

3. **建構完整 Pipeline：**
   - 前處理 + 分類器（使用 RandomForestClassifier）

4. **使用 GridSearchCV 搜尋最佳參數：**
   ```python
   param_grid = {
       'preprocessor__num__imputer__strategy': ['mean', 'median'],
       'classifier__n_estimators': [50, 100, 200],
       'classifier__max_depth': [5, 10, 20, None]
   }
   ```

5. **輸出：**
   - 最佳參數組合與對應的交叉驗證分數
   - 測試集上的分類報告 (Classification Report)

6. 使用 `joblib` 將訓練好的 Pipeline 儲存為 `.joblib` 檔案

---

## 作業四：特徵選擇與 PCA 視覺化（25%）

使用 Breast Cancer Wisconsin 資料集（`sklearn.datasets.load_breast_cancer`），完成以下任務：

1. **特徵選擇實驗：**
   - 使用 Filter Method（SelectKBest + mutual_info_classif），選出前 10 個特徵
   - 使用 Embedded Method（SelectFromModel + RandomForestClassifier），選出重要特徵
   - 比較兩種方法選出的特徵集合，是否有重疊？

2. **PCA 視覺化：**
   - 先標準化資料，再進行 PCA
   - 繪製 Scree Plot（各主成分解釋變異比例的長條圖 + 累積曲線）
   - 繪製前 2 個主成分的 2D 散佈圖，依類別著色
   - 標註每個主成分中權重最大的前 3 個原始特徵

3. **比較實驗：**
   - 比較以下三種策略的分類效能（使用 SVM + 5-Fold CV）：
     - (a) 使用全部 30 個特徵
     - (b) 使用 Filter 選出的 10 個特徵
     - (c) 使用 PCA 降至 10 個主成分
   - 以箱型圖視覺化比較結果

4. 撰寫結論：在這個資料集上，特徵選擇和 PCA 哪個更有效？為什麼？

---

## 加分題 Bonus（+10%）

建構一個**自定義轉換器 (Custom Transformer)**，功能為：
- 自動偵測數值欄位中的離群值（使用 IQR 方法）
- 將離群值截斷 (Clip) 到 1.5 * IQR 範圍內
- 繼承 `BaseEstimator` 和 `TransformerMixin`
- 可無縫放入 Pipeline 使用

展示此自定義轉換器在含離群值資料上的效果，並與不做離群值處理的版本比較模型效能。

---

## 評分標準 Grading Criteria
- 程式碼正確執行 Code Execution: 40%
- 視覺化品質 Visualization Quality: 20%
- Pipeline 設計合理性 Pipeline Design: 20%
- 分析與結論品質 Analysis & Conclusion: 20%

## 注意事項 Notes
- 所有隨機操作請使用固定隨機種子 `random_state=42`
- 圖表需包含標題、軸標籤與圖例
- 程式碼需附必要的註解說明
- 使用 Markdown Cell 撰寫分析文字
