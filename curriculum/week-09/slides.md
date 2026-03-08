# 第 9 週投影片：特徵工程與資料前處理管線

---

## Slide 1: 本週主題
# 特徵工程與資料前處理管線
### Feature Engineering & Preprocessing Pipeline
- 好的特徵 > 複雜的模型
- 從原始資料到可用特徵的完整流程
- 可重現、可部署的 Pipeline 設計

---

## Slide 2: 為什麼特徵工程重要？
### "Applied ML is basically feature engineering." — Andrew Ng

| 情境 | 無特徵工程 | 有特徵工程 |
|:---:|:---:|:---:|
| 準確率 | 72% | 89% |
| 訓練時間 | 長 | 短 |
| 可解釋性 | 低 | 高 |

好的特徵可以讓簡單模型達到接近複雜模型的效能。

---

## Slide 3: 特徵工程全景
### Feature Engineering Pipeline

```
原始資料 → 缺失值處理 → 編碼 → 縮放 → 特徵建構 → 選擇 → 降維 → 模型
```

本週依序學習每一個環節。

---

## Slide 4: 數值特徵縮放
### 三大 Scaler 比較

| Scaler | 公式 | 特點 |
|--------|------|------|
| **StandardScaler** | (x - mean) / std | 標準常態分布 |
| **MinMaxScaler** | (x - min) / (max - min) | 固定 [0, 1] 範圍 |
| **RobustScaler** | (x - median) / IQR | 對離群值穩健 |

**Demo:** 觀察三種 Scaler 在含離群值資料上的效果差異

---

## Slide 5: Scaler 效果視覺化
### [互動 Demo] 各 Scaler 分布對比

- 左圖：原始資料分布（含離群值）
- 右圖：三種 Scaler 後的分布
- 觀察重點：離群值如何影響 MinMaxScaler？

---

## Slide 6: 何時需要縮放？

| 需要 | 不需要 |
|:---:|:---:|
| SVM | 決策樹 |
| KNN | 隨機森林 |
| 邏輯迴歸 | XGBoost |
| 神經網路 | GBDT |
| PCA | - |

**規則：** 基於距離或梯度的演算法需要縮放，基於分割的樹模型不需要。

---

## Slide 7: 類別特徵編碼
### Label Encoding

```
紅 → 0,  藍 → 1,  綠 → 2
```
- 優點：簡單、不增維
- 缺點：引入假順序
- 適用：有序類別、樹模型

---

## Slide 8: 類別特徵編碼
### One-Hot Encoding

```
紅 → [1, 0, 0]
藍 → [0, 1, 0]
綠 → [0, 0, 1]
```
- 優點：無假順序
- 缺點：高基數時維度爆炸
- 適用：名義類別、類別數少

---

## Slide 9: 類別特徵編碼
### Target Encoding

```
台北 → 1200 (目標均值)
台中 → 600
高雄 → 450
```
- 優點：不增維、可處理高基數
- 缺點：資料洩漏風險
- 解決：K-Fold Target Encoding + Smoothing

---

## Slide 10: 編碼方法選擇指南
### 如何選擇編碼方式？

```
類別特徵
  ├── 有序？ → Label Encoding (OrdinalEncoder)
  └── 名義？
        ├── 基數 < 15？ → One-Hot Encoding
        └── 基數 >= 15？ → Target Encoding / Embedding
```

---

## Slide 11: 缺失值處理
### 常見策略

| 策略 | 方法 | 適用 |
|------|------|------|
| 刪除 | dropna() | 缺失 < 5%, MCAR |
| 均值 | SimpleImputer('mean') | 對稱數值 |
| 中位數 | SimpleImputer('median') | 偏態數值 |
| 眾數 | SimpleImputer('most_frequent') | 類別特徵 |
| KNN | KNNImputer | 有相關性的特徵 |
| 迭代 | IterativeImputer | 複雜缺失模式 |

---

## Slide 12: 缺失值處理決策樹

```
缺失比例？
  ├── > 50% → 考慮刪除該欄位
  ├── < 5% → 簡單刪除列或填補均值/中位數
  └── 5-50%
        ├── 數值 + 有離群值 → 中位數 / RobustScaler
        ├── 數值 + 無離群值 → 均值
        ├── 類別 → 眾數 / 新類別 'missing'
        └── 特徵相關性高 → KNN / MICE
```

---

## Slide 13: 特徵選擇三大方法
### Filter vs. Wrapper vs. Embedded

| | Filter | Wrapper | Embedded |
|---|:---:|:---:|:---:|
| 速度 | 快 | 慢 | 中 |
| 準確度 | 低 | 高 | 中-高 |
| 交互作用 | 否 | 是 | 部分 |
| 模型依賴 | 否 | 是 | 是 |

---

## Slide 14: Filter Methods
### 統計量評分

- **Variance Threshold:** 移除低變異特徵
- **Correlation:** 移除高度相關的冗餘特徵
- **Chi-Square / Mutual Information:** 評估特徵與目標的關聯

```python
SelectKBest(mutual_info_classif, k=10)
```

---

## Slide 15: Wrapper & Embedded Methods
### 搜尋式 vs. 內嵌式

**Wrapper (RFE):**
```python
RFE(estimator, n_features_to_select=10)
```
反覆訓練、移除最不重要的特徵

**Embedded (L1 / Tree Importance):**
```python
SelectFromModel(LassoCV())
SelectFromModel(RandomForestClassifier())
```
模型訓練中自動篩選

---

## Slide 16: sklearn Pipeline 概念
### 為什麼需要 Pipeline？

四大好處：
1. **防洩漏：** fit 只在訓練集
2. **可重現：** 所有步驟打包
3. **易部署：** 一個物件搞定
4. **好調參：** 整合 GridSearchCV

---

## Slide 17: ColumnTransformer
### 分欄位處理

```python
ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])
```

```
                 ┌─ impute → scale ─┐
原始資料 ─ split ┤                   ├─ 合併 → 模型
                 └─ impute → encode ─┘
```

---

## Slide 18: 完整 Pipeline 範例

```python
full_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])),
    ('classifier', RandomForestClassifier())
])

# 一行訓練
full_pipeline.fit(X_train, y_train)

# 一行預測
full_pipeline.predict(X_test)
```

---

## Slide 19: Pipeline + GridSearchCV
### 參數命名規則

```
步驟名__子步驟名__參數名
```

```python
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}
GridSearchCV(full_pipeline, param_grid, cv=5)
```

---

## Slide 20: 資料洩漏警告
### Data Leakage

```python
# 錯誤 - 在全部資料上 fit
scaler.fit_transform(X_all)
X_train, X_test = split(X_all)

# 正確 - 只在訓練集 fit
X_train, X_test = split(X_all)
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)

# 最佳 - 使用 Pipeline
Pipeline([('scaler', ...), ('model', ...)])
```

Pipeline 自動防止資料洩漏！

---

## Slide 21: 多項式特徵
### Feature Interaction & Polynomial Features

原始：[a, b]

| degree=2 | 產生的特徵 |
|----------|-----------|
| interaction_only=False | a, b, a^2, ab, b^2 |
| interaction_only=True | a, b, ab |

```python
PolynomialFeatures(degree=2, include_bias=False)
```

**注意：** 特徵數量會快速增長，搭配正則化使用

---

## Slide 22: PCA 視覺化
### 主成分分析

```
高維空間 ──PCA──→ 低維空間（2D 散佈圖）
```

- 找到資料變異最大的方向
- 投影到前 k 個主成分
- 用解釋變異比例選擇 k

**Demo:** Iris 資料集 PCA 2D 投影

---

## Slide 23: PCA Scree Plot
### 如何選擇主成分數？

- 畫累積解釋變異比例圖
- 找到「手肘」位置或 95% 門檻
- 通常 2-3 個主成分用於視覺化

---

## Slide 24: 前處理對模型的影響
### [互動 Demo] 前處理效果比較

```
SVM + No Scaling:      72%
SVM + StandardScaler:  91%
SVM + MinMaxScaler:    89%
SVM + RobustScaler:    90%
```

結論：對距離敏感的模型，前處理影響巨大！

---

## Slide 25: 自定義轉換器
### Custom Transformer

```python
class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lower_ = np.percentile(X, 1, axis=0)
        self.upper_ = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)
```

可無縫整合到 Pipeline 中！

---

## Slide 26: 本週重點回顧
### Week 9 Summary

1. **Scaler:** Standard > MinMax > Robust，依資料分布選擇
2. **Encoding:** One-Hot（名義）、Label（有序）、Target（高基數）
3. **缺失值:** 依比例、類型、相關性選策略
4. **特徵選擇:** Filter（快速）→ Embedded（平衡）→ Wrapper（精細）
5. **Pipeline:** ColumnTransformer + Pipeline = 可重現、防洩漏
6. **PCA:** 降維與視覺化的利器

---

## Slide 27: 本週作業
1. 對 Titanic 資料集建構完整 Pipeline
2. 比較不同前處理策略的效能
3. PCA 視覺化
4. 期中專題方向討論

---

## Slide 28: 下週預告
### Week 10: 超參數調校與學習曲線
- Grid Search vs. Random Search
- 學習曲線 Learning Curve
- 驗證曲線 Validation Curve
- 互動式超參數搜尋熱力圖
