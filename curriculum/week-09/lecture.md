# 第 9 週：特徵工程與資料前處理管線
# Week 9: Feature Engineering & Preprocessing Pipeline

## 學習目標 Learning Objectives
1. 理解特徵工程 (Feature Engineering) 的意義、流程與最佳實務
2. 掌握數值特徵的標準化與正規化方法 (StandardScaler, MinMaxScaler, RobustScaler)
3. 熟悉類別特徵編碼策略 (One-Hot Encoding, Label Encoding, Target Encoding)
4. 了解缺失值處理的多種策略與適用情境
5. 掌握特徵選擇 (Feature Selection) 三大方法：Filter、Wrapper、Embedded
6. 能使用 sklearn Pipeline + ColumnTransformer 建構可重現的前處理流程
7. 認識特徵交互 (Feature Interaction) 與多項式特徵
8. 初步了解 PCA 維度縮減及其視覺化
9. 透過視覺化比較前處理策略對模型效能的影響

---

## 1. 特徵工程的意義與流程 Significance & Workflow of Feature Engineering

### 1.1 什麼是特徵工程？ What is Feature Engineering?

特徵工程是將原始資料 (Raw Data) 轉換為更適合機器學習模型使用的特徵 (Features) 的過程。它是資料科學工作流程中**最具影響力**的環節之一。

> "Applied machine learning is basically feature engineering." — Andrew Ng

好的特徵工程可以：
- 提高模型預測能力 (Predictive Power)
- 降低模型複雜度 (Model Complexity)
- 改善模型的可解釋性 (Interpretability)
- 減少過擬合 (Overfitting) 風險

### 1.2 特徵工程的完整流程 Feature Engineering Pipeline

```
原始資料 Raw Data
  │
  ├─ 1. 資料理解 Data Understanding
  │     └── EDA、領域知識 Domain Knowledge
  │
  ├─ 2. 缺失值處理 Missing Value Handling
  │     └── 刪除 / 填補 Imputation
  │
  ├─ 3. 類別特徵編碼 Categorical Encoding
  │     └── One-Hot / Label / Target Encoding
  │
  ├─ 4. 數值特徵縮放 Numerical Scaling
  │     └── StandardScaler / MinMaxScaler / RobustScaler
  │
  ├─ 5. 特徵建構 Feature Construction
  │     └── 交互特徵 / 多項式特徵 / 領域特徵
  │
  ├─ 6. 特徵選擇 Feature Selection
  │     └── Filter / Wrapper / Embedded
  │
  ├─ 7. 維度縮減 Dimensionality Reduction (optional)
  │     └── PCA / t-SNE / UMAP
  │
  └─ 8. Pipeline 整合 Pipeline Integration
        └── ColumnTransformer + Pipeline
```

### 1.3 資料洩漏警告 Data Leakage Warning

在進行特徵工程時，最容易犯的錯誤就是**資料洩漏 (Data Leakage)**：

- **錯誤做法：** 先在全部資料上做 fit_transform，再分割訓練/測試集
- **正確做法：** 先分割資料，再在訓練集上 fit，用 transform 套用到測試集

```python
# 錯誤 WRONG
scaler.fit_transform(X)          # 全部資料
X_train, X_test = train_test_split(X)

# 正確 CORRECT
X_train, X_test = train_test_split(X)
scaler.fit(X_train)              # 只在訓練集 fit
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

使用 Pipeline 可以自動避免資料洩漏，這是本週的重點之一。

---

## 2. 數值特徵處理 Numerical Feature Scaling

不同的縮放方法適用於不同的資料分布與模型需求。

### 2.1 StandardScaler（標準化 Standardization）

將特徵轉換為均值 (Mean) 為 0、標準差 (Standard Deviation) 為 1 的分布。

$$z = \frac{x - \mu}{\sigma}$$

**適用情境：**
- 特徵近似常態分布 (Normal Distribution)
- 需要均值為 0 的演算法：SVM、Logistic Regression、Neural Networks
- PCA 等需要共變異數 (Covariance) 資訊的方法

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 2.2 MinMaxScaler（最小-最大正規化 Min-Max Normalization）

將特徵縮放到指定範圍（預設 [0, 1]）。

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**適用情境：**
- 不假設特定分布
- 需要特徵值在固定範圍的演算法：KNN、Neural Networks（某些情況）
- 影像像素值 (Pixel Values) 的處理

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)
```

**注意：** 對離群值 (Outliers) 非常敏感。一個極端值可能壓縮其他所有值。

### 2.3 RobustScaler（穩健縮放）

使用中位數 (Median) 與四分位距 (Interquartile Range, IQR) 進行縮放，對離群值更穩健。

$$x_{robust} = \frac{x - Q_{50}}{Q_{75} - Q_{25}}$$

**適用情境：**
- 資料包含離群值
- 不想被極端值影響的場景

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 2.4 三種 Scaler 比較摘要

| 方法 | 中心化依據 | 縮放依據 | 離群值敏感度 | 結果範圍 |
|------|-----------|---------|:---:|------|
| StandardScaler | 均值 Mean | 標準差 Std | 高 | 無固定範圍 |
| MinMaxScaler | 最小值 Min | 最大-最小 Range | 極高 | [0, 1] |
| RobustScaler | 中位數 Median | IQR | 低 | 無固定範圍 |

### 2.5 何時需要特徵縮放？ When is Scaling Needed?

| 模型 | 需要縮放？ | 原因 |
|------|:---:|------|
| 線性回歸 Linear Regression | 看情況 | 正則化時需要 |
| 邏輯迴歸 Logistic Regression | 是 | 梯度下降收斂速度 |
| SVM | 是 | 距離計算敏感 |
| KNN | 是 | 距離計算敏感 |
| 決策樹 Decision Tree | 否 | 基於分割點，不受尺度影響 |
| 隨機森林 Random Forest | 否 | 同上 |
| XGBoost / GBDT | 否 | 同上 |
| 神經網路 Neural Networks | 是 | 梯度更新穩定性 |

---

## 3. 類別特徵編碼 Categorical Feature Encoding

機器學習模型通常需要數值輸入，因此類別特徵需要進行編碼轉換。

### 3.1 Label Encoding（標籤編碼）

將每個類別映射為一個整數。

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# ['紅', '藍', '綠'] → [2, 0, 1]
encoded = le.fit_transform(['紅', '藍', '綠', '紅', '藍'])
```

**優點：** 簡單、不增加維度
**缺點：** 引入不存在的大小順序 (Ordinal Relationship)
**適用情境：**
- 有序類別 (Ordinal Features)，如：「低 < 中 < 高」
- 樹模型（可處理無序整數）

### 3.2 One-Hot Encoding（獨熱編碼）

為每個類別建立一個二元 (Binary) 特徵欄位。

```
顏色      →   顏色_紅  顏色_藍  顏色_綠
紅              1       0       0
藍              0       1       0
綠              0       0       1
```

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' 避免多重共線性
encoded = ohe.fit_transform(X_categorical)
```

**優點：** 不引入順序關係，大多數模型適用
**缺點：** 高基數特徵 (High Cardinality) 會導致維度爆炸 (Curse of Dimensionality)
**適用情境：**
- 名義類別 (Nominal Features)，如：顏色、國家
- 類別數量不多時（一般建議 < 15-20 個類別）

### 3.3 Target Encoding（目標編碼）

用目標變數 (Target Variable) 的統計量（如均值）取代類別值。

```
城市    房價均值 (Target Mean)
台北    1200 萬
台中    600 萬
高雄    450 萬
```

```python
# 使用 category_encoders 套件
import category_encoders as ce
te = ce.TargetEncoder(cols=['city'])
X_encoded = te.fit_transform(X_train, y_train)
```

**優點：** 不增加維度，可處理高基數特徵
**缺點：** 容易導致資料洩漏與過擬合
**解決方式：**
- 使用 K-Fold Target Encoding（在交叉驗證中計算）
- 加入平滑項 (Smoothing) 避免小樣本類別的極端值

### 3.4 編碼方法比較

| 方法 | 維度增加 | 順序假設 | 過擬合風險 | 高基數適用 |
|------|:---:|:---:|:---:|:---:|
| Label Encoding | 無 | 有 | 低 | 是 |
| One-Hot Encoding | 大 | 無 | 低 | 否 |
| Target Encoding | 無 | 無 | 高 | 是 |

---

## 4. 缺失值處理策略 Missing Value Handling

### 4.1 缺失值的類型 Types of Missing Data

| 類型 | 英文 | 說明 | 範例 |
|------|------|------|------|
| 完全隨機缺失 | MCAR (Missing Completely At Random) | 缺失與任何變數無關 | 資料傳輸錯誤 |
| 隨機缺失 | MAR (Missing At Random) | 缺失與其他觀察到的變數有關 | 年輕人較少填寫收入 |
| 非隨機缺失 | MNAR (Missing Not At Random) | 缺失與該變數本身有關 | 高收入者不願揭露收入 |

### 4.2 刪除策略 Deletion Strategies

#### 整列刪除 Listwise Deletion
```python
df_clean = df.dropna()  # 刪除任何含缺失值的列
```
- **適用：** 缺失比例很低（< 5%），且為 MCAR
- **風險：** 大量刪除可能導致樣本偏差 (Sample Bias)

#### 整欄刪除 Column Deletion
```python
df_clean = df.drop(columns=['col_with_many_na'])
```
- **適用：** 某欄位缺失比例過高（> 50-70%）

### 4.3 填補策略 Imputation Strategies

#### 簡單填補 Simple Imputation

```python
from sklearn.impute import SimpleImputer

# 均值填補 Mean Imputation（適用數值特徵）
imputer_mean = SimpleImputer(strategy='mean')

# 中位數填補 Median Imputation（適用有離群值的數值特徵）
imputer_median = SimpleImputer(strategy='median')

# 眾數填補 Mode Imputation（適用類別特徵）
imputer_mode = SimpleImputer(strategy='most_frequent')

# 常數填補 Constant Imputation
imputer_const = SimpleImputer(strategy='constant', fill_value=0)
```

| 策略 | 優點 | 缺點 | 適用場景 |
|------|------|------|---------|
| 均值 Mean | 簡單快速 | 受離群值影響，縮小變異 | 對稱分布的數值特徵 |
| 中位數 Median | 對離群值穩健 | 縮小變異 | 偏態分布的數值特徵 |
| 眾數 Mode | 適用類別 | 可能過度集中 | 類別特徵 |
| 常數 Constant | 保留缺失資訊 | 可能引入偏差 | 特定業務需求 |

#### KNN 填補 KNN Imputation

利用 K 個最近鄰居的值進行加權填補，考慮了特徵間的相關性。

```python
from sklearn.impute import KNNImputer
imputer_knn = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = imputer_knn.fit_transform(X)
```

**優點：** 考慮特徵間關係，填補值更合理
**缺點：** 計算成本高，需先處理類別特徵

#### 進階填補方法

- **迭代填補 Iterative Imputation (MICE)：** 多次迭代，以其他特徵預測缺失值
  ```python
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  imputer_iter = IterativeImputer(max_iter=10, random_state=42)
  ```

- **缺失值指示器 Missing Indicator：** 新增二元欄位標記原始缺失位置
  ```python
  from sklearn.impute import MissingIndicator
  indicator = MissingIndicator()
  missing_flags = indicator.fit_transform(X)
  ```

---

## 5. 特徵選擇 Feature Selection

特徵選擇的目的是從眾多特徵中挑選出最具資訊量的子集，以提高模型效能並降低計算成本。

### 5.1 Filter Methods（過濾法）

**原理：** 依據統計量對每個特徵進行獨立評分，不涉及模型訓練。

#### 常用方法：

1. **變異數閾值 Variance Threshold：** 移除低變異的特徵
   ```python
   from sklearn.feature_selection import VarianceThreshold
   selector = VarianceThreshold(threshold=0.01)
   X_selected = selector.fit_transform(X)
   ```

2. **相關係數 Correlation：** 移除與目標相關性低的特徵，或彼此高度相關的特徵
   ```python
   import pandas as pd
   corr_matrix = df.corr().abs()
   # 移除相關係數 > 0.95 的冗餘特徵
   upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
   to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
   ```

3. **卡方檢定 Chi-Square Test：** 適用類別特徵與類別目標
   ```python
   from sklearn.feature_selection import chi2, SelectKBest
   selector = SelectKBest(chi2, k=10)
   X_selected = selector.fit_transform(X, y)
   ```

4. **互資訊 Mutual Information：** 可捕捉非線性關係
   ```python
   from sklearn.feature_selection import mutual_info_classif
   mi_scores = mutual_info_classif(X, y)
   ```

**優點：** 速度快、不依賴特定模型
**缺點：** 忽略特徵間的交互作用

### 5.2 Wrapper Methods（包裝法）

**原理：** 使用特定模型的效能作為評估準則，透過搜尋策略選擇特徵子集。

#### 常用方法：

1. **遞迴特徵消除 Recursive Feature Elimination (RFE)：**
   ```python
   from sklearn.feature_selection import RFE
   from sklearn.ensemble import RandomForestClassifier

   estimator = RandomForestClassifier(n_estimators=100, random_state=42)
   rfe = RFE(estimator, n_features_to_select=10, step=1)
   X_selected = rfe.fit_transform(X, y)
   print("Selected features:", rfe.support_)
   print("Feature ranking:", rfe.ranking_)
   ```

2. **帶交叉驗證的 RFE：**
   ```python
   from sklearn.feature_selection import RFECV
   rfecv = RFECV(estimator, step=1, cv=5, scoring='accuracy')
   rfecv.fit(X, y)
   print(f"最佳特徵數: {rfecv.n_features_}")
   ```

**優點：** 考慮特徵交互作用，通常效果較好
**缺點：** 計算成本高（需多次訓練模型）

### 5.3 Embedded Methods（嵌入法）

**原理：** 在模型訓練過程中自動進行特徵選擇。

#### 常用方法：

1. **L1 正則化 (Lasso)：** 自動將不重要特徵的係數壓縮為 0
   ```python
   from sklearn.linear_model import LassoCV
   lasso = LassoCV(cv=5, random_state=42)
   lasso.fit(X_train, y_train)
   important = np.where(lasso.coef_ != 0)[0]
   ```

2. **基於樹模型的重要度 Tree-based Feature Importance：**
   ```python
   from sklearn.feature_selection import SelectFromModel
   from sklearn.ensemble import GradientBoostingClassifier

   gbdt = GradientBoostingClassifier(n_estimators=100, random_state=42)
   selector = SelectFromModel(gbdt, threshold='median')
   X_selected = selector.fit_transform(X, y)
   ```

**優點：** 計算效率適中，內建於訓練流程
**缺點：** 與特定模型綁定

### 5.4 三大方法比較

| 方法 | 速度 | 考慮交互作用 | 模型依賴 | 適用場景 |
|------|:---:|:---:|:---:|------|
| Filter | 快 | 否 | 否 | 初步篩選、大量特徵 |
| Wrapper | 慢 | 是 | 是 | 精細選擇、特徵不多 |
| Embedded | 中 | 部分 | 是 | 平衡效率與效果 |

---

## 6. sklearn Pipeline 建構 Building sklearn Pipelines

### 6.1 為什麼需要 Pipeline？ Why Pipeline?

1. **避免資料洩漏：** Pipeline 確保 fit 只在訓練集進行
2. **可重現性 (Reproducibility)：** 將所有步驟打包為單一物件
3. **簡化部署：** 一個物件即可進行預測
4. **方便調參：** 與 GridSearchCV / RandomizedSearchCV 無縫整合

### 6.2 基礎 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# fit + predict 一氣呵成
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score = pipe.score(X_test, y_test)
```

### 6.3 ColumnTransformer：分欄位處理

實務中，數值特徵和類別特徵需要不同的處理方式，ColumnTransformer 可以做到：

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 定義數值欄位的處理流程
numeric_features = ['age', 'income', 'credit_score']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 定義類別欄位的處理流程
categorical_features = ['gender', 'education', 'city']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 組合成 ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # 或 'passthrough' 保留其餘欄位
)
```

### 6.4 完整 Pipeline：前處理 + 模型

```python
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 一行完成所有前處理與訓練
full_pipeline.fit(X_train, y_train)

# 一行完成所有前處理與預測
accuracy = full_pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### 6.5 Pipeline + GridSearchCV

Pipeline 的參數名稱遵循 `步驟名__參數名` 的格式：

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"最佳參數: {grid_search.best_params_}")
print(f"最佳分數: {grid_search.best_score_:.4f}")
```

---

## 7. 可重現的前處理流程設計 Reproducible Preprocessing Design

### 7.1 設計原則 Design Principles

1. **版本控制：** 將 Pipeline 定義納入版本控制 (Git)
2. **隨機種子：** 所有隨機操作使用固定 `random_state`
3. **序列化儲存：** 使用 `joblib` 儲存訓練好的 Pipeline
4. **文件記錄：** 記錄每一步的參數設定與理由

### 7.2 Pipeline 序列化 Serialization

```python
import joblib

# 儲存
joblib.dump(full_pipeline, 'preprocessing_pipeline.joblib')

# 載入
loaded_pipeline = joblib.load('preprocessing_pipeline.joblib')
y_pred = loaded_pipeline.predict(X_new)
```

### 7.3 自定義轉換器 Custom Transformers

有時需要自定義的前處理步驟，可繼承 `BaseEstimator` 和 `TransformerMixin`：

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    """將離群值截斷到指定百分位範圍"""
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        self.lower_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_, self.upper_)
        return X_clipped
```

在 Pipeline 中使用自定義轉換器：

```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', OutlierClipper(lower_percentile=1, upper_percentile=99)),
    ('scaler', StandardScaler())
])
```

---

## 8. 特徵交互與多項式特徵 Feature Interaction & Polynomial Features

### 8.1 特徵交互 Feature Interaction

有時候，單個特徵本身資訊量有限，但特徵之間的**交互作用**可能蘊含重要資訊。

例如，預測房價時：
- `面積` 和 `樓層` 各自有一定預測力
- `面積 x 樓層` 的交互特徵可能更有意義（高樓層的大面積住宅有更高的溢價）

### 8.2 多項式特徵 Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"原始特徵數: {X.shape[1]}")
print(f"多項式特徵數: {X_poly.shape[1]}")
print("特徵名稱:", poly.get_feature_names_out())
```

若原始特徵為 `[a, b]`，`degree=2` 生成：
- `interaction_only=False`: `[a, b, a^2, ab, b^2]`
- `interaction_only=True`: `[a, b, ab]`

**注意事項：**
- 特徵數量會急劇增加（`degree=2` 約 $O(n^2)$，`degree=3` 約 $O(n^3)$）
- 高次多項式容易過擬合，建議搭配正則化 (Regularization)
- 在 Pipeline 中放在 Scaler 之後，以避免數值溢出

---

## 9. 維度縮減簡介：PCA 視覺化 Dimensionality Reduction: PCA Visualization

### 9.1 PCA 基本概念 Principal Component Analysis

主成分分析 (PCA) 尋找資料中**變異最大**的方向（主成分 Principal Components），將高維資料投影到低維空間。

**核心步驟：**
1. 資料標準化 (Standardization)
2. 計算共變異數矩陣 (Covariance Matrix)
3. 特徵值分解 (Eigendecomposition)
4. 選擇前 k 個主成分
5. 投影到低維空間

```python
from sklearn.decomposition import PCA

# 降到 2 維用於視覺化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"第一主成分解釋變異比例: {pca.explained_variance_ratio_[0]:.4f}")
print(f"第二主成分解釋變異比例: {pca.explained_variance_ratio_[1]:.4f}")
print(f"累積解釋變異: {sum(pca.explained_variance_ratio_):.4f}")
```

### 9.2 PCA 的用途

| 用途 | 說明 |
|------|------|
| 視覺化 Visualization | 將高維資料投影到 2D/3D 進行觀察 |
| 降維 Dimensionality Reduction | 減少特徵數量，加速模型訓練 |
| 去雜訊 Denoising | 移除低變異的成分 |
| 多重共線性 Multicollinearity | 消除特徵間的高度相關 |

### 9.3 選擇主成分數量

通常使用**累積解釋變異比例 (Cumulative Explained Variance Ratio)** 的陡坡圖 (Scree Plot) 來決定：

```python
pca_full = PCA()
pca_full.fit(X_scaled)

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
# 找到累積解釋變異 >= 95% 的最小主成分數
n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
```

### 9.4 PCA 注意事項

- PCA 是**線性**方法，無法捕捉非線性結構
- 必須先標準化，否則高變異特徵會主導結果
- 主成分不具可解釋性（是原始特徵的線性組合）
- 非線性替代方案：t-SNE（適合視覺化）、UMAP（兼顧全域與局部結構）

---

## 10. 前處理對模型效能影響的視覺化比較 Visual Comparison of Preprocessing Impact

### 10.1 實驗設計

為了量化前處理的影響，可以設計對照實驗：

```python
from sklearn.model_selection import cross_val_score

configs = {
    'No Scaling': Pipeline([('clf', SVC())]),
    'StandardScaler': Pipeline([('scaler', StandardScaler()), ('clf', SVC())]),
    'MinMaxScaler': Pipeline([('scaler', MinMaxScaler()), ('clf', SVC())]),
    'RobustScaler': Pipeline([('scaler', RobustScaler()), ('clf', SVC())]),
}

results = {}
for name, pipe in configs.items():
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 10.2 視覺化方法

**箱型圖比較 Box Plot Comparison：** 使用箱型圖展示不同前處理策略下的交叉驗證分數分布。

**雷達圖比較 Radar Chart：** 多維度比較不同前處理組合的效能指標（準確率、精確率、召回率、F1、訓練時間等）。

### 10.3 實務建議

1. **先建立 Baseline：** 不做任何前處理的基準效能
2. **逐步添加：** 一次只改一個前處理步驟，觀察效能變化
3. **使用交叉驗證：** 避免單次分割的隨機性
4. **記錄每次實驗：** Pipeline + 參數 + 結果，方便比較與回溯

---

## 關鍵詞彙 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 特徵工程 | Feature Engineering | 將原始資料轉換為適合模型的特徵 |
| 標準化 | Standardization | 轉換為均值 0、標準差 1 |
| 正規化 | Normalization | 縮放到指定範圍（如 [0,1]） |
| 獨熱編碼 | One-Hot Encoding | 類別轉為二元欄位 |
| 標籤編碼 | Label Encoding | 類別轉為整數 |
| 目標編碼 | Target Encoding | 以目標變數統計量取代類別 |
| 缺失值填補 | Imputation | 以估計值填補缺失資料 |
| 特徵選擇 | Feature Selection | 選擇最具資訊量的特徵子集 |
| 過濾法 | Filter Method | 以統計量評分的特徵選擇 |
| 包裝法 | Wrapper Method | 以模型效能評分的特徵選擇 |
| 嵌入法 | Embedded Method | 模型訓練中自動選擇特徵 |
| 資料管線 | Pipeline | 串聯多個處理步驟的物件 |
| 欄位轉換器 | ColumnTransformer | 對不同欄位套用不同轉換 |
| 主成分分析 | PCA | 線性降維方法 |
| 多項式特徵 | Polynomial Features | 建構特徵的高次項與交互項 |
| 資料洩漏 | Data Leakage | 訓練中使用了未來或測試資訊 |
| 維度詛咒 | Curse of Dimensionality | 高維空間中資料稀疏的問題 |

---

## 延伸閱讀 Further Reading
- scikit-learn Pipeline 官方教學：https://scikit-learn.org/stable/modules/compose.html
- scikit-learn Preprocessing：https://scikit-learn.org/stable/modules/preprocessing.html
- Feature Engineering and Selection (Max Kuhn)：https://bookdown.org/max/FES/
- Kaggle Feature Engineering 微課程：https://www.kaggle.com/learn/feature-engineering
- Category Encoders 文件：https://contrib.scikit-learn.org/category_encoders/
