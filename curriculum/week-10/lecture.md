# 第 10 週：超參數調校與學習曲線
# Week 10: Hyperparameter Tuning & Learning Curves

## 學習目標 Learning Objectives
1. 理解超參數 (Hyperparameters) 與模型參數 (Model Parameters) 的區別
2. 掌握 Grid Search（窮舉搜尋）與 Random Search（隨機搜尋）方法
3. 了解 Bayesian Optimization（貝葉斯最佳化）的基本概念
4. 透過學習曲線 (Learning Curve) 診斷過擬合 (Overfitting) 與欠擬合 (Underfitting)
5. 使用驗證曲線 (Validation Curve) 選擇最佳超參數
6. 認識 Optuna 等現代超參數搜尋框架

---

## 1. 超參數 vs 模型參數 Hyperparameters vs Model Parameters

### 1.1 定義與區別 Definition & Distinction

在機器學習中，有兩類截然不同的「參數」：

| 比較項目 | 模型參數 Model Parameters | 超參數 Hyperparameters |
|----------|:---:|:---:|
| 定義 | 模型從資料中自動學習的數值 | 訓練前由人工設定的配置值 |
| 學習方式 | 由訓練演算法最佳化 | 由人工或搜尋演算法決定 |
| 範例 | 線性回歸的權重 w、偏差 b | 學習率、正則化強度、樹的深度 |
| 儲存位置 | 模型內部 | 模型外部配置 |
| 數量 | 可能非常多（深度學習可達數十億） | 通常數個到數十個 |

> **類比：** 模型參數就像考試時學生填寫的答案，超參數則像老師設計的考試規則（考試時長、題目數量、配分方式）。學生透過學習得到答案，但考試規則是事先決定好的。

### 1.2 常見超參數範例 Common Hyperparameter Examples

```
模型 Model                    超參數 Hyperparameters
──────────────────────────────────────────────────────
線性回歸 Linear Regression     正則化強度 alpha (Ridge/Lasso)
邏輯迴歸 Logistic Regression   正則化強度 C, 懲罰類型 penalty
決策樹 Decision Tree           最大深度 max_depth, 最小分割樣本數 min_samples_split
隨機森林 Random Forest         樹的數量 n_estimators, 最大特徵數 max_features
SVM                           C, kernel, gamma
KNN                           鄰居數 n_neighbors, 距離度量 metric
梯度提升 Gradient Boosting     學習率 learning_rate, 樹的數量, 深度
神經網路 Neural Network        學習率, 層數, 神經元數, Dropout rate, Batch size
```

### 1.3 為什麼超參數調校重要？ Why Does Tuning Matter?

超參數的選擇直接影響模型的表現：
- **欠擬合 (Underfitting)：** 超參數設定使模型過於簡單，無法捕捉資料的規律
- **過擬合 (Overfitting)：** 超參數設定使模型過於複雜，記住了訓練資料的雜訊
- **最佳泛化 (Optimal Generalization)：** 找到恰當的超參數組合，在訓練效能與泛化能力之間取得平衡

```
模型複雜度 →
  低 ──────────── 適中 ──────────── 高
  欠擬合           最佳泛化           過擬合
  高偏差           低偏差+低變異       高變異
  High Bias        Sweet Spot         High Variance
```

---

## 2. Grid Search（窮舉搜尋）

### 2.1 原理 How It Works

Grid Search 是最直觀的超參數搜尋方法：對每個超參數定義一組候選值，然後**窮舉所有可能的組合**，逐一評估模型效能。

```
範例：搜尋 SVM 的 C 和 gamma

C     = [0.1, 1, 10, 100]
gamma = [0.001, 0.01, 0.1, 1]

搜尋空間 = 4 × 4 = 16 種組合

C=0.1, gamma=0.001  →  評估
C=0.1, gamma=0.01   →  評估
C=0.1, gamma=0.1    →  評估
...（共 16 次）
```

### 2.2 搭配交叉驗證 With Cross-Validation

在 scikit-learn 中，`GridSearchCV` 結合了 Grid Search 與 k-Fold Cross-Validation：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,                    # 5-fold 交叉驗證
    scoring='accuracy',       # 評估指標
    n_jobs=-1,               # 平行化
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"最佳參數: {grid_search.best_params_}")
print(f"最佳分數: {grid_search.best_score_:.4f}")
```

### 2.3 優缺點 Pros & Cons

| 優點 Pros | 缺點 Cons |
|-----------|-----------|
| 簡單直觀，容易實作 | 計算成本隨維度指數增長 (Curse of Dimensionality) |
| 保證找到搜尋空間中的最佳組合 | 搜尋點分布均勻，可能錯過最佳區域 |
| 容易平行化 (Parallelizable) | 不適合連續型超參數 |
| 結果可重現 (Reproducible) | 對不重要的超參數也花同樣的計算量 |

### 2.4 計算量分析 Computational Cost

```
總評估次數 = (候選值1的數量) × (候選值2的數量) × ... × (k-fold 數) × (每次訓練時間)

範例：3 個超參數，每個 5 個候選值，5-fold CV
= 5 × 5 × 5 × 5 = 625 次模型訓練
```

---

## 3. Random Search（隨機搜尋）及其優勢

### 3.1 原理 How It Works

Random Search 不窮舉所有組合，而是從超參數空間中**隨機採樣**固定次數的組合進行評估。

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'C': uniform(loc=0.01, scale=100),      # 連續均勻分布
    'gamma': uniform(loc=0.0001, scale=1),   # 連續均勻分布
    'kernel': ['rbf', 'poly']                # 離散選擇
}

random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_distributions,
    n_iter=50,               # 隨機嘗試 50 組
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
```

### 3.2 Random Search 的關鍵優勢 Key Advantages

#### 優勢一：高維搜尋更有效率

Bergstra & Bengio (2012) 的研究表明：**當只有少數超參數真正重要時，Random Search 比 Grid Search 更高效**。

```
Grid Search 的問題（9 次評估，2 個超參數）：

  超參數 B（不重要）
  ↑
  │ ● ● ●     ← 只在 3 個不同的 A 值上取樣
  │ ● ● ●     ← 在 B 上浪費了計算資源
  │ ● ● ●
  └──────→ 超參數 A（重要）

Random Search（9 次評估）：

  超參數 B（不重要）
  ↑
  │   ●   ●       ← 在 9 個不同的 A 值上取樣
  │ ●     ●       ← 更充分地探索了重要的維度
  │   ●  ● ●
  │  ●     ●
  └──────────→ 超參數 A（重要）
```

#### 優勢二：支援連續分布

Grid Search 只能在離散點上搜尋，Random Search 可以從**連續分布**中取樣：

| 分布類型 | 適用場景 | scipy.stats |
|----------|----------|-------------|
| 均勻分布 Uniform | 範圍已知但無偏好 | `uniform(loc, scale)` |
| 對數均勻分布 Log-Uniform | 跨多個數量級（如學習率） | `loguniform(a, b)` |
| 整數均勻分布 | 離散整數值 | `randint(low, high)` |
| 常態分布 Normal | 有先驗知識的參數 | `norm(loc, scale)` |

#### 優勢三：預算彈性

可以根據計算預算靈活調整 `n_iter`，不像 Grid Search 必須跑完整個網格。

### 3.3 Grid vs Random 的選擇指南 When to Use Which

| 情境 | 建議方法 |
|------|----------|
| 超參數 <= 2 個，候選值少 | Grid Search |
| 超參數 >= 3 個 | Random Search |
| 計算資源充裕 | Grid Search（保證完整搜尋） |
| 計算資源有限 | Random Search（更高效的探索） |
| 超參數為連續值 | Random Search |
| 初步探索（不知道好的範圍） | Random Search |
| 精細調校（已知好的範圍） | Grid Search |

---

## 4. Bayesian Optimization（貝葉斯最佳化）簡介

### 4.1 核心思想 Core Idea

Grid Search 和 Random Search 都是**無模型 (Model-free)** 的搜尋方法——每次評估都不參考之前的結果。Bayesian Optimization 則會**從過去的評估結果中學習**，建立一個「代理模型 (Surrogate Model)」來預測哪些超參數組合可能表現更好。

```
Bayesian Optimization 流程：

1. 初始化：隨機評估幾組超參數
2. 建立代理模型：用已有結果建立目標函數的近似模型
3. 選擇下一組：根據「採集函數 (Acquisition Function)」
   平衡探索 (Exploration) 與利用 (Exploitation)
4. 評估新組合：實際訓練模型並記錄結果
5. 更新代理模型：加入新的觀測值
6. 重複步驟 3-5，直到預算用完
```

### 4.2 關鍵元件 Key Components

| 元件 | 說明 | 常見方法 |
|------|------|----------|
| 代理模型 Surrogate Model | 近似目標函數的便宜替代 | 高斯過程 (Gaussian Process, GP)、TPE |
| 採集函數 Acquisition Function | 決定下一個取樣點 | EI (Expected Improvement)、UCB、PI |

### 4.3 與其他方法的比較 Comparison

```
搜尋效率（相同評估次數下的最佳分數）：

  分數
  ↑
  │     ┌──── Bayesian Optimization
  │   ┌─┘ ┌── Random Search
  │ ┌─┘ ┌─┘
  │─┘ ┌─┘ ┌── Grid Search
  │ ┌─┘ ┌─┘
  │─┘──┘
  └──────────→ 評估次數
```

Bayesian Optimization 通常在**評估次數較少**時就能找到好的解，適合：
- 單次訓練耗時長（如深度學習）
- 搜尋空間大且複雜
- 計算預算嚴格受限

---

## 5. 學習曲線 Learning Curve：診斷過擬合/欠擬合

### 5.1 什麼是學習曲線？ What is a Learning Curve?

學習曲線描繪了**隨著訓練資料量增加**，模型在訓練集和驗證集上的表現如何變化。它是診斷模型問題的強大工具。

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X, y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10% 到 100% 的訓練資料
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### 5.2 三種典型模式 Three Typical Patterns

#### 模式一：高偏差 — 欠擬合 High Bias — Underfitting

```
分數 Score
  ↑
1.0│
   │
   │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  訓練分數 (低)
   │
   │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  驗證分數 (低)
   │       ↑ 兩者都低且接近
0.0│
   └──────────────────────→ 訓練樣本數
```

**特徵：**
- 訓練分數和驗證分數都低
- 兩者之間的差距小
- 增加資料量也無法改善

**解決方案：**
- 增加模型複雜度（更深的樹、更多特徵、更複雜的模型）
- 增加更有用的特徵
- 減少正則化強度

#### 模式二：高變異 — 過擬合 High Variance — Overfitting

```
分數 Score
  ↑
1.0│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  訓練分數 (高)
   │                    ↑ 差距大
   │
   │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  驗證分數 (低)
   │       ↑ 兩者差距大
0.0│
   └──────────────────────→ 訓練樣本數
```

**特徵：**
- 訓練分數高，驗證分數低
- 兩者之間的差距大
- 增加資料量可能有幫助（驗證分數逐漸上升）

**解決方案：**
- 增加正則化 (Regularization)
- 簡化模型
- 增加訓練資料量
- 特徵選擇 (Feature Selection)
- 早停法 (Early Stopping)

#### 模式三：良好擬合 Good Fit

```
分數 Score
  ↑
1.0│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  訓練分數
   │                  ↑ 差距小
   │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  驗證分數 (高)
   │
   │
0.0│
   └──────────────────────→ 訓練樣本數
```

**特徵：**
- 訓練分數和驗證分數都高
- 兩者之間的差距小
- 隨著資料增加趨於穩定

### 5.3 學習曲線的實務應用 Practical Applications

1. **判斷是否需要更多資料：** 如果驗證分數仍在上升且尚未平坦，收集更多資料可能有幫助
2. **判斷模型複雜度是否適當：** 根據訓練/驗證分數的差距做調整
3. **比較不同模型：** 在相同資料量下比較不同模型的學習曲線

---

## 6. 驗證曲線 Validation Curve

### 6.1 什麼是驗證曲線？ What is a Validation Curve?

驗證曲線描繪了**隨著某個超參數值的變化**，模型在訓練集和驗證集上的表現如何變化。與學習曲線不同，驗證曲線的 x 軸是超參數值，而非資料量。

```python
from sklearn.model_selection import validation_curve

train_scores, val_scores = validation_curve(
    estimator=SVC(),
    X=X, y=y,
    param_name='gamma',
    param_range=np.logspace(-6, 1, 20),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### 6.2 解讀驗證曲線 Interpreting the Validation Curve

```
分數 Score
  ↑
1.0│         ┌───┐
   │        /│   │\  ── 訓練分數
   │       / │   │ \
   │      /  │   │  \ ── 驗證分數
   │     /   │最佳│
   │    /    │   │
0.5│   /     │   │
   │  /      │   │
   │ /       │   │
   └─────────────────→ 超參數值
     小 ←── 欠擬合 ──最佳── 過擬合 ──→ 大
```

**解讀：**
- **左側（超參數值小）：** 模型欠擬合，訓練/驗證分數都低
- **中間（最佳值）：** 驗證分數最高的位置
- **右側（超參數值大）：** 模型過擬合，訓練分數高但驗證分數下降

### 6.3 常見超參數的驗證曲線形態 Common Patterns

| 超參數 | 值增大時的效果 | 典型曲線形態 |
|--------|---------------|-------------|
| 決策樹 max_depth | 模型更複雜 | 驗證分數先升後降 |
| SVM C | 正則化減弱 | 驗證分數先升後降 |
| SVM gamma | 決策邊界更複雜 | 驗證分數先升後降 |
| KNN n_neighbors | 模型更平滑 | 驗證分數先升後降（反向） |
| Ridge alpha | 正則化增強 | 驗證分數先升後降 |

---

## 7. 超參數搜尋空間設計 Designing the Search Space

### 7.1 設計原則 Design Principles

好的搜尋空間設計是高效調校的關鍵：

#### 原則一：使用對數尺度 Use Log Scale

許多超參數（如學習率、正則化強度）的影響跨越多個數量級，應使用對數尺度搜尋：

```python
# 不好的做法 — 線性尺度
learning_rate = [0.001, 0.002, 0.003, ..., 0.1]  # 大部分搜尋集中在高值

# 好的做法 — 對數尺度
learning_rate = np.logspace(-4, -1, 20)  # 均勻覆蓋 0.0001 到 0.1
# 或使用 scipy
from scipy.stats import loguniform
learning_rate = loguniform(1e-4, 1e-1)
```

#### 原則二：先粗後細 Coarse-to-Fine

```
第一輪（粗搜尋）：
  學習率: [0.0001, 0.001, 0.01, 0.1]
  最佳 ≈ 0.01

第二輪（細搜尋）：
  學習率: [0.005, 0.008, 0.01, 0.02, 0.05]
  最佳 ≈ 0.008
```

#### 原則三：了解超參數的相互作用 Understand Interactions

某些超參數之間存在交互作用 (Interactions)：
- 學習率 (Learning Rate) 與批次大小 (Batch Size) 經常需要一起調整
- 正則化強度與模型複雜度互相影響
- 決策樹的 max_depth 與 min_samples_split 互相制約

### 7.2 搜尋空間範本 Search Space Templates

```python
# Random Forest 搜尋空間
rf_params = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Gradient Boosting 搜尋空間
gb_params = {
    'n_estimators': randint(50, 500),
    'learning_rate': loguniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4),        # 0.6 到 1.0
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# SVM 搜尋空間
svm_params = {
    'C': loguniform(0.01, 100),
    'gamma': loguniform(0.0001, 1),
    'kernel': ['rbf', 'poly'],
    'degree': randint(2, 5)                # 僅 poly kernel 使用
}
```

---

## 8. 計算資源考量與早停策略 Computational Considerations & Early Stopping

### 8.1 計算成本估算 Estimating Computational Cost

在開始搜尋前，估算總計算量是良好的習慣：

```
總時間 ≈ n_combinations × n_folds × time_per_fit

範例：
  Grid Search: 5 × 5 × 3 = 75 組合
  5-fold CV: 75 × 5 = 375 次訓練
  每次訓練 2 分鐘: 375 × 2 = 750 分鐘 ≈ 12.5 小時
```

### 8.2 減少計算量的策略 Strategies to Reduce Computation

| 策略 | 說明 | 節省幅度 |
|------|------|----------|
| 減少 CV folds | 5-fold → 3-fold | ~40% |
| 使用子集訓練 | 先用 10% 資料粗搜 | ~90% |
| Random Search | 限制 n_iter | 可控 |
| 早停法 Early Stopping | 提前結束差的組合 | 變動 |
| Successive Halving | 逐步淘汰差的候選 | ~50-80% |
| 平行化 Parallelization | n_jobs=-1 | 依核心數 |

### 8.3 早停法 Early Stopping

早停法的核心思想：**如果在訓練過程中，驗證效能已經開始下降，就提前停止訓練**。

```python
# scikit-learn 的 HalvingRandomSearchCV（逐步篩選）
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

halving_search = HalvingRandomSearchCV(
    estimator=model,
    param_distributions=param_dist,
    factor=3,               # 每輪淘汰 2/3 的候選
    resource='n_samples',   # 逐步增加資料量
    cv=5,
    random_state=42,
    n_jobs=-1
)
```

### 8.4 Successive Halving（逐步減半）流程

```
第 1 輪：81 組候選，每組用 100 筆資料訓練
  → 保留前 27 名

第 2 輪：27 組候選，每組用 300 筆資料訓練
  → 保留前 9 名

第 3 輪：9 組候選，每組用 900 筆資料訓練
  → 保留前 3 名

第 4 輪：3 組候選，每組用 2700 筆資料訓練
  → 選出最佳
```

---

## 9. 常見模型的關鍵超參數指南 Key Hyperparameters Guide

### 9.1 決策樹與集成模型 Tree-Based Models

| 超參數 | 影響 | 建議範圍 | 調校優先級 |
|--------|------|----------|:---:|
| `n_estimators` (RF/GBDT) | 樹的數量，越多通常越好但邊際效益遞減 | 100-1000 | 中 |
| `max_depth` | 單棵樹的深度，控制複雜度 | 3-20 | 高 |
| `learning_rate` (GBDT) | 每棵樹的貢獻權重 | 0.01-0.3 | 高 |
| `min_samples_split` | 節點分裂的最小樣本數 | 2-20 | 中 |
| `min_samples_leaf` | 葉節點的最小樣本數 | 1-10 | 中 |
| `max_features` | 每次分裂考慮的特徵數 | 'sqrt', 'log2' | 中 |
| `subsample` (GBDT) | 訓練每棵樹使用的資料比例 | 0.6-1.0 | 中 |

### 9.2 SVM

| 超參數 | 影響 | 建議範圍 | 調校優先級 |
|--------|------|----------|:---:|
| `C` | 正則化（越大懲罰越嚴格） | 0.01-100（對數尺度） | 高 |
| `gamma` (RBF) | 核函數寬度（越大邊界越複雜） | 0.0001-1（對數尺度） | 高 |
| `kernel` | 核函數類型 | 'rbf', 'poly', 'linear' | 高 |

### 9.3 神經網路 Neural Networks

| 超參數 | 影響 | 建議範圍 | 調校優先級 |
|--------|------|----------|:---:|
| Learning Rate | 梯度更新步長 | 1e-5 ~ 1e-1（對數尺度） | 最高 |
| Batch Size | 每次更新的樣本數 | 16, 32, 64, 128, 256 | 高 |
| 隱藏層數 Layers | 網路深度 | 依任務而定 | 高 |
| 隱藏層寬度 Units | 每層神經元數 | 32-512 | 中 |
| Dropout Rate | 隨機丟棄比例 | 0.1-0.5 | 中 |
| Weight Decay (L2) | 權重正則化 | 1e-5 ~ 1e-2 | 中 |
| Optimizer | 最佳化器 | Adam, SGD, AdamW | 高 |

### 9.4 調校的經驗法則 Rules of Thumb

1. **最重要的超參數先調：** 學習率 > 模型架構 > 正則化 > 其他
2. **使用已知的好預設值作為起點：** 例如 Adam optimizer 的 lr=0.001
3. **一次只改一個超參數**（驗證曲線），或使用搜尋演算法同時調多個
4. **注意超參數之間的交互作用**
5. **記錄每次實驗結果**（可用 MLflow 或 Weights & Biases）

---

## 10. Optuna 簡介 Introduction to Optuna

### 10.1 什麼是 Optuna？

Optuna 是一個現代化的超參數最佳化框架，由日本 Preferred Networks 開發。它使用 **TPE (Tree-structured Parzen Estimator)** 作為預設的 Bayesian Optimization 演算法。

### 10.2 Optuna 的核心優勢

| 優勢 | 說明 |
|------|------|
| **Define-by-Run** | 搜尋空間在程式碼中動態定義，支援條件式搜尋 |
| **高效搜尋** | 內建 TPE、CMA-ES 等先進演算法 |
| **剪枝 Pruning** | 自動停止不好的試驗 (Trial)，節省資源 |
| **視覺化** | 內建豐富的視覺化工具 |
| **分散式** | 支援多機器平行搜尋 |

### 10.3 基本用法 Basic Usage

```python
import optuna

def objective(trial):
    # 定義搜尋空間
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42
    )

    # 使用交叉驗證評估
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# 建立 Study 並最佳化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"最佳分數: {study.best_value:.4f}")
print(f"最佳參數: {study.best_params}")
```

### 10.4 條件式搜尋空間 Conditional Search Space

Optuna 的 Define-by-Run 架構允許建立**條件式搜尋空間**：

```python
def objective(trial):
    classifier_name = trial.suggest_categorical('classifier', ['SVM', 'RF', 'GBDT'])

    if classifier_name == 'SVM':
        C = trial.suggest_float('svm_C', 0.01, 100, log=True)
        kernel = trial.suggest_categorical('svm_kernel', ['rbf', 'poly'])
        model = SVC(C=C, kernel=kernel)

    elif classifier_name == 'RF':
        n_estimators = trial.suggest_int('rf_n_estimators', 50, 500)
        max_depth = trial.suggest_int('rf_max_depth', 3, 20)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    elif classifier_name == 'GBDT':
        lr = trial.suggest_float('gb_lr', 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int('gb_n_estimators', 50, 500)
        model = GradientBoostingClassifier(learning_rate=lr, n_estimators=n_estimators)

    scores = cross_val_score(model, X_train, y_train, cv=5)
    return scores.mean()
```

### 10.5 Optuna 內建視覺化 Built-in Visualization

```python
import optuna.visualization as vis

# 最佳化歷程
vis.plot_optimization_history(study)

# 超參數重要度
vis.plot_param_importances(study)

# 平行座標圖
vis.plot_parallel_coordinate(study)

# 等高線圖
vis.plot_contour(study, params=['learning_rate', 'max_depth'])
```

---

## 關鍵詞彙 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 超參數 | Hyperparameter | 訓練前由人工設定的配置值 |
| 模型參數 | Model Parameter | 模型從資料中學習的數值 |
| 窮舉搜尋 | Grid Search | 嘗試所有超參數組合 |
| 隨機搜尋 | Random Search | 從分布中隨機取樣超參數組合 |
| 貝葉斯最佳化 | Bayesian Optimization | 利用過去結果指導搜尋的方法 |
| 學習曲線 | Learning Curve | 訓練樣本數 vs 模型表現的曲線 |
| 驗證曲線 | Validation Curve | 超參數值 vs 模型表現的曲線 |
| 代理模型 | Surrogate Model | Bayesian Optimization 中近似目標函數的模型 |
| 採集函數 | Acquisition Function | 決定下一個探索點的策略函數 |
| 早停法 | Early Stopping | 在驗證效能不再改善時提前停止訓練 |
| 搜尋空間 | Search Space | 超參數候選值的範圍定義 |
| 交叉驗證 | Cross-Validation | 將資料分成多份輪流驗證的方法 |
| 逐步減半 | Successive Halving | 逐輪淘汰表現差的候選組合 |
| 過擬合 | Overfitting | 模型過度適應訓練資料 |
| 欠擬合 | Underfitting | 模型無法捕捉資料規律 |

---

## 延伸閱讀 Further Reading
- Bergstra & Bengio (2012), "Random Search for Hyper-Parameter Optimization" — Random Search 優勢的經典論文
- scikit-learn 超參數調校文件：https://scikit-learn.org/stable/modules/grid_search.html
- Optuna 官方文件：https://optuna.readthedocs.io/
- AutoML 概述：https://www.automl.org/
- Snoek et al. (2012), "Practical Bayesian Optimization of Machine Learning Algorithms"
