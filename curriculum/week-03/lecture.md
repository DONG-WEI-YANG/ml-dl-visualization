# 第 3 週：監督式學習概念、資料分割與交叉驗證
# Week 3: Supervised Learning, Data Splitting & Cross-Validation

## 學習目標 Learning Objectives
1. 理解監督式學習 (Supervised Learning) 的完整框架與工作流程
2. 掌握訓練集 (Training Set)、驗證集 (Validation Set) 與測試集 (Test Set) 的角色與分割方法
3. 了解 k 折交叉驗證 (k-Fold Cross-Validation) 的原理、變體與應用場景
4. 認識過擬合 (Overfitting) 與欠擬合 (Underfitting) 的診斷方法
5. 掌握偏差-變異權衡 (Bias-Variance Tradeoff) 的核心概念
6. 了解分層抽樣 (Stratified Sampling) 與時間序列分割 (Time Series Split) 等特殊策略

---

## 1. 監督式學習框架 Supervised Learning Framework

### 1.1 什麼是監督式學習？ What is Supervised Learning?

監督式學習是機器學習中最常見的範式 (Paradigm)。它的核心思想是：給定一組**已標註** (Labeled) 的訓練資料，讓模型學習從**輸入特徵 (Input Features, X)** 到**目標標籤 (Target Label, y)** 的映射函數 (Mapping Function)。

> **形式化定義**：給定資料集 $D = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$，目標是找到一個函數 $f: X \rightarrow Y$，使得 $f(x_i) \approx y_i$ 對所有樣本成立，並且能對**未見過的資料**做出良好預測。

### 1.2 監督式學習的工作流程 Workflow

```
輸入資料 (Input Data)
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ 特徵 X  │ ──▶ │ 模型 f  │ ──▶ │ 預測 ŷ  │ ──▶ │ 損失 L  │
│ Features│     │  Model  │     │Prediction│    │  Loss   │
└─────────┘     └────┬────┘     └─────────┘     └────┬────┘
                     ▲                                │
                     │         ┌─────────┐            │
                     └─────────│ 更新參數 │◀───────────┘
                               │ Update  │
                               │ Params  │
                               └─────────┘
```

**五個核心步驟：**

1. **輸入 (Input)**：將特徵向量 $x$ 送入模型
2. **前向傳播 (Forward Pass)**：模型根據當前參數 $\theta$ 計算預測值 $\hat{y} = f(x; \theta)$
3. **計算損失 (Compute Loss)**：用損失函數 $L(\hat{y}, y)$ 衡量預測值與真實值的差距
4. **反向傳播 (Backward Pass)**：計算損失對參數的梯度 $\nabla_\theta L$
5. **參數更新 (Parameter Update)**：利用最佳化演算法（如梯度下降 Gradient Descent）更新參數 $\theta \leftarrow \theta - \eta \nabla_\theta L$

### 1.3 監督式學習的兩大任務 Two Main Tasks

| 任務 Task | 輸出類型 Output | 損失函數 Loss | 範例 Example |
|-----------|----------------|--------------|-------------|
| 回歸 Regression | 連續值 Continuous | MSE, MAE | 房價預測、溫度預測 |
| 分類 Classification | 離散類別 Discrete | Cross-Entropy | 圖片辨識、垃圾郵件偵測 |

### 1.4 模型泛化能力 Generalization

監督式學習的終極目標不只是在訓練資料上表現良好，而是要具備**泛化能力 (Generalization)**——在**未見過的新資料**上也能做出準確預測。這就引出了為什麼我們需要「資料分割」的核心問題。

---

## 2. 資料分割 Data Splitting

### 2.1 為什麼需要分割資料？ Why Split Data?

想像一個學生用考古題 (Past Exams) 準備考試：

- 如果考試題目**完全一樣**，成績只能反映「背誦能力」，不能反映「真正理解」
- 如果考試出**新題目**，才能測試學生是否真正掌握了知識

同理，如果我們用**全部資料**訓練模型，再用**同一批資料**評估模型，得到的分數會過度樂觀 (Overly Optimistic)，無法反映模型的真實泛化能力。

> **核心原則：評估模型時，必須使用模型在訓練過程中從未見過的資料。**

### 2.2 訓練集 vs 測試集 Training Set vs Test Set

最基本的分割方式是將資料分為兩部分：

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="訓練測試驗證分割示意圖">
  <rect x="0" y="0" width="640" height="200" fill="#ffffff"/>
  <!-- TWO-WAY SPLIT -->
  <text x="320" y="26" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">二分法：Training / Test</text>
  <rect x="40" y="40" width="420" height="44" fill="#dbeafe" stroke="#1e40af" stroke-width="1.5"/>
  <rect x="460" y="40" width="140" height="44" fill="#fecaca" stroke="#991b1b" stroke-width="1.5"/>
  <text x="250" y="68" text-anchor="middle" font-size="13" fill="#1e3a8a" font-weight="600">訓練集 Training (75%)</text>
  <text x="530" y="68" text-anchor="middle" font-size="13" fill="#7f1d1d" font-weight="600">測試 Test (25%)</text>
  <!-- THREE-WAY SPLIT -->
  <text x="320" y="116" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">三分法：Training / Validation / Test</text>
  <rect x="40" y="130" width="340" height="44" fill="#dbeafe" stroke="#1e40af" stroke-width="1.5"/>
  <rect x="380" y="130" width="110" height="44" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <rect x="490" y="130" width="110" height="44" fill="#fecaca" stroke="#991b1b" stroke-width="1.5"/>
  <text x="210" y="158" text-anchor="middle" font-size="13" fill="#1e3a8a" font-weight="600">訓練 Training (60%)</text>
  <text x="435" y="158" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">驗證 Val (20%)</text>
  <text x="545" y="158" text-anchor="middle" font-size="12" fill="#7f1d1d" font-weight="600">測試 Test (20%)</text>
  <!-- Usage notes under each segment -->
  <text x="210" y="192" text-anchor="middle" font-size="10" fill="#6b7280">學習參數</text>
  <text x="435" y="192" text-anchor="middle" font-size="10" fill="#6b7280">選超參數</text>
  <text x="545" y="192" text-anchor="middle" font-size="10" fill="#6b7280">最終評估（僅一次）</text>
</svg>
<figcaption>示意圖：資料分割。二分法只切訓練與測試；三分法多出驗證集用於超參數選擇，測試集保留到最後一次使用，避免資訊洩漏。</figcaption>
</figure>
```

- **訓練集 (Training Set)**：模型從中學習規律（調整參數）
- **測試集 (Test Set)**：模型訓練完成後，用於評估最終表現的「考試卷」

**重要守則：** 測試集只能在模型完全訓練好之後使用**一次**。如果反覆用測試集調整模型，測試集就會變相成為訓練資料，導致效能評估不可靠。

### 2.3 驗證集的必要性 The Need for a Validation Set

在實務上，我們常需要**調整超參數 (Hyperparameters)**（如學習率、正則化強度、模型複雜度等）。如果用測試集來選擇超參數，就會造成「資訊洩漏 (Data Leakage)」——測試集的資訊間接影響了模型設計，破壞了測試集的獨立性。

解決方案：再劃分出一個**驗證集 (Validation Set)**。

```
┌──────────────────────────────────────────────────────────────────┐
│                       完整資料集 Full Dataset                      │
├──────────────────────────┬───────────────┬───────────────────────┤
│    訓練集 Training Set    │ 驗證集 Val Set │    測試集 Test Set      │
│       (60-70%)           │  (10-15%)     │     (15-20%)          │
│   學習模型參數             │ 超參數調整      │    最終效能評估          │
└──────────────────────────┴───────────────┴───────────────────────┘
```

**三者角色對照：**

| 資料集 | 英文 | 用途 Purpose | 使用頻率 Frequency |
|--------|------|-------------|-------------------|
| 訓練集 | Training Set | 訓練模型（擬合參數） | 每個 Epoch 都使用 |
| 驗證集 | Validation Set | 選擇超參數、監控過擬合 | 訓練過程中反覆使用 |
| 測試集 | Test Set | 最終效能評估 | 只使用一次 |

### 2.4 使用 `train_test_split` 函數

scikit-learn 提供了方便的 `train_test_split` 函數：

```python
from sklearn.model_selection import train_test_split

# 基本用法：80% 訓練，20% 測試
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 測試集比例
    random_state=42,      # 隨機種子（確保可重現性 Reproducibility）
    shuffle=True          # 是否洗牌（預設 True）
)

# 進一步分割出驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,        # 從剩餘 80% 中取 20% 作為驗證集
    random_state=42
)
```

**關鍵參數：**
- `test_size`：可以是浮點數（比例）或整數（樣本數）
- `random_state`：固定隨機種子以確保結果可重現
- `shuffle`：是否在分割前打亂資料順序
- `stratify`：分層抽樣的依據（下節說明）

---

## 3. 過擬合與欠擬合 Overfitting and Underfitting

### 3.1 定義 Definitions

**過擬合 (Overfitting)**：模型過度適應訓練資料，記住了資料中的雜訊 (Noise)，導致在新資料上表現不佳。

- 症狀：訓練誤差很低，但測試/驗證誤差很高
- 類比：學生死背考古題答案，遇到新題目就不會做

**欠擬合 (Underfitting)**：模型過於簡單，無法捕捉資料中的真實規律 (True Pattern)。

- 症狀：訓練誤差和測試誤差都很高
- 類比：學生完全沒讀書，連考古題都寫不好

```
誤差
Error
  │
  │  欠擬合                    過擬合
  │ Underfitting              Overfitting
  │    ╲                        ╱
  │     ╲    ── 測試誤差 ──    ╱
  │      ╲   Test Error      ╱
  │       ╲                 ╱
  │        ╲    最佳點     ╱
  │         ╲  Optimal   ╱
  │          ╲    ↓     ╱
  │           ╲       ╱
  │            ╲     ╱
  │             ╲   ╱
  │              ╲ ╱
  │     ── 訓練誤差 ──────────────▶
  │     Training Error
  └────────────────────────────────▶
                模型複雜度
              Model Complexity
```

### 3.2 診斷方法 Diagnosis

| 現象 Symptom | 訓練誤差 Train Error | 測試誤差 Test Error | 診斷 Diagnosis |
|-------------|---------------------|-------------------|----------------|
| 兩者都高 | 高 High | 高 High | 欠擬合 Underfitting |
| 訓練低、測試高 | 低 Low | 高 High | 過擬合 Overfitting |
| 兩者都低且接近 | 低 Low | 低 Low | 良好擬合 Good Fit |

### 3.3 解決策略 Remedies

**過擬合的解決方案：**
- 增加訓練資料量 (More Data)
- 降低模型複雜度 (Reduce Model Complexity)
- 正則化 (Regularization)：L1/L2、Dropout
- 早停 (Early Stopping)
- 交叉驗證 (Cross-Validation)

**欠擬合的解決方案：**
- 增加模型複雜度 (Increase Model Complexity)
- 增加特徵 (Feature Engineering)
- 減少正則化強度
- 訓練更長時間 (Train Longer)

---

## 4. 偏差-變異權衡 Bias-Variance Tradeoff

### 4.1 核心概念 Core Concepts

模型的**泛化誤差 (Generalization Error)** 可以分解為三個部分：

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| 成分 Component | 說明 Description | 直觀理解 Intuition |
|---------------|-----------------|-------------------|
| 偏差 Bias | 模型預測值的期望與真實值之間的差異 | 模型的「系統性偏移」—— 靶心偏移量 |
| 變異 Variance | 模型預測值隨訓練資料不同而波動的程度 | 模型的「不穩定性」—— 射擊散布程度 |
| 不可約誤差 Irreducible Noise | 資料本身的隨機性，無法透過模型改善 | 環境風力等不可控因素 |

### 4.2 射擊靶心類比 Bullseye Analogy

```
    高偏差 + 低變異          低偏差 + 低變異
   High Bias, Low Var     Low Bias, Low Var
   ┌──────────────┐       ┌──────────────┐
   │   ○          │       │              │
   │ ○  ○    ◎    │       │     ○◎○      │
   │  ○           │       │      ○       │
   │              │       │              │
   └──────────────┘       └──────────────┘
   一致但偏離中心            精準且集中
   Consistent but off      Accurate & precise

    高偏差 + 高變異          低偏差 + 高變異
   High Bias, High Var    Low Bias, High Var
   ┌──────────────┐       ┌──────────────┐
   │ ○         ○  │       │    ○         │
   │      ◎       │       │  ○   ◎   ○   │
   │   ○      ○   │       │        ○     │
   │        ○     │       │   ○          │
   └──────────────┘       └──────────────┘
   散亂且偏離中心            散亂但圍繞中心
   Scattered and off       Scattered around center
```

### 4.3 複雜度與權衡 Complexity and the Tradeoff

以**多項式回歸 (Polynomial Regression)** 為例：

- **Degree = 1**（直線）：高偏差、低變異 → 欠擬合
- **Degree = 4-5**（適當曲線）：平衡偏差與變異 → 最佳擬合
- **Degree = 15**（高次多項式）：低偏差、高變異 → 過擬合

```
degree=1                 degree=4               degree=15
  │     /                │    ╱╲                │ ╱╲  ╱╲╱╲
  │    /  ·  ·           │   ╱  ╲  ·            │╱  ╲╱    ╲
  │   / ·                │  ·╱  ·╲              │·    ·  · ╲
  │  / ·    ·            │  ╱  ·  ╲·            │          ╲·
  │ /·                   │ ╱       ╲            │
  │/ ·                   │╱    ·    ╲           │·
  └────────              └──────────            └──────────
  欠擬合                  良好擬合               過擬合
  Underfitting           Good fit              Overfitting
```

### 4.4 數學直覺 Mathematical Intuition

假設真實函數為 $f(x)$，模型預測為 $\hat{f}(x)$，噪音為 $\epsilon \sim N(0, \sigma^2)$。

- **偏差 (Bias)**：$\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$
  - 衡量模型「平均而言」離真實值有多遠
- **變異 (Variance)**：$\text{Var}[\hat{f}(x)] = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$
  - 衡量模型對不同訓練集的「敏感度」

**關鍵洞察：** 隨著模型複雜度增加，偏差通常下降但變異上升。最佳模型複雜度在兩者的交叉點附近。

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="偏差-變異權衡曲線">
  <rect x="0" y="0" width="640" height="320" fill="#ffffff"/>
  <!-- Plot area -->
  <rect x="80" y="40" width="500" height="220" fill="#fafafa" stroke="#e5e7eb"/>
  <!-- Axes -->
  <line x1="80" y1="260" x2="580" y2="260" stroke="#374151" stroke-width="1.2"/>
  <line x1="80" y1="40" x2="80" y2="260" stroke="#374151" stroke-width="1.2"/>
  <!-- Axis labels -->
  <text x="330" y="295" text-anchor="middle" font-size="12" fill="#111827">模型複雜度 Model Complexity →</text>
  <text x="40" y="150" text-anchor="middle" font-size="12" fill="#111827" transform="rotate(-90 40 150)">誤差 Error</text>
  <text x="100" y="275" font-size="10" fill="#6b7280">欠擬合 Underfit</text>
  <text x="560" y="275" text-anchor="end" font-size="10" fill="#6b7280">過擬合 Overfit</text>
  <!-- Bias² curve — starts HIGH, decays toward 0 -->
  <path d="M 80 70 Q 180 100 280 170 T 500 245 L 580 250" fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <text x="140" y="92" font-size="12" fill="#1e3a8a" font-weight="600">Bias²（偏差²）</text>
  <!-- Variance curve — starts LOW, grows superlinearly -->
  <path d="M 80 252 Q 200 245 300 225 T 480 140 Q 540 70 580 50" fill="none" stroke="#ef4444" stroke-width="2.5"/>
  <text x="520" y="60" text-anchor="end" font-size="12" fill="#991b1b" font-weight="600">Variance（變異）</text>
  <!-- Total error — U shape = bias + variance + irreducible noise -->
  <path d="M 80 80 Q 200 150 330 130 Q 420 120 480 160 Q 540 200 580 220" fill="none" stroke="#111827" stroke-width="3"/>
  <text x="340" y="118" text-anchor="middle" font-size="12" fill="#111827" font-weight="700">Total Error（總誤差）</text>
  <!-- Irreducible noise floor -->
  <line x1="80" y1="230" x2="580" y2="230" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="575" y="224" text-anchor="end" font-size="10" fill="#6b7280">不可約誤差 Irreducible noise</text>
  <!-- Optimum marker -->
  <line x1="335" y1="40" x2="335" y2="260" stroke="#059669" stroke-width="1.5" stroke-dasharray="6 3"/>
  <circle cx="335" cy="127" r="5" fill="#059669" stroke="#065f46" stroke-width="1.5"/>
  <text x="335" y="34" text-anchor="middle" font-size="11" fill="#065f46" font-weight="600">最佳複雜度 Sweet spot</text>
</svg>
<figcaption>示意圖：偏差-變異權衡。隨模型複雜度提升，Bias² 單調下降、Variance 單調上升；兩者之和加上不可約噪聲構成總誤差（黑線呈 U 型）。最佳複雜度位於 U 型曲線的谷底，即 Bias² 與 Variance 大致平衡之處。</figcaption>
</figure>
```

---

## 5. 交叉驗證 Cross-Validation

### 5.1 為什麼需要交叉驗證？ Why Cross-Validation?

單次的 train/test 分割有以下問題：

1. **結果不穩定**：不同的隨機分割可能產生不同的評估結果
2. **資料利用不充分**：大量資料被鎖在測試集中，無法用於訓練
3. **小樣本問題**：當資料量有限時，單次分割的評估可能不可靠

交叉驗證 (Cross-Validation, CV) 透過**多次分割與評估**來解決這些問題。

### 5.2 k 折交叉驗證 k-Fold Cross-Validation

**原理：** 將資料均等分割為 $k$ 份 (Folds)，輪流用其中 1 份作為驗證集、其餘 $k-1$ 份作為訓練集，共進行 $k$ 次訓練與評估。

```svg
<figure class="md-figure">
<svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="5-Fold Cross-Validation 示意圖">
  <rect x="0" y="0" width="680" height="360" fill="#ffffff"/>
  <text x="340" y="26" text-anchor="middle" font-size="14" fill="#111827" font-weight="600">5-Fold Cross-Validation</text>
  <!-- Column header (5 equal folds) -->
  <g font-size="11" fill="#6b7280" text-anchor="middle">
    <text x="155" y="50">Fold 1</text><text x="255" y="50">Fold 2</text><text x="355" y="50">Fold 3</text><text x="455" y="50">Fold 4</text><text x="555" y="50">Fold 5</text>
  </g>
  <!-- Row labels (iterations) -->
  <g font-size="12" fill="#111827" text-anchor="end" font-weight="600">
    <text x="95" y="84">Iter 1</text><text x="95" y="124">Iter 2</text><text x="95" y="164">Iter 3</text><text x="95" y="204">Iter 4</text><text x="95" y="244">Iter 5</text>
  </g>
  <!-- Grid of 5 rows × 5 columns: validation cell per row shifts -->
  <!-- Iter 1 — val = fold 1 -->
  <rect x="105" y="68" width="100" height="24" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <rect x="205" y="68" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="305" y="68" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="405" y="68" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="505" y="68" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <!-- Iter 2 — val = fold 2 -->
  <rect x="105" y="108" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="205" y="108" width="100" height="24" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <rect x="305" y="108" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="405" y="108" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="505" y="108" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <!-- Iter 3 — val = fold 3 -->
  <rect x="105" y="148" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="205" y="148" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="305" y="148" width="100" height="24" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <rect x="405" y="148" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="505" y="148" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <!-- Iter 4 — val = fold 4 -->
  <rect x="105" y="188" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="205" y="188" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="305" y="188" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="405" y="188" width="100" height="24" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <rect x="505" y="188" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <!-- Iter 5 — val = fold 5 -->
  <rect x="105" y="228" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="205" y="228" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="305" y="228" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="405" y="228" width="100" height="24" fill="#dbeafe" stroke="#1e40af"/>
  <rect x="505" y="228" width="100" height="24" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <!-- Per-iter score labels on the right -->
  <g font-size="11" fill="#111827" text-anchor="start">
    <text x="620" y="84">0.83</text><text x="620" y="124">0.86</text><text x="620" y="164">0.81</text><text x="620" y="204">0.85</text><text x="620" y="244">0.84</text>
  </g>
  <text x="620" y="62" font-size="10" fill="#6b7280">score</text>
  <!-- Legend -->
  <rect x="120" y="284" width="16" height="14" fill="#dbeafe" stroke="#1e40af"/>
  <text x="144" y="296" font-size="12" fill="#1e3a8a">訓練 Train</text>
  <rect x="260" y="284" width="16" height="14" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <text x="284" y="296" font-size="12" fill="#92400e">驗證 Validation</text>
  <!-- Final score formula -->
  <rect x="105" y="318" width="500" height="32" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="355" y="338" text-anchor="middle" font-size="12" fill="#111827">最終分數 = mean(scores) ± std(scores) = <tspan font-weight="600">0.838 ± 0.019</tspan></text>
</svg>
<figcaption>示意圖：5-Fold Cross-Validation。資料均等分為 5 折，每次迭代用其中 1 折當驗證（黃色）、其餘 4 折當訓練（藍色）；共跑 5 次得到 5 個分數，最終以均值 ± 標準差作為穩健效能估計。k 越大偏差越小但計算量越大。</figcaption>
</figure>
```

**使用 scikit-learn 實作：**

```python
from sklearn.model_selection import cross_val_score, KFold

# 建立 KFold 物件
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 進行交叉驗證
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"各 Fold 分數: {scores}")
print(f"平均分數: {scores.mean():.4f} ± {scores.std():.4f}")
```

**k 值的選擇 Choosing k：**

| k 值 | 優點 Pros | 缺點 Cons | 適用場景 Use Case |
|------|----------|----------|------------------|
| k=5 | 計算快、偏差與變異均衡 | 可能變異稍大 | 通用預設值 |
| k=10 | 評估更穩定 | 計算量稍大 | 常見選擇 |
| k=n (LOOCV) | 偏差最低 | 計算量大、變異可能高 | 小資料集 |

### 5.3 分層 k 折交叉驗證 Stratified k-Fold Cross-Validation

**問題場景：** 在分類問題中，如果某些類別的樣本數很少（如疾病檢測中只有 5% 的陽性），隨機分割可能導致某些 Fold 中完全沒有少數類樣本。

**解決方案：** 分層抽樣 (Stratified Sampling) 確保每個 Fold 中各類別的比例與原始資料集一致。

```
原始資料：80% 類別 A (○)、20% 類別 B (●)

一般 KFold（可能的不良分割）：
Fold 1: [○ ○ ○ ○ ○ ○ ○ ○ ○ ○]  ← 驗證集全是 A，沒有 B！
Fold 2: [○ ○ ○ ○ ○ ● ● ● ● ●]  ← B 全集中在這裡

Stratified KFold（比例一致）：
Fold 1: [○ ○ ○ ○ ○ ○ ○ ○ ● ●]  ← 80% A + 20% B
Fold 2: [○ ○ ○ ○ ○ ○ ○ ○ ● ●]  ← 80% A + 20% B
```

**何時必須使用分層抽樣？**
- 類別不平衡 (Imbalanced Classes) 的分類問題
- 資料量較小的情況
- 多類別分類 (Multi-class Classification)

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

### 5.4 Leave-One-Out Cross-Validation (LOOCV)

LOOCV 是 k-Fold 的極端情況，其中 $k = n$（n 為樣本數）。每次只留**一個**樣本作為驗證，其餘所有樣本用於訓練。

```
n 個樣本 → n 次訓練與驗證

Iter 1: [驗] [訓] [訓] [訓] [訓] ... [訓]
Iter 2: [訓] [驗] [訓] [訓] [訓] ... [訓]
Iter 3: [訓] [訓] [驗] [訓] [訓] ... [訓]
   ⋮
Iter n: [訓] [訓] [訓] [訓] [訓] ... [驗]
```

**優點：**
- 幾乎使用了所有資料進行訓練，偏差最小
- 結果是確定性的（不受隨機分割影響）

**缺點：**
- 計算成本非常高（需要訓練 n 個模型）
- 變異可能較高（每次驗證只有 1 個樣本）
- 不適用於大型資料集

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
print(f"LOOCV 準確率: {scores.mean():.4f}")
```

### 5.5 時間序列分割 Time Series Split

**為什麼時間序列不能用一般的交叉驗證？**

一般 CV 假設資料是**獨立同分布 (i.i.d.)** 的，但時間序列資料具有**時間相依性 (Temporal Dependency)**。如果允許使用「未來資料」來訓練、預測「過去資料」，就會造成**前瞻偏誤 (Look-Ahead Bias)**，導致效能評估嚴重高估。

**原則：訓練資料必須在時間上早於測試資料。**

```
時間序列分割 TimeSeriesSplit (n_splits=5):

Split 1: [訓練] | [驗證]
Split 2: [訓練    訓練] | [驗證]
Split 3: [訓練    訓練    訓練] | [驗證]
Split 4: [訓練    訓練    訓練    訓練] | [驗證]
Split 5: [訓練    訓練    訓練    訓練    訓練] | [驗證]
          ──────────────────────────────────────▶ 時間 Time
```

注意每次分割中，訓練集的範圍逐步擴大，驗證集始終在訓練集之後，確保不會使用到「未來」的資訊。

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # 訓練並評估模型...
```

---

## 6. 實務考量與最佳實踐 Practical Considerations & Best Practices

### 6.1 資料分割比例的選擇 Choosing Split Ratios

| 資料量 Data Size | 建議分割 Suggested Split | 說明 |
|-----------------|------------------------|------|
| < 1,000 筆 | 使用交叉驗證 | 資料太少，單次分割不穩定 |
| 1,000 - 100,000 筆 | 60/20/20 或 70/15/15 | 常見的三分法 |
| > 100,000 筆 | 98/1/1 甚至 99/0.5/0.5 | 大資料集中，少量即可代表分布 |

### 6.2 常見錯誤 Common Mistakes

1. **資料洩漏 (Data Leakage)**：在分割前做了資料前處理（如 StandardScaler），導致測試集的統計資訊「洩漏」到訓練過程中
   ```python
   # 錯誤做法 WRONG
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)  # 用全部資料計算均值/標準差
   X_train, X_test = train_test_split(X_scaled)

   # 正確做法 CORRECT
   X_train, X_test = train_test_split(X)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)  # 只用訓練集計算
   X_test = scaler.transform(X_test)        # 用訓練集的參數轉換
   ```

2. **未設定隨機種子**：每次執行結果不同，無法重現
3. **忘記打亂資料**：如果資料按類別排序，不打亂會導致嚴重的分割偏差
4. **對測試集做了多次評估**：反覆用測試集微調模型，等於把測試集變成驗證集

### 6.3 交叉驗證的完整流程 Complete CV Workflow

```
1. 保留測試集（最終評估用）
   Split off Test Set (for final evaluation)
        │
2. 對訓練集進行 k-Fold CV
   Perform k-Fold CV on Training Set
        │
3. 在每個 Fold 內部做資料前處理
   Preprocess within each Fold
        │
4. 計算平均驗證分數，選出最佳超參數
   Compute mean validation score, select best hyperparameters
        │
5. 用最佳超參數在全部訓練集上重新訓練
   Retrain on full Training Set with best hyperparameters
        │
6. 在測試集上評估一次（報告最終結果）
   Evaluate once on Test Set (report final results)
```

---

## 7. 進階交叉驗證方法 Advanced CV Methods

### 7.1 重複 k 折交叉驗證 Repeated k-Fold CV

為了進一步降低隨機性帶來的變異，可以重複多次 k-Fold CV（每次用不同的隨機分割），然後計算所有結果的平均值。

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf, scoring='accuracy')
print(f"5x10 Repeated KFold: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 7.2 群組 k 折 Group k-Fold

當資料中存在**群組結構 (Group Structure)** 時（如同一個病人的多次檢查），必須確保同一群組的資料不會同時出現在訓練集和驗證集中。

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=gkf, groups=patient_ids)
```

---

## 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 監督式學習 | Supervised Learning | 使用標註資料訓練模型的學習方式 |
| 訓練集 | Training Set | 用於訓練模型參數的資料子集 |
| 驗證集 | Validation Set | 用於調整超參數和監控訓練過程的資料子集 |
| 測試集 | Test Set | 用於最終效能評估的保留資料子集 |
| 過擬合 | Overfitting | 模型過度適應訓練資料，泛化能力差 |
| 欠擬合 | Underfitting | 模型過於簡單，無法捕捉資料中的規律 |
| 偏差 | Bias | 模型預測值期望與真實值之間的系統性差異 |
| 變異 | Variance | 模型預測值隨不同訓練集的波動程度 |
| 偏差-變異權衡 | Bias-Variance Tradeoff | 模型複雜度增加時偏差下降但變異上升的現象 |
| 交叉驗證 | Cross-Validation (CV) | 多次分割資料進行訓練與評估的方法 |
| k 折交叉驗證 | k-Fold Cross-Validation | 將資料分為 k 份輪流驗證的技術 |
| 分層抽樣 | Stratified Sampling | 確保各類別比例一致的抽樣方式 |
| 資料洩漏 | Data Leakage | 測試/驗證集的資訊不當流入訓練過程 |
| 泛化能力 | Generalization | 模型在未見過資料上的表現能力 |
| 前瞻偏誤 | Look-Ahead Bias | 時間序列中不當使用未來資訊的偏誤 |
| 時間序列分割 | Time Series Split | 針對時間序列資料的特殊驗證方法 |
| 超參數 | Hyperparameter | 不由模型自動學習，需人為設定的參數 |
| 損失函數 | Loss Function | 衡量模型預測值與真實值差異的函數 |
| 梯度下降 | Gradient Descent | 沿梯度方向更新參數以最小化損失的演算法 |
| 正則化 | Regularization | 限制模型複雜度以防止過擬合的技術 |
| 可重現性 | Reproducibility | 相同條件下能重現相同實驗結果的特性 |

---

## 延伸閱讀 Further Reading

- scikit-learn 官方文件 — Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- James, G. et al., "An Introduction to Statistical Learning (ISLR)" — Chapter 5: Resampling Methods
- Hastie, T. et al., "The Elements of Statistical Learning (ESL)" — Chapter 7: Model Assessment and Selection
- Andrew Ng, "Machine Learning Yearning" — Chapter on Train/Dev/Test Split
