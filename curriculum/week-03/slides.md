# 第 3 週 投影片：監督式學習概念、資料分割與交叉驗證
# Week 3 Slides: Supervised Learning, Data Splitting & Cross-Validation

---

## Slide 1：本週主題 Today's Topic

### 監督式學習、資料分割與交叉驗證
### Supervised Learning, Data Splitting & Cross-Validation

**本週學習重點 Key Topics：**
- 監督式學習 (Supervised Learning) 的完整框架
- 資料分割的原則與實作
- k 折交叉驗證 (k-Fold CV) 與分層抽樣 (Stratified Sampling)
- 偏差-變異權衡 (Bias-Variance Tradeoff)
- 過擬合 (Overfitting) vs 欠擬合 (Underfitting) 的診斷

**先備知識：** Week 1 Python 環境 + Week 2 EDA 與視覺化

---

## Slide 2：監督式學習框架 Supervised Learning Framework

### 核心流程 Core Pipeline

```
  特徵 X ──▶ 模型 f(x;θ) ──▶ 預測 ŷ ──▶ 損失 L(ŷ, y) ──▶ 更新 θ
  Features    Model          Prediction   Loss             Update
                ▲                                            │
                └────────────────────────────────────────────┘
```

| 步驟 | 說明 |
|------|------|
| 1. 前向傳播 Forward Pass | 輸入特徵 → 模型計算預測值 |
| 2. 計算損失 Compute Loss | 比較預測值 ŷ 與真實值 y |
| 3. 反向傳播 Backward Pass | 計算梯度 ∇L |
| 4. 參數更新 Update | θ ← θ - η∇L |

**兩大任務：**
- **回歸 (Regression)**：預測連續值（房價、溫度）
- **分類 (Classification)**：預測離散類別（貓/狗、正/負）

> 教學提示：用「學生考試」的類比 — 模型 = 學生，損失 = 扣分，更新 = 改進讀書方法

---

## Slide 3：為什麼需要分割資料？ Why Split Data?

### 考試類比 Exam Analogy

| 情境 | 類比 | 結果 |
|------|------|------|
| 用全部資料訓練 + 評估 | 考古題完全一樣 | 只測到「背誦」，不測「理解」 |
| 用訓練集訓練 + 測試集評估 | 出新題考試 | 真正測到「泛化能力」 |

### 核心原則

> **評估模型時，必須使用模型在訓練過程中從未見過的資料。**

**目的：** 防止過擬合 (Overfitting)，確保模型的泛化能力 (Generalization)

---

## Slide 4：三種資料集的角色 Three Dataset Roles

```
┌──────────────────────────────────────────────────────┐
│                完整資料集 Full Dataset                  │
├────────────────┬──────────┬──────────────────────────┤
│  訓練集 60-70%  │ 驗證集    │     測試集 15-20%         │
│ Training Set   │ 10-15%   │     Test Set             │
│                │ Val Set  │                          │
│ 學習模型參數     │ 調超參數   │    最終效能評估            │
│ Learn params   │ Tune HP  │    Final evaluation      │
└────────────────┴──────────┴──────────────────────────┘
```

| 資料集 | 用途 | 使用時機 | 使用次數 |
|--------|------|---------|---------|
| 訓練集 Training | 擬合模型參數 | 每個 Epoch | 多次 |
| 驗證集 Validation | 選超參數、監控過擬合 | 訓練過程中 | 多次 |
| 測試集 Test | 報告最終效能 | 訓練完成後 | **一次** |

**`train_test_split` 用法：**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## Slide 5：過擬合 vs 欠擬合 Overfitting vs Underfitting

### 視覺化對比 Visual Comparison

| | 欠擬合 Underfitting | 良好擬合 Good Fit | 過擬合 Overfitting |
|---|---|---|---|
| 圖示 | 直線穿過曲線資料 | 平滑曲線貼合趨勢 | 鋸齒曲線穿過每個點 |
| 訓練誤差 | 高 | 低 | 非常低 |
| 測試誤差 | 高 | 低 | 高 |
| 模型複雜度 | 太低 | 適當 | 太高 |
| 類比 | 沒讀書 | 真正理解 | 死背考古題 |

### 診斷方法 Diagnosis

```
    誤差                        ╱ 測試誤差
    Error                     ╱  Test Error
      │                     ╱
      │                   ╱
      │    ╲            ╱
      │     ╲    最佳  ╱
      │      ╲── ↓ ──╱
      │       ╲     ╱
      │     訓練誤差 ╲╱──────────────▶
      │     Training Error
      └──────────────────────────────▶
             模型複雜度 Complexity
```

> 觀察重點：訓練誤差與測試誤差之間的**差距 (Gap)**

---

## Slide 6：偏差-變異權衡 Bias-Variance Tradeoff

### 射擊靶心類比 Bullseye Analogy

```
 高偏差+低變異        低偏差+低變異       低偏差+高變異       高偏差+高變異
 (欠擬合)           (最佳!)            (過擬合)          (最差)
  ·· ·    ◎          ·◎·               ·    ◎    ·        ·     ◎
   ··                 ·                    ·     ·            ·  ·
                                        ·                  ·
```

### 誤差分解 Error Decomposition

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

| 模型複雜度 ↑ | Bias | Variance | 現象 |
|-------------|------|----------|------|
| 簡單模型 | 高 ↑ | 低 ↓ | 欠擬合 |
| 適當模型 | 適中 | 適中 | 最佳泛化 |
| 複雜模型 | 低 ↓ | 高 ↑ | 過擬合 |

> 本週實作：用多項式回歸 (Polynomial Regression) 不同 degree 視覺化觀察此權衡

---

## Slide 7：k 折交叉驗證 k-Fold Cross-Validation

### 原理 Principle

將資料分為 k 份，輪流用 1 份當驗證集、其餘 k-1 份當訓練集

```
5-Fold CV:
Fold 1: [驗證] [訓練] [訓練] [訓練] [訓練]  → Score₁
Fold 2: [訓練] [驗證] [訓練] [訓練] [訓練]  → Score₂
Fold 3: [訓練] [訓練] [驗證] [訓練] [訓練]  → Score₃
Fold 4: [訓練] [訓練] [訓練] [驗證] [訓練]  → Score₄
Fold 5: [訓練] [訓練] [訓練] [訓練] [驗證]  → Score₅

Final = Mean(Score₁...Score₅) ± Std
```

### 程式碼 Code
```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
```

### k 值如何選？ How to Choose k?
- **k=5 或 k=10**：最常用，計算量與評估穩定性的平衡
- **k=n (LOOCV)**：偏差最小但計算量最大，適用小資料集

---

## Slide 8：分層抽樣 Stratified Sampling

### 為什麼需要分層？ Why Stratify?

**問題：** 類別不平衡時，普通隨機分割可能造成某些 Fold 中少數類比例嚴重失衡

| | 普通 KFold | Stratified KFold |
|---|---|---|
| 原始比例 | 90% 負 / 10% 正 | 90% 負 / 10% 正 |
| Fold 1 可能 | 95% 負 / 5% 正 | 90% 負 / 10% 正 |
| Fold 2 可能 | 100% 負 / 0% 正 | 90% 負 / 10% 正 |

### 實作 Code
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

### 何時必須使用？ When to Use?
- 類別不平衡 (Imbalanced) 的分類問題
- 資料量較小時
- 多類別分類

> `train_test_split(X, y, stratify=y)` 也支援分層！

---

## Slide 9：LOOCV 與時間序列分割 LOOCV & Time Series Split

### Leave-One-Out CV (LOOCV)

- k = n（每次只留 1 個樣本驗證）
- 優點：偏差最小、結果確定性
- 缺點：計算量 O(n)、變異可能較高
- 適用：**小型資料集**（n < 100）

### 時間序列分割 Time Series Split

**核心原則：不能用「未來」預測「過去」**

```
Split 1: [====訓練====] | [=驗證=]
Split 2: [========訓練========] | [=驗證=]
Split 3: [============訓練============] | [=驗證=]
         ──────────────────────────────────▶ 時間
```

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

> 金融、氣象、IoT 等領域必須使用時間序列分割！

---

## Slide 10：資料洩漏 Data Leakage — 最常見的陷阱

### 什麼是資料洩漏？

測試/驗證集的資訊**不當流入**訓練過程，導致效能被**高估**

### 典型錯誤範例

```python
# 錯誤 WRONG — 先標準化，再分割
scaler.fit_transform(X)       # ← 用了全部資料的統計量
X_train, X_test = split(X)

# 正確 CORRECT — 先分割，再標準化
X_train, X_test = split(X)
scaler.fit(X_train)           # ← 只用訓練集
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # ← 用訓練集的參數
```

### 其他洩漏情境
- 使用「未來資訊」做特徵（時間序列）
- 重複資料 (Duplicates) 同時出現在訓練與測試集
- 目標編碼 (Target Encoding) 未在 CV 內部進行

---

## Slide 11：完整工作流程 Complete Workflow

```
      全部資料
         │
    ┌────┴────┐
    │         │
  訓練集    測試集（封存）
    │
  k-Fold CV
    │
 ┌──┼──┬──┬──┐
 F1 F2 F3 F4 F5    ← 每個 Fold 內部做前處理
 │  │  │  │  │
 S1 S2 S3 S4 S5    ← 各 Fold 驗證分數
    │
  平均分數 → 選出最佳超參數
    │
  全訓練集 + 最佳超參數 → 重新訓練
    │
  測試集 → 最終報告（只做一次）
```

### 關鍵記憶點 Key Takeaways
1. **測試集是「密封試卷」**— 只用一次
2. **前處理在 CV 內部做** — 防止資料洩漏
3. **分類問題用 Stratified** — 維持類別比例
4. **時間序列用 TimeSeriesSplit** — 不能「看未來」

---

## Slide 12：本週實作預覽與作業 Lab Preview & Assignment

### 本週 Notebook 實作內容
1. `train_test_split` 實作與分割結果視覺化
2. k-Fold CV 視覺化 — 觀察每個 Fold 的分割情況
3. Stratified vs Non-Stratified 比較實驗
4. 偏差-變異權衡圖 — 多項式回歸不同 degree
5. 過擬合/欠擬合診斷視覺化

### 本週作業 Assignment
- 第一部分：概念理解題
- 第二部分：`train_test_split` 應用
- 第三部分：k-Fold CV 實作與分析
- 第四部分：偏差-變異權衡實驗
- 第五部分：綜合挑戰 — 完整 CV Pipeline

### 下週預告 Next Week Preview
**第 4 週：線性回歸 — 損失函數、梯度下降視覺化**
- 損失函數 (Loss Function) 與均方誤差 (MSE)
- 梯度下降 (Gradient Descent) 的互動視覺化
- 學習率 (Learning Rate) 對收斂的影響
