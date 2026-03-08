# 第 7 週：樹模型與集成（RF、GBDT）
# Week 7: Tree Models & Ensemble Methods (RF, GBDT)

## 學習目標 Learning Objectives
1. 理解決策樹 (Decision Tree) 的建構原理與分裂準則
2. 掌握剪枝 (Pruning) 策略以防止過擬合
3. 了解集成學習 (Ensemble Learning) 的核心思想
4. 比較 Bagging（隨機森林 Random Forest）與 Boosting（梯度提升 GBDT）的差異
5. 透過視覺化觀察樹的生長過程與集成效果

---

## 1. 決策樹 Decision Tree

### 1.1 什麼是決策樹？ What is a Decision Tree?

決策樹是一種基於樹狀結構 (Tree Structure) 的監督式學習演算法，可同時用於分類 (Classification) 與回歸 (Regression) 任務。其核心思想是：**透過一系列的 if-else 規則，將特徵空間 (Feature Space) 遞迴地切割為越來越純的子區域**。

決策樹由以下元素組成：

| 元素 | 英文 | 說明 |
|------|------|------|
| 根節點 | Root Node | 包含完整資料集的起始節點 |
| 內部節點 | Internal Node | 依特徵進行分裂的決策節點 |
| 分支 | Branch | 根據特徵值的判斷結果走向不同路徑 |
| 葉節點 | Leaf Node | 最終的預測結果（類別或數值） |

```
           [花瓣長度 <= 2.45?]         <-- 根節點 Root
            /                \
          是 Yes              否 No
          /                    \
     [Setosa]        [花瓣寬度 <= 1.75?]   <-- 內部節點
                      /              \
                    是 Yes           否 No
                    /                  \
           [Versicolor]           [Virginica]   <-- 葉節點 Leaves
```

### 1.2 分裂準則 Splitting Criteria

建構決策樹的關鍵在於：**如何選擇最佳的特徵與閾值來進行分裂？** 我們需要一個指標來衡量分裂前後「純度」(Purity) 的提升。

#### 1.2.1 熵 Entropy

熵 (Entropy) 源自資訊論 (Information Theory)，衡量資料集的不確定性 (Uncertainty)：

$$H(S) = -\sum_{i=1}^{c} p_i \log_2 p_i$$

其中 $p_i$ 是類別 $i$ 在資料集 $S$ 中的比例，$c$ 是類別數。

- 熵 = 0：資料完全純淨（只有一個類別）
- 熵 = 1（二元分類時最大值）：資料最混亂（各類別等比例）

**直覺理解：** 想像一個裝有彩色球的袋子。如果所有球都是同一顏色，你完全確定下次取出的顏色（熵 = 0）；如果各顏色等量，你最不確定（熵最大）。

#### 1.2.2 資訊增益 Information Gain

資訊增益 (Information Gain, IG) 衡量分裂前後熵的減少量：

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

其中 $A$ 是分裂特徵，$S_v$ 是特徵 $A$ 取值 $v$ 後的子集。

**選擇使資訊增益最大化的特徵進行分裂**，這就是 ID3 演算法的核心。

**範例計算：**
假設有 10 筆資料，6 筆正類 (+)、4 筆負類 (-)：
$$H(S) = -\frac{6}{10}\log_2\frac{6}{10} - \frac{4}{10}\log_2\frac{4}{10} \approx 0.971$$

若以特徵 A 分裂後：
- 左子集：5 筆（4+, 1-），$H(S_L) \approx 0.722$
- 右子集：5 筆（2+, 3-），$H(S_R) \approx 0.971$

$$IG = 0.971 - \frac{5}{10}(0.722) - \frac{5}{10}(0.971) \approx 0.124$$

#### 1.2.3 基尼不純度 Gini Impurity

基尼不純度 (Gini Impurity) 是另一種衡量純度的指標，計算上比熵更簡單：

$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

- Gini = 0：完全純淨
- Gini = 0.5（二元分類時最大值）：最不純

**直覺理解：** 隨機取兩筆資料，兩者類別不同的機率。Gini = 0 表示不可能取到不同類別。

**熵 vs. 基尼不純度：**

| 特性 | 熵 Entropy | 基尼 Gini |
|------|-----------|-----------|
| 計算複雜度 | 需要對數運算 | 只需平方運算，較快 |
| 值域（二元） | [0, 1] | [0, 0.5] |
| 偏好 | 偏好平衡分裂 | 偏好將最大類別隔離 |
| 使用場景 | ID3, C4.5 | CART, sklearn 預設 |

在實務上，兩者產生的決策樹通常差異不大（僅約 2% 的情況不同）。

### 1.3 CART 演算法

CART (Classification and Regression Trees) 是最廣泛使用的決策樹演算法，由 Breiman 等人於 1984 年提出。

**CART 的核心特點：**

1. **二元分裂 (Binary Split)：** 每個節點只分裂為兩個子節點
2. **分類樹：** 使用基尼不純度 (Gini Impurity) 作為分裂準則
3. **回歸樹：** 使用均方誤差 (MSE) 最小化，即選擇使各子區域內預測值的方差最小的分裂

**CART 分類樹的建構流程：**

```
1. 從根節點開始，包含所有訓練樣本
2. 對每個特徵的每個可能閾值：
   a. 計算分裂後的加權基尼不純度
   b. 選擇使基尼不純度降低最多的 (特徵, 閾值) 組合
3. 依最佳分裂將資料分為兩個子集
4. 對每個子集遞迴重複步驟 2-3
5. 當滿足停止條件時停止：
   - 節點內所有樣本屬於同一類別
   - 達到最大深度 (max_depth)
   - 節點內樣本數低於門檻 (min_samples_split)
   - 分裂後的改善量低於門檻 (min_impurity_decrease)
```

**CART 回歸樹：**

回歸樹的分裂準則是最小化均方誤差 (MSE)：

$$\text{MSE} = \frac{1}{|S_L|}\sum_{i \in S_L}(y_i - \bar{y}_L)^2 + \frac{1}{|S_R|}\sum_{i \in S_R}(y_i - \bar{y}_R)^2$$

每個葉節點的預測值是該區域內所有樣本目標值的平均。

### 1.4 剪枝 Pruning

未經限制的決策樹會持續生長直到每個葉節點都完全純淨，這通常導致嚴重的過擬合 (Overfitting)。剪枝是控制樹複雜度的重要技術。

#### 1.4.1 預剪枝 Pre-pruning（提前停止 Early Stopping）

在樹的生長過程中設定停止條件，**防止樹長得太大**：

| 參數 | sklearn 參數名 | 說明 |
|------|---------------|------|
| 最大深度 | `max_depth` | 限制樹的最大層數 |
| 最小分裂樣本數 | `min_samples_split` | 節點至少需要這麼多樣本才能分裂 |
| 最小葉節點樣本數 | `min_samples_leaf` | 每個葉節點至少需要這麼多樣本 |
| 最大葉節點數 | `max_leaf_nodes` | 限制葉節點的總數 |
| 最小不純度減少量 | `min_impurity_decrease` | 分裂必須帶來足夠的不純度改善 |

**優點：** 計算效率高，訓練過程中直接控制
**缺點：** 可能過早停止，錯過後續有價值的分裂

#### 1.4.2 後剪枝 Post-pruning

先讓樹完全生長，再自底向上 (Bottom-up) 地移除不顯著的子樹：

**代價複雜度剪枝 (Cost-Complexity Pruning, CCP)：**

最小化以下目標函數：

$$R_\alpha(T) = R(T) + \alpha |T|$$

其中：
- $R(T)$：樹 $T$ 的訓練誤差（如不純度之和）
- $|T|$：葉節點數量（衡量複雜度）
- $\alpha$：複雜度參數 (Complexity Parameter)，控制懲罰力度

在 scikit-learn 中，使用 `ccp_alpha` 參數進行代價複雜度剪枝。$\alpha$ 越大，剪枝越積極，樹越簡單。

**優點：** 不會錯過有價值的分裂，通常產生更好的樹
**缺點：** 計算成本較高，需要先完整生長再修剪

---

## 2. 集成學習 Ensemble Learning

### 2.1 為什麼需要集成？ Why Ensemble?

> "The wisdom of crowds" — 群體的智慧

單一模型往往有其局限性：
- **高變異 (High Variance)：** 決策樹對資料微小變化非常敏感
- **高偏差 (High Bias)：** 簡單模型可能無法捕捉複雜模式
- **不穩定 (Instability)：** 訓練資料的些微變動可能導致完全不同的模型

集成學習的核心思想是：**結合多個弱學習器 (Weak Learners) 來構建一個強學習器 (Strong Learner)**。

**Condorcet 陪審團定理的啟示：**
如果每個「陪審員」（模型）正確的機率大於 50%，且彼此獨立，那麼多數決的正確率會隨人數增加而趨近 100%。

集成方法主要分為兩大類：

| 方法 | 核心策略 | 代表演算法 | 目標 |
|------|---------|-----------|------|
| Bagging | 並行訓練 + 投票/平均 | Random Forest | 降低變異 (Variance) |
| Boosting | 序列訓練 + 加權累加 | GBDT, XGBoost, LightGBM | 降低偏差 (Bias) |

### 2.2 Bagging 方法

Bagging (Bootstrap AGGregatING) 由 Leo Breiman 於 1996 年提出。

**核心思想：透過對訓練資料進行多次 Bootstrap 取樣，訓練多個獨立的模型，再將結果聚合。**

#### 2.2.1 Bootstrap 取樣

Bootstrap 是一種有放回抽樣 (Sampling with Replacement) 的統計方法：

- 從 $N$ 筆訓練資料中，有放回地抽取 $N$ 筆組成一個新的訓練集
- 每次 Bootstrap 抽樣大約有 63.2% 的原始資料被選中（$1 - (1 - 1/N)^N \approx 1 - 1/e \approx 0.632$）
- 未被選中的約 36.8% 資料稱為 **Out-of-Bag (OOB) 樣本**，可用於估計泛化誤差

#### 2.2.2 Bagging 流程

```
1. 對訓練集進行 B 次 Bootstrap 取樣，得到 B 個子訓練集
2. 在每個子訓練集上獨立訓練一個基學習器（通常是決策樹）
3. 聚合所有基學習器的預測結果：
   - 分類任務：多數投票 (Majority Voting)
   - 回歸任務：平均值 (Averaging)
```

**為什麼 Bagging 能降低變異？**

假設 $B$ 個基學習器的預測彼此獨立，每個的變異為 $\sigma^2$，則聚合後的變異為 $\sigma^2 / B$。即使不完全獨立（相關係數 $\rho$），變異也能降至 $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$，仍然比單一模型小。

---

## 3. 隨機森林 Random Forest

### 3.1 概念

隨機森林 (Random Forest, RF) 是 Bagging 的進化版，由 Leo Breiman 於 2001 年提出。它在 Bagging 的基礎上加入了**特徵隨機子集 (Random Feature Subset)** 的機制，進一步降低了樹與樹之間的相關性。

### 3.2 兩層隨機性 Double Randomness

| 隨機層次 | 方法 | 目的 |
|---------|------|------|
| 資料隨機 | Bootstrap 取樣 | 每棵樹看到不同的資料 |
| 特徵隨機 | 每次分裂只考慮隨機選取的 $m$ 個特徵 | 降低樹間相關性 |

特徵子集大小 $m$ 的經驗值：
- 分類：$m = \sqrt{p}$（$p$ 為總特徵數）
- 回歸：$m = p/3$

### 3.3 隨機森林演算法

```
輸入：訓練集 D, 樹的數量 B, 特徵子集大小 m
輸出：隨機森林模型

for b = 1 to B:
    1. 從 D 中 Bootstrap 取樣得到 D_b
    2. 在 D_b 上建構決策樹 T_b：
       - 在每個節點：
         a. 從 p 個特徵中隨機選取 m 個候選特徵
         b. 從 m 個特徵中選擇最佳分裂
         c. 分裂節點
       - 生長至最大深度（通常不剪枝）
    3. 將 T_b 加入森林

預測時：
  分類 → f(x) = 多數投票({T_1(x), T_2(x), ..., T_B(x)})
  回歸 → f(x) = (1/B) Σ T_b(x)
```

### 3.4 隨機森林的關鍵超參數

| 參數 | sklearn 名稱 | 建議範圍 | 影響 |
|------|-------------|---------|------|
| 樹的數量 | `n_estimators` | 100-1000 | 越多越穩定，但計算成本增加 |
| 最大特徵數 | `max_features` | 'sqrt', 'log2' | 越少→多樣性高，但單樹能力降 |
| 最大深度 | `max_depth` | None 或 10-30 | 限制過擬合 |
| 最小分裂樣本 | `min_samples_split` | 2-20 | 正則化效果 |
| OOB 評估 | `oob_score` | True/False | 免費的交叉驗證替代方案 |

### 3.5 OOB 估計 Out-of-Bag Estimation

每棵樹都有約 36.8% 的資料未被用於訓練（OOB 樣本）。利用這些「隱形的驗證集」：

1. 對每筆資料，找出所有「未使用它的樹」
2. 用這些樹對它進行預測
3. 聚合預測結果作為 OOB 預測
4. 計算 OOB 分數

**OOB 估計的精度接近 k-fold 交叉驗證，但計算成本更低。**

---

## 4. Boosting 與梯度提升 Gradient Boosting

### 4.1 Boosting 的核心思想

Boosting 的策略與 Bagging 完全不同：

- **Bagging：** 多個模型並行 (Parallel) 訓練，各自獨立
- **Boosting：** 多個模型序列 (Sequential) 訓練，每個新模型專注於修正前面模型的錯誤

**關鍵觀念：** Boosting 將多個「弱學習器」(Weak Learner，表現僅比隨機猜測好一些的模型) 逐步累加，最終組成一個強學習器。

### 4.2 梯度提升 Gradient Boosting

梯度提升決策樹 (Gradient Boosted Decision Trees, GBDT) 由 Friedman 於 1999 年提出，是目前結構化資料 (Tabular Data) 上最強大的演算法之一。

**核心思想：每一棵新的樹學習的是前面所有樹的殘差 (Residuals)。**

#### 4.2.1 GBDT 演算法流程

```
初始化：F_0(x) = argmin_γ Σ L(y_i, γ)    (例如：平均值)

for m = 1 to M:  (M 是總迭代次數/樹的數量)
    1. 計算負梯度（偽殘差 Pseudo-residuals）：
       r_im = -[∂L(y_i, F(x_i)) / ∂F(x_i)] 在 F = F_{m-1} 處

    2. 在 {(x_i, r_im)} 上擬合一棵回歸樹 h_m(x)

    3. 計算最優步長 γ_m（或使用固定學習率 η）

    4. 更新模型：
       F_m(x) = F_{m-1}(x) + η · h_m(x)

最終預測：F_M(x) = F_0(x) + η · Σ_{m=1}^{M} h_m(x)
```

#### 4.2.2 直覺理解殘差學習

以回歸任務為例：

| 步驟 | 真實值 $y$ | 第 1 棵樹預測 | 殘差 | 第 2 棵樹預測 | 累積預測 | 新殘差 |
|------|-----------|-------------|------|-------------|---------|-------|
| 樣本 1 | 10 | 8 | +2 | +1.5 | 9.5 | +0.5 |
| 樣本 2 | 5 | 6 | -1 | -0.8 | 5.2 | -0.2 |
| 樣本 3 | 15 | 12 | +3 | +2.5 | 14.5 | +0.5 |

可以看到殘差在逐步縮小，模型的預測越來越精確。

#### 4.2.3 學習率的角色 Learning Rate

學習率 $\eta$（通常 0.01-0.3）是 GBDT 的關鍵超參數：

- **較小的學習率：** 每棵樹的貢獻較小，需要更多的樹，但泛化能力通常更好（正則化效果）
- **較大的學習率：** 學習更快，但容易過擬合
- **經驗法則：** 使用較小的學習率 + 較多的樹，通常能獲得更好的效果

### 4.3 GBDT 的關鍵超參數

| 參數 | 說明 | 建議範圍 |
|------|------|---------|
| `n_estimators` | 樹的數量（迭代次數） | 100-3000 |
| `learning_rate` | 學習率 (Shrinkage) | 0.01-0.3 |
| `max_depth` | 每棵樹的最大深度 | 3-8（通常較淺） |
| `subsample` | 每棵樹使用的樣本比例 | 0.5-1.0 |
| `min_samples_leaf` | 葉節點最小樣本數 | 5-50 |

**注意：** GBDT 中的樹通常較淺（depth 3-8），因為每棵樹只需要學習殘差中的簡單模式。

### 4.4 XGBoost 簡介

XGBoost (eXtreme Gradient Boosting) 由陳天奇於 2016 年提出，是 GBDT 的工程優化版本：

**主要改進：**
1. **正則化目標函數：** 在損失函數中加入樹複雜度的 L1/L2 懲罰項
   $$Obj = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$
   $$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$
2. **二階泰勒展開：** 使用梯度和 Hessian（二階導數）進行更精確的優化
3. **稀疏感知 (Sparsity-Aware)：** 自動處理缺失值
4. **近似分裂算法：** 使用加權分位數草圖 (Weighted Quantile Sketch) 加速分裂點搜尋
5. **系統優化：** 支援多執行緒、快取優化、分散式訓練

### 4.5 LightGBM 簡介

LightGBM 由微軟於 2017 年提出，專為大規模資料設計：

**主要特色：**
1. **Leaf-wise 生長策略：** 不同於 Level-wise（每層均勻擴展），LightGBM 選擇損失減少最大的葉節點進行分裂，效率更高
2. **Histogram-based 分裂：** 將連續特徵離散化為直方圖 (Histogram)，大幅加速
3. **GOSS (Gradient-based One-Side Sampling)：** 保留大梯度樣本，隨機取樣小梯度樣本
4. **EFB (Exclusive Feature Bundling)：** 將互斥特徵綁定，降低維度

```
Level-wise（XGBoost 預設）：           Leaf-wise（LightGBM 預設）：

       O                                    O
      / \                                  / \
     O   O        <- 每層都展開           O   O
    / \ / \                              / \
   O  O O  O                           O   O   <- 只展開損失最大的葉
                                           / \
                                          O   O
```

---

## 5. Bagging vs. Boosting 完整比較

| 比較面向 | Bagging (Random Forest) | Boosting (GBDT) |
|---------|------------------------|-----------------|
| 訓練方式 | 並行 (Parallel) | 序列 (Sequential) |
| 基學習器 | 獨立、深樹（低偏差） | 依賴前一個、淺樹（低複雜度） |
| 目標 | 降低變異 (Reduce Variance) | 降低偏差 (Reduce Bias) |
| 過擬合風險 | 較低（增加樹數通常不會過擬合） | 較高（樹數過多可能過擬合） |
| 雜訊敏感度 | 較強健 | 對雜訊和離群值較敏感 |
| 訓練速度 | 可並行化，較快 | 必須序列，較慢 |
| 預測效能 | 良好 | 通常優於 Bagging |
| 調參難度 | 較簡單 | 超參數較多，需仔細調校 |
| 適用場景 | 資料雜訊多、需快速基線 | 追求最佳效能、結構化資料 |

### 何時選擇哪個？

- **優先嘗試 Random Forest：** 當你需要一個穩定、不易過擬合的基線模型，或資料有較多雜訊
- **優先嘗試 GBDT/XGBoost/LightGBM：** 當你追求最佳預測效能，且願意花時間調參
- **Kaggle 競賽中的經驗：** 結構化資料的排行榜幾乎被 Boosting 方法霸占

---

## 6. 樹模型的優缺點 Pros and Cons

### 6.1 優點 Advantages

1. **可解釋性 (Interpretability)：** 決策樹的 if-else 規則直觀易懂，可視覺化呈現
2. **不需特徵縮放 (No Feature Scaling)：** 樹模型基於分裂而非距離，不受特徵尺度影響
3. **自動處理非線性 (Nonlinearity)：** 能捕捉複雜的非線性關係
4. **處理混合型特徵 (Mixed Features)：** 同時處理數值型與類別型特徵
5. **特徵重要度 (Feature Importance)：** 內建特徵重要性排序
6. **對缺失值有一定容忍力：** 特別是 XGBoost 和 LightGBM
7. **集成方法效能卓越：** 在結構化/表格資料上常是最佳選擇

### 6.2 缺點 Disadvantages

1. **過擬合傾向 (Overfitting)：** 單棵深樹容易過擬合（集成可緩解）
2. **不穩定性 (Instability)：** 訓練資料的小變動可能導致完全不同的樹
3. **對線性關係效率低：** 需要很多分裂才能近似線性函數
4. **外推能力差 (Poor Extrapolation)：** 無法預測訓練資料範圍之外的值（回歸樹）
5. **高維稀疏資料：** 不如線性模型或深度學習
6. **類別不平衡：** 需要額外處理（如 class_weight 或重抽樣）

---

## 7. 偏差-變異分解在集成方法中的角色

理解 Bagging 和 Boosting 的另一個角度是偏差-變異分解 (Bias-Variance Decomposition)：

$$\text{總誤差} = \text{偏差}^2 + \text{變異} + \text{不可約誤差}$$

| 情境 | 偏差 | 變異 | 解法 |
|------|------|------|------|
| 單棵深樹 | 低 | 高 | Bagging → 降低變異 |
| 單棵淺樹 | 高 | 低 | Boosting → 降低偏差 |

- **Random Forest** 使用多棵完全生長的深樹（低偏差），透過平均化降低變異
- **GBDT** 使用多棵淺樹（低變異），透過逐步修正降低偏差

---

## 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 決策樹 | Decision Tree | 基於 if-else 規則的樹狀分類/回歸模型 |
| 熵 | Entropy | 衡量資料集不確定性的指標 |
| 資訊增益 | Information Gain | 分裂前後熵的減少量 |
| 基尼不純度 | Gini Impurity | 衡量節點純度的指標 |
| CART | Classification and Regression Trees | 二元分裂決策樹演算法 |
| 剪枝 | Pruning | 控制樹複雜度以防止過擬合的技術 |
| 預剪枝 | Pre-pruning | 在生長過程中提前停止 |
| 後剪枝 | Post-pruning | 先完全生長再修剪 |
| 集成學習 | Ensemble Learning | 結合多個模型以提升效能的方法 |
| Bagging | Bootstrap Aggregating | 並行訓練多個模型後聚合 |
| Bootstrap | Bootstrap Sampling | 有放回抽樣 |
| 隨機森林 | Random Forest | Bagging + 特徵隨機子集的集成方法 |
| OOB 分數 | Out-of-Bag Score | 利用未取樣資料估計的泛化分數 |
| Boosting | Boosting | 序列訓練模型，逐步修正錯誤 |
| 梯度提升 | Gradient Boosting (GBDT) | 以梯度下降思想修正殘差的 Boosting 方法 |
| 殘差 | Residual | 真實值與預測值的差 |
| 學習率 | Learning Rate / Shrinkage | 每棵樹的貢獻權重 |
| 弱學習器 | Weak Learner | 表現僅略優於隨機猜測的模型 |
| 強學習器 | Strong Learner | 泛化能力強的最終集成模型 |
| 特徵重要度 | Feature Importance | 特徵對模型預測的貢獻程度 |
| XGBoost | eXtreme Gradient Boosting | 高效能的梯度提升實作 |
| LightGBM | Light Gradient Boosting Machine | 微軟開發的輕量高效 GBDT 框架 |

---

## 延伸閱讀 Further Reading
- Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine."
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System."
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
- scikit-learn Ensemble Methods: https://scikit-learn.org/stable/modules/ensemble.html
- XGBoost 官方文件: https://xgboost.readthedocs.io/
- LightGBM 官方文件: https://lightgbm.readthedocs.io/
