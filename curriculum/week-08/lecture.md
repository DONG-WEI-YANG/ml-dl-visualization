# 第 8 週：特徵重要度與 Shapley 示意
# Week 8: Feature Importance & SHAP Values

## 學習目標 Learning Objectives
1. 理解可解釋性 AI (Explainable AI, XAI) 的重要性與應用場景
2. 掌握模型內建特徵重要度 (Built-in Feature Importance) 的計算方式
3. 了解排列重要度 (Permutation Importance) 的原理與優勢
4. 理解 SHAP (SHapley Additive exPlanations) 的理論基礎與博弈論背景
5. 能解讀 SHAP 蜂群圖 (Beeswarm)、力圖 (Force) 與依賴圖 (Dependence)
6. 區分局部解釋 (Local Explanation) 與全域解釋 (Global Explanation)
7. 認識 LIME 方法並理解模型透明度 vs 預測能力的取捨

---

## 1. 可解釋性 AI 的重要性 Why Explainable AI Matters

### 1.1 什麼是 XAI？ What is XAI?

可解釋性 AI (Explainable AI, XAI) 是指讓人類能夠理解、信任並有效管理 AI 系統決策過程的技術與方法。

> "如果你無法解釋模型為何做出某個決策，你就不應該信任它。"
> "If you can't explain why a model made a decision, you shouldn't trust it."

### 1.2 為什麼需要可解釋性？ Why Do We Need Explainability?

| 面向 | 說明 | 範例 |
|------|------|------|
| 法規遵循 Regulatory Compliance | 許多法規要求模型決策可解釋 | 歐盟 GDPR 的「解釋權」(Right to Explanation) |
| 信任建立 Trust Building | 使用者需要理解模型行為 | 醫療診斷、信貸審核 |
| 除錯偵錯 Debugging | 找出模型錯誤的原因 | 模型依賴了不相關的特徵 |
| 領域知識驗證 Domain Validation | 確認模型學到合理的規律 | 特徵方向是否符合專家期待 |
| 公平性檢驗 Fairness Auditing | 偵測模型是否存在偏見 | 種族、性別等敏感特徵的影響 |

### 1.3 模型可解釋性的光譜 The Interpretability Spectrum

```
高度可解釋 ←————————————————————————→ 低可解釋性
(High Interpretability)                    (Low Interpretability)

線性回歸    決策樹    隨機森林    GBDT    神經網路    深度學習
Linear Reg  DTree    RF         GBDT    NN         Deep Learning

↑ 白箱模型 White-box                     黑箱模型 Black-box ↑
```

**白箱模型 (White-box Models):** 內部機制可直接理解，如線性回歸的係數。

**黑箱模型 (Black-box Models):** 內部結構複雜，需要額外的工具來解釋，如深度神經網路。

### 1.4 XAI 方法分類 XAI Method Categories

| 分類方式 | 類別 | 說明 |
|----------|------|------|
| 依適用範圍 | 模型不可知 (Model-Agnostic) | 適用於任何模型（SHAP, LIME） |
| | 模型專屬 (Model-Specific) | 僅適用特定模型（樹模型重要度） |
| 依解釋層次 | 全域解釋 (Global) | 整體模型行為 |
| | 局部解釋 (Local) | 單一預測的解釋 |
| 依時機 | 事前 (Ante-hoc) | 設計時即可解釋（線性模型） |
| | 事後 (Post-hoc) | 訓練後再分析（SHAP, LIME） |

---

## 2. 模型內建特徵重要度 Built-in Feature Importance

### 2.1 基於不純度的重要度 Impurity-based Importance

決策樹與集成模型（隨機森林 Random Forest、GBDT）在訓練過程中，會記錄每個特徵在分裂節點 (Split Node) 時所帶來的不純度下降量 (Impurity Decrease)。

**計算方式：**
- 對於分類問題：使用基尼不純度 (Gini Impurity) 或資訊增益 (Information Gain)
- 對於回歸問題：使用均方誤差下降量 (MSE Reduction)

```
特徵重要度 = Σ (該特徵在所有節點的加權不純度下降)
Feature Importance = Σ (Weighted Impurity Decrease at all nodes using the feature)
```

**在 scikit-learn 中：**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 取得特徵重要度
importances = model.feature_importances_
```

### 2.2 基於不純度重要度的限制 Limitations

1. **偏向高基數特徵 (Bias toward High-Cardinality Features):** 具有多個不同取值的特徵容易獲得較高的重要度
2. **不考慮特徵間的交互作用 (Ignores Feature Interactions):** 每次只看單一特徵的貢獻
3. **訓練資料的偏差 (Training Data Bias):** 基於訓練集計算，可能在測試集上表現不同
4. **相關特徵的稀釋 (Dilution of Correlated Features):** 高度相關的特徵會分攤重要度

---

## 3. 排列重要度 Permutation Importance

### 3.1 核心概念 Core Idea

排列重要度的思路非常直覺：**如果某個特徵很重要，打亂 (Shuffle/Permute) 它的值後，模型的效能應該會明顯下降。**

### 3.2 演算法步驟 Algorithm Steps

```
1. 在測試集上計算基準效能 (Baseline Performance)
2. 對每個特徵 f：
   a. 隨機打亂特徵 f 的值（保持其他特徵不變）
   b. 用打亂後的資料計算模型效能
   c. 重要度 = 基準效能 - 打亂後效能
   d. 重複多次取平均（減少隨機性）
3. 依重要度排序
```

### 3.3 數學表示 Mathematical Formulation

設 $s$ 為基準分數 (Baseline Score)，$s_{f,k}$ 為第 $k$ 次打亂特徵 $f$ 後的分數：

$$PI_f = s - \frac{1}{K} \sum_{k=1}^{K} s_{f,k}$$

### 3.4 排列重要度的優勢 Advantages

| 優勢 | 說明 |
|------|------|
| 模型不可知 (Model-Agnostic) | 適用於任何模型 |
| 基於測試集 (Test-set Based) | 反映真實的泛化能力 |
| 直觀易懂 (Intuitive) | 概念簡單、容易解釋 |
| 考慮交互作用 (Captures Interactions) | 間接反映特徵間的交互效果 |

### 3.5 排列重要度的注意事項 Caveats

- **相關特徵 (Correlated Features):** 打亂一個特徵可能產生不合理的資料組合，導致重要度低估
- **計算成本 (Computational Cost):** 需要對每個特徵多次重新評估模型
- **隨機性 (Randomness):** 結果可能因隨機打亂而有變異，建議多次重複

**在 scikit-learn 中：**
```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test,
                                 n_repeats=30, random_state=42)
importances = result.importances_mean
```

---

## 4. SHAP 理論基礎 SHAP: Theoretical Foundation

### 4.1 Shapley 值的博弈論背景 Game Theory Background

SHAP 的核心建立在合作博弈論 (Cooperative Game Theory) 中的 **Shapley 值 (Shapley Value)** 之上，由 Lloyd Shapley 於 1953 年提出（2012 年獲諾貝爾經濟學獎）。

**博弈論類比 Game Theory Analogy：**

想像一場合作遊戲：
- **玩家 (Players)** = 特徵 (Features)
- **聯盟 (Coalition)** = 特徵的子集合 (Subset of Features)
- **報酬 (Payout)** = 模型預測值 (Model Prediction)

問題：**每個玩家（特徵）對最終報酬（預測）的公平貢獻是多少？**

### 4.2 Shapley 值的計算 Computing Shapley Values

特徵 $i$ 的 Shapley 值定義為：

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \cdot (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

其中：
- $N$ = 所有特徵的集合
- $S$ = 不包含特徵 $i$ 的子集
- $v(S)$ = 僅使用特徵子集 $S$ 時的預測值
- $v(S \cup \{i\}) - v(S)$ = 加入特徵 $i$ 後的邊際貢獻 (Marginal Contribution)

**直覺解釋：** Shapley 值是特徵 $i$ 在所有可能的特徵加入順序中的平均邊際貢獻。

### 4.3 Shapley 值的四大公理 Four Axioms

Shapley 值是唯一滿足以下四個公理的分配方式：

| 公理 | 英文 | 說明 |
|------|------|------|
| 效率性 | Efficiency | 所有特徵的 Shapley 值之和 = 預測值 - 基準值 |
| 對稱性 | Symmetry | 若兩特徵的邊際貢獻在所有聯盟中相同，則 Shapley 值相等 |
| 虛擬性 | Dummy/Null Player | 不影響任何聯盟預測的特徵，Shapley 值為 0 |
| 可加性 | Additivity | 多個遊戲的 Shapley 值等於個別遊戲 Shapley 值的和 |

### 4.4 從 Shapley 到 SHAP

Lundberg & Lee (2017) 將 Shapley 值應用到機器學習模型解釋上，提出 SHAP 框架：

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

其中：
- $f(x)$ = 模型對樣本 $x$ 的預測
- $\phi_0$ = 基準值 (Base Value)，通常為訓練集的平均預測
- $\phi_i$ = 特徵 $i$ 的 SHAP 值

**關鍵特性：每個預測都被分解為基準值加上各特徵的貢獻。**

### 4.5 SHAP 的計算方法 SHAP Computation Methods

精確計算 Shapley 值的複雜度為 $O(2^M)$（$M$ 為特徵數），因此需要近似方法：

| 方法 | 英文 | 適用模型 | 複雜度 |
|------|------|----------|--------|
| TreeSHAP | Tree SHAP | 樹模型 (RF, GBDT, XGBoost) | $O(TLD^2)$ |
| KernelSHAP | Kernel SHAP | 任意模型 (Model-Agnostic) | 較慢，需採樣 |
| DeepSHAP | Deep SHAP | 深度學習模型 | 基於 DeepLIFT |
| LinearSHAP | Linear SHAP | 線性模型 | 精確且快速 |

---

## 5. SHAP 圖表解讀 Interpreting SHAP Plots

### 5.1 蜂群圖 Beeswarm Plot (Summary Plot)

蜂群圖是 SHAP 最常用的全域解釋圖表，顯示所有特徵對所有樣本的 SHAP 值分布。

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SHAP Beeswarm Plot 示意圖">
  <defs>
    <linearGradient id="shapBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#2563eb"/>
      <stop offset="50%" stop-color="#e5e7eb"/>
      <stop offset="100%" stop-color="#ef4444"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="640" height="280" fill="#ffffff"/>
  <!-- Axes -->
  <line x1="160" y1="40" x2="160" y2="240" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 4"/>
  <text x="160" y="258" text-anchor="middle" font-size="11" fill="#6b7280">SHAP = 0 (基準)</text>
  <line x1="60" y1="240" x2="600" y2="240" stroke="#374151" stroke-width="1"/>
  <text x="80" y="272" font-size="11" fill="#6b7280">← 壓低預測 (負值)</text>
  <text x="460" y="272" font-size="11" fill="#6b7280">推高預測 (正值) →</text>
  <!-- Feature labels (Y axis) -->
  <text x="150" y="72" text-anchor="end" font-size="12" fill="#111827" font-weight="600">信用評分</text>
  <text x="150" y="114" text-anchor="end" font-size="12" fill="#111827" font-weight="600">年收入</text>
  <text x="150" y="156" text-anchor="end" font-size="12" fill="#111827" font-weight="600">負債比</text>
  <text x="150" y="198" text-anchor="end" font-size="12" fill="#111827" font-weight="600">年齡</text>
  <!-- Row 1: 信用評分 — high values push prediction up (red clustered on right) -->
  <g>
    <circle cx="220" cy="72" r="4" fill="#60a5fa"/><circle cx="235" cy="68" r="4" fill="#93c5fd"/>
    <circle cx="248" cy="74" r="4" fill="#93c5fd"/><circle cx="270" cy="70" r="4" fill="#cbd5e1"/>
    <circle cx="320" cy="72" r="4" fill="#e5e7eb"/><circle cx="360" cy="68" r="4" fill="#fca5a5"/>
    <circle cx="400" cy="74" r="4" fill="#f87171"/><circle cx="430" cy="70" r="4" fill="#ef4444"/>
    <circle cx="470" cy="76" r="4" fill="#ef4444"/><circle cx="505" cy="72" r="4" fill="#dc2626"/>
    <circle cx="540" cy="68" r="4" fill="#b91c1c"/>
  </g>
  <!-- Row 2: 年收入 — similar positive correlation, wider spread -->
  <g>
    <circle cx="200" cy="114" r="4" fill="#3b82f6"/><circle cx="225" cy="110" r="4" fill="#60a5fa"/>
    <circle cx="255" cy="118" r="4" fill="#93c5fd"/><circle cx="290" cy="114" r="4" fill="#cbd5e1"/>
    <circle cx="330" cy="110" r="4" fill="#e5e7eb"/><circle cx="370" cy="116" r="4" fill="#fca5a5"/>
    <circle cx="410" cy="114" r="4" fill="#f87171"/><circle cx="445" cy="112" r="4" fill="#ef4444"/>
    <circle cx="480" cy="118" r="4" fill="#dc2626"/><circle cx="515" cy="114" r="4" fill="#b91c1c"/>
  </g>
  <!-- Row 3: 負債比 — reverse: high values push prediction DOWN (red on left) -->
  <g>
    <circle cx="100" cy="156" r="4" fill="#b91c1c"/><circle cx="125" cy="152" r="4" fill="#dc2626"/>
    <circle cx="145" cy="158" r="4" fill="#ef4444"/><circle cx="180" cy="154" r="4" fill="#f87171"/>
    <circle cx="215" cy="156" r="4" fill="#fca5a5"/><circle cx="260" cy="152" r="4" fill="#e5e7eb"/>
    <circle cx="310" cy="158" r="4" fill="#cbd5e1"/><circle cx="360" cy="154" r="4" fill="#93c5fd"/>
    <circle cx="410" cy="156" r="4" fill="#60a5fa"/><circle cx="460" cy="152" r="4" fill="#3b82f6"/>
  </g>
  <!-- Row 4: 年齡 — tight cluster, weak effect -->
  <g>
    <circle cx="140" cy="198" r="4" fill="#60a5fa"/><circle cx="155" cy="194" r="4" fill="#93c5fd"/>
    <circle cx="170" cy="200" r="4" fill="#cbd5e1"/><circle cx="185" cy="196" r="4" fill="#e5e7eb"/>
    <circle cx="200" cy="198" r="4" fill="#fca5a5"/><circle cx="215" cy="194" r="4" fill="#f87171"/>
    <circle cx="230" cy="200" r="4" fill="#ef4444"/>
  </g>
  <!-- Color legend (right) -->
  <rect x="560" y="60" width="14" height="180" fill="url(#shapBar)" stroke="#d1d5db"/>
  <text x="582" y="66" font-size="10" fill="#111827">高值</text>
  <text x="582" y="154" font-size="10" fill="#6b7280">特徵值</text>
  <text x="582" y="244" font-size="10" fill="#111827">低值</text>
</svg>
<figcaption>示意圖：蜂群圖。Y 軸依重要度排列特徵；X 軸為 SHAP 值；顏色代表該樣本在該特徵的實際值（紅高藍低）。</figcaption>
</figure>
```

**解讀方式：**
```
Y 軸：特徵名稱（依重要度排序）
X 軸：SHAP 值（正值 → 推高預測，負值 → 壓低預測）
顏色：特徵值的高低（紅色 = 高值，藍色 = 低值）
每個點：一個樣本
```

**範例解讀：**
- 若某特徵的紅點集中在 SHAP > 0 的區域 → 該特徵值越高，預測值越高（正相關）
- 若某特徵的點散布範圍很廣 → 該特徵對預測的影響很大
- 若某特徵的紅藍混雜 → 該特徵與預測值的關係較複雜（可能有非線性或交互作用）

### 5.2 力圖 Force Plot

力圖用於**局部解釋 (Local Explanation)**，展示單一樣本的預測如何由基準值推移到最終預測值。

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SHAP Force Plot 示意圖">
  <rect x="0" y="0" width="640" height="200" fill="#ffffff"/>
  <!-- Number line axis -->
  <line x1="40" y1="120" x2="600" y2="120" stroke="#374151" stroke-width="1.5"/>
  <!-- Base value marker -->
  <line x1="260" y1="60" x2="260" y2="140" stroke="#6b7280" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="260" y="52" text-anchor="middle" font-size="11" fill="#6b7280">基準值 base value = 0.35</text>
  <!-- Final prediction marker -->
  <line x1="480" y1="60" x2="480" y2="140" stroke="#111827" stroke-width="2"/>
  <text x="480" y="52" text-anchor="middle" font-size="11" fill="#111827" font-weight="600">最終預測 = 0.78</text>
  <!-- Red pushing-up arrows (left to right from base) -->
  <polygon points="260,100 340,100 340,92 360,108 340,124 340,116 260,116" fill="#ef4444" stroke="#b91c1c" stroke-width="1"/>
  <text x="305" y="106" text-anchor="middle" font-size="11" fill="#ffffff" font-weight="600">信用=780</text>
  <text x="305" y="140" text-anchor="middle" font-size="10" fill="#b91c1c">+0.18</text>
  <polygon points="360,100 420,100 420,92 440,108 420,124 420,116 360,116" fill="#f87171" stroke="#b91c1c" stroke-width="1"/>
  <text x="395" y="106" text-anchor="middle" font-size="11" fill="#ffffff" font-weight="600">收入高</text>
  <text x="395" y="140" text-anchor="middle" font-size="10" fill="#b91c1c">+0.14</text>
  <polygon points="440,100 470,100 470,92 480,108 470,124 470,116 440,116" fill="#fca5a5" stroke="#b91c1c" stroke-width="1"/>
  <text x="457" y="140" text-anchor="middle" font-size="10" fill="#b91c1c">+0.06</text>
  <!-- Blue pushing-down arrow (right to left past base) -->
  <polygon points="260,100 220,100 220,92 200,108 220,124 220,116 260,116" fill="#60a5fa" stroke="#1e40af" stroke-width="1"/>
  <text x="238" y="106" text-anchor="middle" font-size="11" fill="#ffffff" font-weight="600">負債比高</text>
  <text x="238" y="140" text-anchor="middle" font-size="10" fill="#1e40af">-0.05</text>
  <!-- Labels on axis -->
  <text x="40" y="160" font-size="10" fill="#6b7280">0.0</text>
  <text x="600" y="160" text-anchor="end" font-size="10" fill="#6b7280">1.0</text>
  <text x="320" y="180" text-anchor="middle" font-size="11" fill="#6b7280">模型輸出概率 P(核准貸款)</text>
</svg>
<figcaption>示意圖：力圖。紅箭頭代表推高預測的特徵（箭頭長度 = SHAP 值大小），藍箭頭代表壓低預測的特徵，所有箭頭合併使預測從 base value 0.35 推移到 0.78。</figcaption>
</figure>
```

**解讀方式：**
```
基準值 (Base Value)：模型的平均預測
紅色箭頭：推高預測的特徵
藍色箭頭：壓低預測的特徵
箭頭長度：該特徵的 SHAP 值大小
最終預測：所有箭頭推移後的結果
```

**適用情境：**
- 向客戶解釋「為什麼你的貸款被拒絕」
- 向醫生解釋「為什麼模型判斷這位病人為高風險」

### 5.3 依賴圖 Dependence Plot

依賴圖顯示**單一特徵的值**與**其 SHAP 值**之間的關係，並可疊加第二個特徵的交互作用。

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SHAP Dependence Plot 示意圖">
  <defs>
    <linearGradient id="depBar" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stop-color="#2563eb"/>
      <stop offset="100%" stop-color="#ef4444"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="640" height="320" fill="#ffffff"/>
  <!-- Plot area -->
  <rect x="80" y="40" width="460" height="220" fill="#fafafa" stroke="#e5e7eb"/>
  <!-- Axes -->
  <line x1="80" y1="260" x2="540" y2="260" stroke="#374151" stroke-width="1"/>
  <line x1="80" y1="40" x2="80" y2="260" stroke="#374151" stroke-width="1"/>
  <!-- Zero reference -->
  <line x1="80" y1="160" x2="540" y2="160" stroke="#9ca3af" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="72" y="164" text-anchor="end" font-size="10" fill="#6b7280">0</text>
  <!-- Axis labels -->
  <text x="310" y="296" text-anchor="middle" font-size="12" fill="#111827">特徵值：年齡</text>
  <text x="32" y="150" font-size="12" fill="#111827" transform="rotate(-90 32 150)">SHAP 值 (對預測的貢獻)</text>
  <!-- X axis ticks -->
  <text x="120" y="276" text-anchor="middle" font-size="10" fill="#6b7280">20</text>
  <text x="210" y="276" text-anchor="middle" font-size="10" fill="#6b7280">30</text>
  <text x="310" y="276" text-anchor="middle" font-size="10" fill="#6b7280">40</text>
  <text x="410" y="276" text-anchor="middle" font-size="10" fill="#6b7280">50</text>
  <text x="510" y="276" text-anchor="middle" font-size="10" fill="#6b7280">60</text>
  <!-- Y axis ticks -->
  <text x="72" y="64" text-anchor="end" font-size="10" fill="#6b7280">+0.3</text>
  <text x="72" y="114" text-anchor="end" font-size="10" fill="#6b7280">+0.15</text>
  <text x="72" y="214" text-anchor="end" font-size="10" fill="#6b7280">-0.15</text>
  <text x="72" y="258" text-anchor="end" font-size="10" fill="#6b7280">-0.3</text>
  <!-- Non-linear curve shown via dense scatter + spline -->
  <path d="M 100 220 Q 180 210 230 180 T 340 130 T 450 80 T 530 70" fill="none" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 2"/>
  <!-- Scatter points — X = age, Y = SHAP, color = income (interaction feature) -->
  <g>
    <circle cx="105" cy="230" r="3.5" fill="#2563eb"/><circle cx="118" cy="215" r="3.5" fill="#3b82f6"/>
    <circle cx="135" cy="225" r="3.5" fill="#60a5fa"/><circle cx="150" cy="205" r="3.5" fill="#2563eb"/>
    <circle cx="170" cy="210" r="3.5" fill="#93c5fd"/><circle cx="185" cy="195" r="3.5" fill="#3b82f6"/>
    <circle cx="200" cy="200" r="3.5" fill="#cbd5e1"/><circle cx="220" cy="180" r="3.5" fill="#60a5fa"/>
    <circle cx="240" cy="175" r="3.5" fill="#93c5fd"/><circle cx="260" cy="165" r="3.5" fill="#cbd5e1"/>
    <circle cx="280" cy="155" r="3.5" fill="#e5e7eb"/><circle cx="300" cy="148" r="3.5" fill="#fca5a5"/>
    <circle cx="320" cy="140" r="3.5" fill="#e5e7eb"/><circle cx="340" cy="130" r="3.5" fill="#fca5a5"/>
    <circle cx="360" cy="118" r="3.5" fill="#f87171"/><circle cx="380" cy="108" r="3.5" fill="#fca5a5"/>
    <circle cx="400" cy="98" r="3.5" fill="#ef4444"/><circle cx="420" cy="88" r="3.5" fill="#f87171"/>
    <circle cx="440" cy="92" r="3.5" fill="#dc2626"/><circle cx="460" cy="78" r="3.5" fill="#ef4444"/>
    <circle cx="480" cy="75" r="3.5" fill="#b91c1c"/><circle cx="500" cy="70" r="3.5" fill="#dc2626"/>
    <circle cx="520" cy="68" r="3.5" fill="#b91c1c"/>
  </g>
  <!-- Interaction color legend -->
  <rect x="560" y="60" width="14" height="160" fill="url(#depBar)" stroke="#d1d5db"/>
  <text x="582" y="66" font-size="10" fill="#111827">高</text>
  <text x="582" y="144" font-size="10" fill="#6b7280">年收入</text>
  <text x="582" y="228" font-size="10" fill="#111827">低</text>
</svg>
<figcaption>示意圖：依賴圖。X 軸為「年齡」的實際值，Y 軸為該樣本「年齡」特徵的 SHAP 值；顏色顯示「年收入」作為交互特徵 — 同年齡下收入高的點（紅）SHAP 較高，代表存在年齡 × 收入交互作用。</figcaption>
</figure>
```

**解讀方式：**
```
X 軸：特徵的實際值
Y 軸：該特徵的 SHAP 值
顏色：交互特徵的值（自動選擇最強交互特徵）
```

**可觀察到的模式：**
- 線性關係：點排列成直線 → 特徵與預測值呈線性關係
- 非線性關係：曲線或分段函數 → 特徵效果在不同區段不同
- 交互作用：同一 X 值的點因顏色不同而有不同 SHAP 值 → 存在交互作用

### 5.4 其他常用圖表 Other Common Plots

| 圖表 | 英文 | 功能 |
|------|------|------|
| 瀑布圖 | Waterfall Plot | 單一樣本的特徵貢獻堆疊 |
| 長條圖 | Bar Plot | 全域特徵重要度（SHAP 值絕對值的平均） |
| 熱力圖 | Heatmap | 多樣本的 SHAP 值矩陣 |
| 群集力圖 | Clustered Force Plot | 多個力圖堆疊，觀察群體模式 |

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SHAP Waterfall Plot 示意圖">
  <rect x="0" y="0" width="640" height="360" fill="#ffffff"/>
  <!-- Base value line -->
  <line x1="200" y1="300" x2="560" y2="300" stroke="#6b7280" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="190" y="304" text-anchor="end" font-size="11" fill="#6b7280">E[f(x)]=0.35</text>
  <!-- Final prediction line -->
  <line x1="200" y1="60" x2="560" y2="60" stroke="#111827" stroke-width="1.5"/>
  <text x="190" y="64" text-anchor="end" font-size="11" fill="#111827" font-weight="600">f(x)=0.78</text>
  <!-- Step rectangles — from bottom (base) to top (final), stacked cumulatively -->
  <!-- Each row: feature name (left), bar, SHAP value (right) -->
  <!-- Row 1: 信用評分 +0.18 -->
  <text x="190" y="272" text-anchor="end" font-size="11" fill="#111827">信用評分=780</text>
  <rect x="200" y="252" width="110" height="22" fill="#ef4444" stroke="#b91c1c"/>
  <text x="318" y="268" font-size="11" fill="#b91c1c" font-weight="600">+0.18</text>
  <line x1="310" y1="252" x2="310" y2="300" stroke="#d1d5db" stroke-width="1" stroke-dasharray="2 2"/>
  <!-- Row 2: 年收入 +0.14 (continue from 310) -->
  <text x="190" y="224" text-anchor="end" font-size="11" fill="#111827">年收入=高</text>
  <rect x="310" y="204" width="86" height="22" fill="#f87171" stroke="#b91c1c"/>
  <text x="404" y="220" font-size="11" fill="#b91c1c" font-weight="600">+0.14</text>
  <line x1="396" y1="204" x2="396" y2="252" stroke="#d1d5db" stroke-width="1" stroke-dasharray="2 2"/>
  <!-- Row 3: 工作年資 +0.08 -->
  <text x="190" y="176" text-anchor="end" font-size="11" fill="#111827">工作年資=10</text>
  <rect x="396" y="156" width="50" height="22" fill="#fca5a5" stroke="#b91c1c"/>
  <text x="454" y="172" font-size="11" fill="#b91c1c" font-weight="600">+0.08</text>
  <line x1="446" y1="156" x2="446" y2="204" stroke="#d1d5db" stroke-width="1" stroke-dasharray="2 2"/>
  <!-- Row 4: 負債比 -0.05 (subtract, bar goes LEFT) -->
  <text x="190" y="128" text-anchor="end" font-size="11" fill="#111827">負債比=0.4</text>
  <rect x="416" y="108" width="30" height="22" fill="#60a5fa" stroke="#1e40af"/>
  <text x="412" y="124" text-anchor="end" font-size="11" fill="#1e40af" font-weight="600">-0.05</text>
  <line x1="416" y1="108" x2="416" y2="156" stroke="#d1d5db" stroke-width="1" stroke-dasharray="2 2"/>
  <!-- Row 5: 年齡 +0.08 -->
  <text x="190" y="80" text-anchor="end" font-size="11" fill="#111827">年齡=42</text>
  <rect x="416" y="60" width="48" height="22" fill="#fca5a5" stroke="#b91c1c"/>
  <text x="472" y="76" font-size="11" fill="#b91c1c" font-weight="600">+0.08</text>
  <!-- X-axis -->
  <line x1="200" y1="330" x2="560" y2="330" stroke="#374151" stroke-width="1"/>
  <text x="200" y="348" font-size="10" fill="#6b7280">0.3</text>
  <text x="310" y="348" text-anchor="middle" font-size="10" fill="#6b7280">0.5</text>
  <text x="446" y="348" text-anchor="middle" font-size="10" fill="#6b7280">0.7</text>
  <text x="560" y="348" text-anchor="end" font-size="10" fill="#6b7280">0.9</text>
  <text x="380" y="20" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">瀑布圖：從 E[f(x)] 逐步累積到 f(x)</text>
</svg>
<figcaption>示意圖：瀑布圖。從下往上依 SHAP 值絕對值排序堆疊：紅色條往右延伸（推高預測）、藍色條往左延伸（壓低預測），最終抵達該樣本的實際預測值 f(x)=0.78。</figcaption>
</figure>
```

---

## 6. 局部解釋 vs 全域解釋 Local vs Global Explanation

### 6.1 定義與區別 Definitions

| 面向 | 局部解釋 Local | 全域解釋 Global |
|------|----------------|-----------------|
| 範圍 | 解釋單一預測 | 解釋整體模型行為 |
| 問題 | 「為什麼模型對這個客戶給出這個分數？」 | 「模型整體上最看重哪些特徵？」 |
| SHAP 圖表 | 力圖、瀑布圖 | 蜂群圖、長條圖 |
| 適用場景 | 個案審查、客訴回覆 | 模型審計、特徵選擇 |

### 6.2 從局部到全域 From Local to Global

SHAP 的優勢在於局部解釋可以**聚合 (Aggregate)** 成全域解釋：

```
局部 SHAP 值 → 取絕對值的平均 → 全域特徵重要度
Local SHAP  → Mean(|SHAP|)     → Global Feature Importance
```

這代表全域重要度是建立在堅實的局部解釋之上，而非獨立的估計。

### 6.3 實務建議 Practical Advice

- **開發階段：** 先看全域解釋（蜂群圖），確認模型行為合理
- **上線審查：** 用局部解釋（力圖）抽查高風險預測
- **異常偵測：** 比較異常樣本與正常樣本的 SHAP 值差異
- **報告撰寫：** 全域圖表放總結，局部圖表放附錄或個案說明

---

## 7. LIME 簡介 Introduction to LIME

### 7.1 什麼是 LIME？

LIME (Local Interpretable Model-agnostic Explanations) 是另一個重要的局部解釋方法，由 Ribeiro et al. (2016) 提出。

### 7.2 LIME 的原理 How LIME Works

```
1. 選擇一個要解釋的樣本 x
2. 在 x 附近生成擾動樣本 (Perturbed Samples)
3. 用原始模型對擾動樣本進行預測
4. 在擾動樣本上訓練一個簡單的可解釋模型（如線性回歸）
5. 用可解釋模型的係數作為特徵重要度
```

**核心思想：** 即使全域模型很複雜，在局部區域 (Local Region) 可以用簡單模型近似。

### 7.3 SHAP vs LIME 比較

| 面向 | SHAP | LIME |
|------|------|------|
| 理論基礎 | 博弈論 (Shapley Value) | 局部線性近似 |
| 一致性 | 有理論保證（四大公理） | 無嚴格理論保證 |
| 穩定性 | 較穩定（確定性演算法） | 較不穩定（隨機擾動） |
| 全域解釋 | 可聚合為全域 | 僅限局部 |
| 計算速度 | TreeSHAP 很快；KernelSHAP 較慢 | 較快 |
| 易用性 | 圖表豐富、社群活躍 | 簡單直覺 |

### 7.4 何時選用哪種方法？ When to Use Which?

- **需要理論保證與一致性：** SHAP
- **需要快速的局部解釋：** LIME
- **樹模型（RF, XGBoost, LightGBM）：** TreeSHAP（速度快且精確）
- **深度學習模型：** DeepSHAP 或 Integrated Gradients
- **向非技術人員報告：** LIME 的線性模型較易理解

---

## 8. 模型透明度 vs 預測能力的取捨 Transparency vs Performance Trade-off

### 8.1 經典兩難 The Classic Dilemma

通常更複雜的模型有更好的預測能力，但可解釋性更低：

```
準確率 (Accuracy)
  ↑
  │          * 深度學習
  │        * GBDT/XGBoost
  │      * 隨機森林
  │    * SVM
  │  * 決策樹
  │* 線性回歸
  └──────────────────→ 可解釋性 (Interpretability)
```

### 8.2 打破兩難的策略 Strategies to Break the Dilemma

1. **事後解釋 (Post-hoc Explanation):**
   - 使用複雜模型 + SHAP/LIME 進行解釋
   - 不犧牲預測能力，額外加上解釋層

2. **可解釋的強模型 (Interpretable Yet Powerful Models):**
   - EBM (Explainable Boosting Machine)：可解釋且效能接近 GBDT
   - GAM (Generalized Additive Models)：可視化各特徵的邊際效果

3. **模型蒸餾 (Model Distillation):**
   - 用複雜模型訓練簡單模型
   - 簡單模型近似複雜模型的行為，同時保持可解釋性

4. **分層策略 (Layered Approach):**
   - 簡單案例用可解釋模型處理
   - 複雜案例用黑箱模型 + 事後解釋

### 8.3 法規與產業要求 Regulatory & Industry Requirements

| 產業 | 要求 | 建議方法 |
|------|------|----------|
| 金融 Finance | 信貸決策需可解釋 | Logistic Regression + SHAP |
| 醫療 Healthcare | 診斷建議需透明 | 決策樹 / EBM + SHAP |
| 保險 Insurance | 定價因子需明確 | GAM + SHAP |
| 一般商業 General | 較寬鬆 | 自由選擇 + 適當文件 |

### 8.4 實務決策框架 Practical Decision Framework

```
1. 預測錯誤的代價高嗎？ Is the cost of error high?
   ├── 是 → 優先考慮預測能力，搭配 SHAP 解釋
   └── 否 → 優先考慮可解釋模型

2. 有法規要求嗎？ Are there regulatory requirements?
   ├── 是 → 必須提供解釋（SHAP / LIME / 白箱模型）
   └── 否 → 依據業務需求決定

3. 使用者需要理解決策嗎？ Do users need to understand decisions?
   ├── 是 → 提供局部解釋（力圖、瀑布圖）
   └── 否 → 著重全域解釋（蜂群圖）用於開發團隊
```

---

## 關鍵詞彙 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 可解釋性 AI | Explainable AI (XAI) | 讓 AI 決策可被人類理解的技術 |
| 特徵重要度 | Feature Importance | 衡量特徵對模型預測的貢獻程度 |
| 排列重要度 | Permutation Importance | 透過打亂特徵值衡量重要度的方法 |
| SHAP 值 | SHAP Values | 基於 Shapley 值的模型解釋方法 |
| Shapley 值 | Shapley Value | 博弈論中公平分配貢獻的方法 |
| 蜂群圖 | Beeswarm Plot | 顯示全部特徵 SHAP 值分布的圖表 |
| 力圖 | Force Plot | 顯示單一預測的特徵貢獻圖表 |
| 依賴圖 | Dependence Plot | 顯示特徵值與 SHAP 值關係的圖表 |
| 局部解釋 | Local Explanation | 針對單一預測的解釋 |
| 全域解釋 | Global Explanation | 針對整體模型行為的解釋 |
| LIME | LIME | 局部可解釋的模型不可知方法 |
| 白箱模型 | White-box Model | 內部機制可直接理解的模型 |
| 黑箱模型 | Black-box Model | 內部結構複雜需額外解釋的模型 |
| 邊際貢獻 | Marginal Contribution | 加入某特徵後的效能增量 |
| 模型蒸餾 | Model Distillation | 用複雜模型訓練簡單模型的技術 |

---

## 延伸閱讀 Further Reading

- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." (NIPS)
- Molnar, C. (2022). *Interpretable Machine Learning.* https://christophm.github.io/interpretable-ml-book/
- Ribeiro, M. T., et al. (2016). "Why Should I Trust You? Explaining the Predictions of Any Classifier." (KDD)
- SHAP 官方文件：https://shap.readthedocs.io/
- scikit-learn Permutation Importance：https://scikit-learn.org/stable/modules/permutation_importance.html
- Shapley, L. S. (1953). "A Value for n-Person Games." *Contributions to the Theory of Games.*
