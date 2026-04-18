# 第 11 週：神經網路基礎 — 激活函數、正則化與批次正規化可視化
# Week 11: Neural Network Basics — Activation, Regularization & BatchNorm Visualization

## 學習目標 Learning Objectives
1. 理解感知器 (Perceptron) 模型與生物神經元的類比
2. 掌握多層感知器 (Multi-Layer Perceptron, MLP) 與前饋網路 (Feedforward Network) 結構
3. 深入了解主流激活函數 (Activation Functions) 的特性、導數與適用場景
4. 直覺理解反向傳播 (Backpropagation) 與梯度消失/爆炸問題
5. 掌握正則化技術：Dropout、L1/L2 正則化 (Regularization)、早停 (Early Stopping)
6. 理解批次正規化 (Batch Normalization) 的原理與訓練效果
7. 認識權重初始化 (Weight Initialization) 策略：Xavier、He
8. 入門 PyTorch 基礎：Tensor、autograd、nn.Module

**先備知識 Prerequisites:** Week 1-10（Python 環境、梯度下降、損失函數、過擬合概念）、基礎線性代數與微積分

---

## 1. 從生物神經元到人工神經元 From Biological to Artificial Neurons

### 1.1 生物神經元 Biological Neuron

人腦約有 860 億個神經元 (Neurons)，每個神經元透過突觸 (Synapses) 與其他神經元連接。一個神經元的基本運作流程：

1. **樹突 (Dendrites)** 接收來自其他神經元的信號
2. **細胞體 (Cell Body / Soma)** 將信號進行整合
3. 當信號總和超過閾值 (Threshold)，神經元**激發 (Fire)**
4. **軸突 (Axon)** 將信號傳遞給下游神經元

> "All-or-none" 原則：神經元要麼激發、要麼沉默，這啟發了早期的人工神經元模型。

### 1.2 感知器 Perceptron（Rosenblatt, 1958）

感知器是最簡單的人工神經元模型，直接類比生物神經元：

| 生物神經元 | 感知器 | 角色 |
|-----------|--------|------|
| 樹突 (Dendrites) | 輸入 (Inputs) x1, x2, ..., xn | 接收信號 |
| 突觸強度 (Synaptic Strength) | 權重 (Weights) w1, w2, ..., wn | 信號重要程度 |
| 細胞體整合 (Soma Integration) | 加權求和 z = sum(wi * xi) + b | 彙總所有輸入 |
| 閾值激發 (Threshold Firing) | 激活函數 (Activation Function) y_hat = f(z) | 決定是否激發 |
| 軸突輸出 (Axon Output) | 輸出 (Output) y_hat | 傳遞信號 |

感知器的數學表達：

```
z = w^T * x + b = sum_{i=1}^{n} w_i * x_i + b

y_hat = step(z) = 1  if z >= 0
                  0  if z < 0
```

其中 b 為偏差項 (Bias)，扮演調整閾值的角色。

### 1.3 感知器的局限 Limitations

1969 年 Minsky 和 Papert 證明：**單層感知器無法解決 XOR 問題**。XOR 是非線性可分的 (Non-linearly Separable)，這意味著一條直線無法將兩類分開。

> 這個發現導致了第一次「AI 寒冬 (AI Winter)」，但也激發了多層網路的研究。

---

## 2. 多層感知器與前饋網路 MLP & Feedforward Networks

### 2.1 多層感知器 Multi-Layer Perceptron (MLP)

為了解決非線性問題，我們將多個感知器堆疊成多層結構：

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MLP 多層感知器架構圖">
  <rect x="0" y="0" width="640" height="380" fill="#ffffff"/>
  <!-- Layer labels -->
  <text x="80" y="30" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">輸入層</text>
  <text x="80" y="46" text-anchor="middle" font-size="10" fill="#6b7280">Input</text>
  <text x="240" y="30" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">隱藏層 1</text>
  <text x="240" y="46" text-anchor="middle" font-size="10" fill="#6b7280">Hidden 1</text>
  <text x="400" y="30" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">隱藏層 2</text>
  <text x="400" y="46" text-anchor="middle" font-size="10" fill="#6b7280">Hidden 2</text>
  <text x="560" y="30" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">輸出層</text>
  <text x="560" y="46" text-anchor="middle" font-size="10" fill="#6b7280">Output</text>
  <!-- Edges (drawn first so nodes cover them) -->
  <g stroke="#cbd5e1" stroke-width="0.8" fill="none">
    <!-- Input→H1 (4×5) -->
    <line x1="80" y1="100" x2="240" y2="90"/><line x1="80" y1="100" x2="240" y2="150"/><line x1="80" y1="100" x2="240" y2="210"/><line x1="80" y1="100" x2="240" y2="270"/><line x1="80" y1="100" x2="240" y2="330"/>
    <line x1="80" y1="170" x2="240" y2="90"/><line x1="80" y1="170" x2="240" y2="150"/><line x1="80" y1="170" x2="240" y2="210"/><line x1="80" y1="170" x2="240" y2="270"/><line x1="80" y1="170" x2="240" y2="330"/>
    <line x1="80" y1="240" x2="240" y2="90"/><line x1="80" y1="240" x2="240" y2="150"/><line x1="80" y1="240" x2="240" y2="210"/><line x1="80" y1="240" x2="240" y2="270"/><line x1="80" y1="240" x2="240" y2="330"/>
    <line x1="80" y1="310" x2="240" y2="90"/><line x1="80" y1="310" x2="240" y2="150"/><line x1="80" y1="310" x2="240" y2="210"/><line x1="80" y1="310" x2="240" y2="270"/><line x1="80" y1="310" x2="240" y2="330"/>
    <!-- H1→H2 (5×4) -->
    <line x1="240" y1="90" x2="400" y2="120"/><line x1="240" y1="90" x2="400" y2="180"/><line x1="240" y1="90" x2="400" y2="240"/><line x1="240" y1="90" x2="400" y2="300"/>
    <line x1="240" y1="150" x2="400" y2="120"/><line x1="240" y1="150" x2="400" y2="180"/><line x1="240" y1="150" x2="400" y2="240"/><line x1="240" y1="150" x2="400" y2="300"/>
    <line x1="240" y1="210" x2="400" y2="120"/><line x1="240" y1="210" x2="400" y2="180"/><line x1="240" y1="210" x2="400" y2="240"/><line x1="240" y1="210" x2="400" y2="300"/>
    <line x1="240" y1="270" x2="400" y2="120"/><line x1="240" y1="270" x2="400" y2="180"/><line x1="240" y1="270" x2="400" y2="240"/><line x1="240" y1="270" x2="400" y2="300"/>
    <line x1="240" y1="330" x2="400" y2="120"/><line x1="240" y1="330" x2="400" y2="180"/><line x1="240" y1="330" x2="400" y2="240"/><line x1="240" y1="330" x2="400" y2="300"/>
    <!-- H2→Output (4×3) -->
    <line x1="400" y1="120" x2="560" y2="150"/><line x1="400" y1="120" x2="560" y2="210"/><line x1="400" y1="120" x2="560" y2="270"/>
    <line x1="400" y1="180" x2="560" y2="150"/><line x1="400" y1="180" x2="560" y2="210"/><line x1="400" y1="180" x2="560" y2="270"/>
    <line x1="400" y1="240" x2="560" y2="150"/><line x1="400" y1="240" x2="560" y2="210"/><line x1="400" y1="240" x2="560" y2="270"/>
    <line x1="400" y1="300" x2="560" y2="150"/><line x1="400" y1="300" x2="560" y2="210"/><line x1="400" y1="300" x2="560" y2="270"/>
  </g>
  <!-- Input nodes -->
  <g fill="#dbeafe" stroke="#2563eb" stroke-width="1.5">
    <circle cx="80" cy="100" r="16"/><circle cx="80" cy="170" r="16"/><circle cx="80" cy="240" r="16"/><circle cx="80" cy="310" r="16"/>
  </g>
  <g font-size="12" fill="#1e3a5f" text-anchor="middle" font-family="serif">
    <text x="80" y="104">x₁</text><text x="80" y="174">x₂</text><text x="80" y="244">x₃</text><text x="80" y="314">x₄</text>
  </g>
  <!-- Hidden 1 nodes -->
  <g fill="#fef3c7" stroke="#d97706" stroke-width="1.5">
    <circle cx="240" cy="90" r="16"/><circle cx="240" cy="150" r="16"/><circle cx="240" cy="210" r="16"/><circle cx="240" cy="270" r="16"/><circle cx="240" cy="330" r="16"/>
  </g>
  <!-- Hidden 2 nodes -->
  <g fill="#fef3c7" stroke="#d97706" stroke-width="1.5">
    <circle cx="400" cy="120" r="16"/><circle cx="400" cy="180" r="16"/><circle cx="400" cy="240" r="16"/><circle cx="400" cy="300" r="16"/>
  </g>
  <!-- Output nodes -->
  <g fill="#fee2e2" stroke="#dc2626" stroke-width="1.5">
    <circle cx="560" cy="150" r="16"/><circle cx="560" cy="210" r="16"/><circle cx="560" cy="270" r="16"/>
  </g>
  <g font-size="12" fill="#7f1d1d" text-anchor="middle" font-family="serif">
    <text x="560" y="154">y₁</text><text x="560" y="214">y₂</text><text x="560" y="274">y₃</text>
  </g>
  <!-- Depth/Width annotations -->
  <text x="320" y="368" text-anchor="middle" font-size="11" fill="#6b7280">深度 Depth = 隱藏層數 = 2　・　寬度 Width = 每層神經元數（此例各為 5, 4）</text>
</svg>
<figcaption>示意圖：四維輸入、兩層隱藏層（寬度 5 + 4）、三維輸出的多層感知器。每一層節點與下一層全連接，每條連線代表一個可學習的權重 w。</figcaption>
</figure>
```

**關鍵術語：**
- **層 (Layer)**：一組神經元的集合
- **輸入層 (Input Layer)**：接收原始資料，不做計算
- **隱藏層 (Hidden Layer)**：介於輸入與輸出之間的計算層
- **輸出層 (Output Layer)**：產生最終預測
- **深度 (Depth)**：隱藏層的層數（「深度」學習由此得名）
- **寬度 (Width)**：每層神經元的數量

### 2.2 前饋網路 Feedforward Network

前饋網路 (Feedforward Neural Network, FNN) 是最基本的神經網路架構：

- 信號**單向流動**：從輸入層 → 隱藏層 → 輸出層
- 不存在迴圈或反饋連接（與 RNN 的差異）
- MLP 是前饋網路的一種，且層與層之間**全連接 (Fully Connected)**

每一層的計算：

```
h^(l) = f( W^(l) * h^(l-1) + b^(l) )
```

其中：
- h^(l) 是第 l 層的輸出向量
- W^(l) 是第 l 層的權重矩陣 (Weight Matrix)
- b^(l) 是第 l 層的偏差向量 (Bias Vector)
- f(.) 是激活函數 (Activation Function)

### 2.3 萬能近似定理 Universal Approximation Theorem

> 一個具有至少一個隱藏層且使用非線性激活函數的前饋網路，理論上可以逼近任何連續函數（在有限區域內，精度任意高）。— Cybenko, 1989; Hornik, 1991

這意味著：**網路的表達能力 (Expressive Power) 非常強大**，但定理僅保證「存在性」，不保證學得到或學得快。

---

## 3. 激活函數 Activation Functions

激活函數的核心作用：**引入非線性 (Non-linearity)**。

> 如果沒有激活函數（或只用線性激活函數），多層網路等價於單層線性變換，無法學習非線性模式。

### 3.1 Sigmoid（Logistic）

```
sigma(z) = 1 / (1 + exp(-z))

sigma'(z) = sigma(z) * (1 - sigma(z))
```

**特性：**
- 輸出範圍：(0, 1)，可解釋為機率
- 圖形：S 形曲線，平滑且連續
- 導數最大值為 0.25（當 z = 0）

**優點：**
- 輸出有界，適合二元分類 (Binary Classification) 的輸出層
- 平滑可微，數學性質良好

**缺點：**
- **梯度消失 (Vanishing Gradient)**：當 |z| 很大時，導數趨近 0，深層網路梯度指數衰減
- **非零中心 (Non-zero Centered)**：輸出永遠為正，導致參數更新方向受限
- **計算 exp(-z) 相對較慢**

### 3.2 Tanh（雙曲正切 Hyperbolic Tangent）

```
tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)) = 2 * sigma(2z) - 1

tanh'(z) = 1 - tanh(z)^2
```

**特性：**
- 輸出範圍：(-1, 1)，零中心 (Zero-centered)
- 導數最大值為 1（當 z = 0）

**優點：**
- 零中心輸出，參數更新更高效
- 導數最大值比 Sigmoid 大（1 vs. 0.25），梯度稍強

**缺點：**
- 仍然存在**梯度消失問題**（兩端飽和）
- 計算成本與 Sigmoid 相當

### 3.3 ReLU（修正線性單元 Rectified Linear Unit）

```
ReLU(z) = max(0, z)

ReLU'(z) = 1  if z > 0
            0  if z < 0
```

**特性：**
- 當 z > 0 時為線性，當 z <= 0 時為零
- 導數在正半軸恆為 1

**優點：**
- **計算高效**：只需比較與取 max
- **緩解梯度消失**：正半軸梯度恆為 1，不會飽和
- **稀疏激活 (Sparse Activation)**：部分神經元輸出為 0，等效於「關閉」
- 實務中收斂速度比 Sigmoid/Tanh 快約 6 倍（Krizhevsky et al., 2012）

**缺點：**
- **死亡 ReLU (Dying ReLU)**：若神經元進入 z < 0 區域，梯度永遠為 0，該神經元永久「死亡」
- 非零中心輸出
- z = 0 處不可微（實務中取 0 或 1 皆可）

### 3.4 Leaky ReLU

```
LeakyReLU(z) = z      if z > 0
               alpha*z if z <= 0

LeakyReLU'(z) = 1      if z > 0
                alpha   if z <= 0
```

其中 alpha 通常取 0.01。

**優點：**
- 解決「死亡 ReLU」問題：負半軸有微小梯度 alpha
- 保留 ReLU 的計算效率

**變體：**
- **Parametric ReLU (PReLU)**：alpha 作為可學習參數 (Learnable Parameter)
- **Randomized ReLU (RReLU)**：alpha 在訓練時隨機取值

### 3.5 GELU（高斯誤差線性單元 Gaussian Error Linear Unit）

```
GELU(z) = z * Phi(z)

其中 Phi(z) 是標準正態分佈的 CDF (累積分佈函數)
```

近似公式：

```
GELU(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
```

**特性：**
- 平滑版的 ReLU，在 z = 0 附近平滑過渡
- 可以視為：以 z 的大小為「信心」，決定是否保留該值

**優點：**
- Transformer 架構（BERT, GPT 系列）的預設激活函數
- 在 NLP 任務中表現通常優於 ReLU
- 平滑可微，利於優化

**缺點：**
- 計算成本高於 ReLU（需計算高斯分佈 CDF）

### 3.6 Swish（SiLU, Sigmoid Linear Unit）

```
Swish(z) = z * sigma(beta * z) = z / (1 + exp(-beta * z))
```

通常 beta = 1，此時：

```
Swish(z) = z / (1 + exp(-z))
```

**特性：**
- 由 Google Brain 透過自動搜尋 (Automated Search) 發現
- 非單調 (Non-monotonic)：在 z < 0 區域先下降再上升
- beta -> infinity 時趨近 ReLU，beta = 0 時退化為線性

**優點：**
- 在深層網路中表現常優於 ReLU
- 平滑且處處可微
- 自門控 (Self-gating) 特性

**缺點：**
- 計算成本高於 ReLU

```svg
<figure class="md-figure">
<svg viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="常見激活函數比較">
  <rect x="0" y="0" width="680" height="260" fill="#ffffff"/>
  <!-- 4 panels: Sigmoid, Tanh, ReLU, GELU — shared structure -->
  <!-- Panel 1: Sigmoid -->
  <g transform="translate(20,30)">
    <rect x="0" y="0" width="150" height="180" fill="#fafafa" stroke="#e5e7eb"/>
    <line x1="75" y1="0" x2="75" y2="180" stroke="#9ca3af" stroke-dasharray="3 3"/>
    <line x1="0" y1="90" x2="150" y2="90" stroke="#374151"/>
    <!-- σ(z) = 1/(1+e^-z), sampled -->
    <path d="M 0 178 Q 45 170 60 140 T 75 90 T 90 40 Q 105 10 150 2" fill="none" stroke="#ef4444" stroke-width="2.5"/>
    <text x="75" y="200" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">Sigmoid</text>
    <text x="75" y="216" text-anchor="middle" font-size="10" fill="#6b7280">(0, 1)</text>
  </g>
  <!-- Panel 2: Tanh -->
  <g transform="translate(190,30)">
    <rect x="0" y="0" width="150" height="180" fill="#fafafa" stroke="#e5e7eb"/>
    <line x1="75" y1="0" x2="75" y2="180" stroke="#9ca3af" stroke-dasharray="3 3"/>
    <line x1="0" y1="90" x2="150" y2="90" stroke="#374151"/>
    <path d="M 0 176 Q 45 170 60 130 T 75 90 T 90 50 Q 105 10 150 4" fill="none" stroke="#2563eb" stroke-width="2.5"/>
    <text x="75" y="200" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">Tanh</text>
    <text x="75" y="216" text-anchor="middle" font-size="10" fill="#6b7280">(-1, 1)</text>
  </g>
  <!-- Panel 3: ReLU -->
  <g transform="translate(360,30)">
    <rect x="0" y="0" width="150" height="180" fill="#fafafa" stroke="#e5e7eb"/>
    <line x1="75" y1="0" x2="75" y2="180" stroke="#9ca3af" stroke-dasharray="3 3"/>
    <line x1="0" y1="90" x2="150" y2="90" stroke="#374151"/>
    <polyline points="0,90 75,90 150,2" fill="none" stroke="#059669" stroke-width="2.5"/>
    <text x="75" y="200" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">ReLU</text>
    <text x="75" y="216" text-anchor="middle" font-size="10" fill="#6b7280">max(0, z)</text>
  </g>
  <!-- Panel 4: GELU -->
  <g transform="translate(530,30)">
    <rect x="0" y="0" width="150" height="180" fill="#fafafa" stroke="#e5e7eb"/>
    <line x1="75" y1="0" x2="75" y2="180" stroke="#9ca3af" stroke-dasharray="3 3"/>
    <line x1="0" y1="90" x2="150" y2="90" stroke="#374151"/>
    <!-- GELU: slight dip below 0 before rising -->
    <path d="M 0 90 Q 30 92 50 98 Q 65 102 75 92 Q 85 78 100 50 Q 120 20 150 2" fill="none" stroke="#7c3aed" stroke-width="2.5"/>
    <text x="75" y="200" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">GELU</text>
    <text x="75" y="216" text-anchor="middle" font-size="10" fill="#6b7280">z·Φ(z)</text>
  </g>
  <!-- Title -->
  <text x="340" y="18" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">常見激活函數形狀比較（x 為輸入 z，y 為 σ(z)）</text>
  <!-- Shared x-axis annotation -->
  <text x="340" y="248" text-anchor="middle" font-size="10" fill="#6b7280">虛線 = z=0 軸；注意 Sigmoid/Tanh 兩端飽和、ReLU 負半軸為 0、GELU 於 z&lt;0 平滑</text>
</svg>
<figcaption>示意圖：四種常見激活函數。Sigmoid/Tanh 在兩端梯度趨近 0 易造成梯度消失；ReLU 解決飽和問題但負半軸梯度為 0 造成「死亡 ReLU」；GELU 在 z&lt;0 保有微小梯度且平滑，已成 Transformer 標配。</figcaption>
</figure>
```

### 3.7 激活函數選擇指南 Selection Guide

| 場景 | 推薦 | 理由 |
|------|------|------|
| 隱藏層（一般） | ReLU | 簡單高效，首選嘗試 |
| 隱藏層（死亡 ReLU 問題） | Leaky ReLU / PReLU | 保留負半軸梯度 |
| Transformer / NLP | GELU | BERT/GPT 標配 |
| 深層 CNN | Swish / ReLU | Swish 常有微小提升 |
| 二元分類輸出層 | Sigmoid | 輸出機率 [0, 1] |
| 多元分類輸出層 | Softmax | 輸出機率分佈 |
| 回歸輸出層 | 無激活（線性） | 輸出任意實數 |

---

## 4. 反向傳播 Backpropagation

### 4.1 直覺解釋 Intuitive Explanation

反向傳播 (Backpropagation, 簡稱 Backprop) 是訓練神經網路的核心演算法。它回答了一個關鍵問題：

> **每個權重對最終誤差貢獻了多少？**

直覺類比 — **工廠品管追責：**

1. 工廠 (網路) 生產產品 (預測值)
2. 品管 (損失函數) 發現產品有瑕疵 (誤差)
3. 品管**往回追溯** (反向傳播) 每個工序 (層) 的責任
4. 每個工人 (權重) 根據自己的責任大小**調整操作** (參數更新)

### 4.2 數學原理 Mathematical Principle

反向傳播基於**鏈式法則 (Chain Rule)**：

對於簡單的兩層網路 x → W1 → h → W2 → y：

```
dL/dW1 = dL/dy * dy/dh * dh/dW1
```

**前向傳播 (Forward Pass)：**
1. z^(1) = W1 * x + b1
2. h = f(z^(1))（激活函數）
3. z^(2) = W2 * h + b2
4. y_hat = g(z^(2))（輸出激活）
5. L = Loss(y_hat, y)

**反向傳播 (Backward Pass)：**
1. 計算 dL/dy_hat
2. 計算 dL/dz^(2) = dL/dy_hat * g'(z^(2))
3. 計算 dL/dW2 = dL/dz^(2) * h^T
4. 計算 dL/dh = W2^T * dL/dz^(2)
5. 計算 dL/dz^(1) = dL/dh * f'(z^(1))
6. 計算 dL/dW1 = dL/dz^(1) * x^T

### 4.3 計算圖 Computational Graph

現代框架（如 PyTorch）在前向傳播時自動建構計算圖 (Computational Graph)，反向傳播時沿著圖的反方向自動計算梯度 — 這就是 **autograd** 的原理。

---

## 5. 梯度消失與梯度爆炸 Vanishing & Exploding Gradients

### 5.1 梯度消失 Vanishing Gradient

考慮一個 L 層深的網路，反向傳播時梯度需要連乘各層的導數：

```
dL/dW1 ~ 連乘(l=1 to L) f'(z^(l)) * W_l
```

**使用 Sigmoid 時：**
- sigma'(z) 的最大值僅 0.25
- 連乘 L 層後：0.25^L
- 10 層：0.25^10 ≈ 1e-6，梯度幾乎消失！

**後果：**
- 淺層 (靠近輸入的層) 幾乎不更新
- 網路無法有效學習深層特徵

### 5.2 梯度爆炸 Exploding Gradient

如果權重矩陣的最大特徵值 (Largest Eigenvalue) > 1，連乘後梯度會指數增長：

```
連乘(l=1 to L) ||W_l|| → infinity
```

**後果：**
- 參數更新幅度過大，訓練不穩定
- 損失函數劇烈震盪甚至 NaN

### 5.3 解決方案 Solutions

| 問題 | 解決方案 |
|------|---------|
| 梯度消失 | ReLU 系列激活函數、殘差連接 (Residual Connection)、LSTM 門控機制 |
| 梯度爆炸 | 梯度裁剪 (Gradient Clipping)、適當的權重初始化、BatchNorm |
| 兩者皆有 | 適當的初始化 (Xavier/He)、BatchNorm、學習率排程 |

---

## 6. 正則化技術 Regularization Techniques

正則化的目標：**控制模型複雜度，防止過擬合 (Overfitting)**。

### 6.1 L1 正則化（Lasso）

```
L_total = L_data + lambda * sum(|w_i|)
```

**效果：**
- 傾向讓部分權重變為**精確的 0**（稀疏性 Sparsity）
- 等效於特徵選擇 (Feature Selection)
- 幾何解釋：L1 約束區域為菱形，解容易落在角上

### 6.2 L2 正則化（Ridge / Weight Decay）

```
L_total = L_data + lambda * sum(w_i^2)
```

**效果：**
- 讓所有權重**趨向更小的值**，但不會精確為 0
- 防止任何單一特徵主導模型
- 幾何解釋：L2 約束區域為圓/球，解均勻收縮

**PyTorch 中的使用：**
```python
# weight_decay 即 L2 正則化係數
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### 6.3 Dropout（Srivastava et al., 2014）

**原理：**
在訓練時，每個神經元以機率 p（通常 p = 0.5 或 p = 0.2）被隨機「關閉」（輸出設為 0）。

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dropout 訓練 vs 推論示意圖">
  <rect x="0" y="0" width="640" height="340" fill="#ffffff"/>
  <!-- Titles -->
  <text x="160" y="26" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">訓練時 Training（p=0.5）</text>
  <text x="480" y="26" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">推論時 Inference</text>
  <!-- LEFT: training with some neurons dropped -->
  <g transform="translate(20,40)">
    <!-- Edges (dimmed for dropped ones) -->
    <g stroke="#cbd5e1" stroke-width="0.8" fill="none">
      <line x1="40" y1="60" x2="140" y2="40"/><line x1="40" y1="60" x2="140" y2="200"/>
      <line x1="40" y1="220" x2="140" y2="40"/><line x1="40" y1="220" x2="140" y2="200"/>
      <line x1="140" y1="40" x2="240" y2="60"/><line x1="140" y1="40" x2="240" y2="220"/>
      <line x1="140" y1="200" x2="240" y2="60"/><line x1="140" y1="200" x2="240" y2="220"/>
    </g>
    <!-- Dropped edges (very faded) -->
    <g stroke="#f3f4f6" stroke-width="0.8" fill="none">
      <line x1="40" y1="60" x2="140" y2="120"/><line x1="40" y1="220" x2="140" y2="120"/>
      <line x1="140" y1="120" x2="240" y2="60"/><line x1="140" y1="120" x2="240" y2="220"/>
    </g>
    <!-- Input layer -->
    <g fill="#dbeafe" stroke="#2563eb" stroke-width="1.5">
      <circle cx="40" cy="60" r="14"/><circle cx="40" cy="220" r="14"/>
    </g>
    <!-- Hidden layer — middle neuron dropped (X marker) -->
    <g fill="#fef3c7" stroke="#d97706" stroke-width="1.5">
      <circle cx="140" cy="40" r="14"/>
      <circle cx="140" cy="200" r="14"/>
    </g>
    <!-- Dropped neuron (dashed outline, red X) -->
    <circle cx="140" cy="120" r="14" fill="#f3f4f6" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3 2"/>
    <line x1="132" y1="112" x2="148" y2="128" stroke="#ef4444" stroke-width="2"/>
    <line x1="148" y1="112" x2="132" y2="128" stroke="#ef4444" stroke-width="2"/>
    <!-- Output layer -->
    <g fill="#fee2e2" stroke="#dc2626" stroke-width="1.5">
      <circle cx="240" cy="60" r="14"/><circle cx="240" cy="220" r="14"/>
    </g>
    <!-- Ghost of boxed area -->
    <rect x="0" y="0" width="280" height="260" fill="none" stroke="#e5e7eb" stroke-dasharray="4 4"/>
    <text x="140" y="284" text-anchor="middle" font-size="11" fill="#ef4444" font-weight="600">中間神經元以機率 p=0.5 被關閉</text>
  </g>
  <!-- RIGHT: inference with all neurons active -->
  <g transform="translate(340,40)">
    <g stroke="#cbd5e1" stroke-width="0.8" fill="none">
      <line x1="40" y1="60" x2="140" y2="40"/><line x1="40" y1="60" x2="140" y2="120"/><line x1="40" y1="60" x2="140" y2="200"/>
      <line x1="40" y1="220" x2="140" y2="40"/><line x1="40" y1="220" x2="140" y2="120"/><line x1="40" y1="220" x2="140" y2="200"/>
      <line x1="140" y1="40" x2="240" y2="60"/><line x1="140" y1="40" x2="240" y2="220"/>
      <line x1="140" y1="120" x2="240" y2="60"/><line x1="140" y1="120" x2="240" y2="220"/>
      <line x1="140" y1="200" x2="240" y2="60"/><line x1="140" y1="200" x2="240" y2="220"/>
    </g>
    <g fill="#dbeafe" stroke="#2563eb" stroke-width="1.5">
      <circle cx="40" cy="60" r="14"/><circle cx="40" cy="220" r="14"/>
    </g>
    <g fill="#fef3c7" stroke="#d97706" stroke-width="1.5">
      <circle cx="140" cy="40" r="14"/><circle cx="140" cy="120" r="14"/><circle cx="140" cy="200" r="14"/>
    </g>
    <g fill="#fee2e2" stroke="#dc2626" stroke-width="1.5">
      <circle cx="240" cy="60" r="14"/><circle cx="240" cy="220" r="14"/>
    </g>
    <rect x="0" y="0" width="280" height="260" fill="none" stroke="#e5e7eb" stroke-dasharray="4 4"/>
    <text x="140" y="284" text-anchor="middle" font-size="11" fill="#059669" font-weight="600">所有神經元啟用（輸出乘以 1-p 保持期望值）</text>
  </g>
  <!-- Arrow between halves -->
  <path d="M 305 170 L 340 170" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowHead)"/>
  <defs><marker id="arrowHead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af"/></marker></defs>
</svg>
<figcaption>示意圖：Dropout。訓練階段每次 forward 隨機關閉一部分神經元（紅色 X），強迫剩餘神經元獨立學習有效特徵；推論時啟用全部神經元並將輸出縮放 (1-p)，以維持期望激活值與訓練時一致（PyTorch 的 Inverted Dropout 在訓練時先除 1-p，推論時不需調整）。</figcaption>
</figure>
```

**為什麼有效？**
1. **集成效果 (Ensemble Effect)**：每次 forward pass 等於訓練一個不同的子網路，最終效果類似多模型投票
2. **防止共適應 (Co-adaptation)**：強迫每個神經元獨立學習有用的特徵
3. **隱含正則化**：減少模型有效容量

**注意事項：**
- 測試時不使用 Dropout，但需要對權重進行縮放（PyTorch 的 nn.Dropout 在訓練時自動除以 (1-p)，稱為 Inverted Dropout）
- BatchNorm 與 Dropout 同時使用時需謹慎（可能產生分佈不匹配）

### 6.4 早停 Early Stopping

**原理：**
監控驗證集損失 (Validation Loss)，當驗證集損失不再下降（或開始上升）時，提前終止訓練。

```
損失 Loss
  |      \
  |       \   訓練損失 (Training Loss)
  |        \___________________________
  |         \  /
  |          \/ <- 最佳模型 (Best Model)
  |           /\
  |          /  \ 驗證損失 (Validation Loss)
  |         /    \
  └──────────────────→ Epoch
            ^
       早停點 (Early Stopping Point)
```

**實作要點：**
- 設定耐心值 (Patience)：連續幾個 epoch 無改善才停止
- 保存驗證損失最低時的模型權重 (Best Model Checkpoint)
- 通常 patience = 5-20 epochs

---

## 7. 批次正規化 Batch Normalization（Ioffe & Szegedy, 2015）

### 7.1 動機 Motivation

**內部協變量偏移 (Internal Covariate Shift, ICS)：**
每一層的輸入分佈會隨著前面層的參數更新而改變，導致後面層需要不斷「追趕」新的分佈，訓練效率低下。

> 近年研究（Santurkar et al., 2018）指出 BatchNorm 的效果可能更多來自「平滑損失面 (Smoothing the Loss Landscape)」，而非直接解決 ICS。

### 7.2 計算過程 Computation

對於一個 mini-batch B = {x1, x2, ..., xm}：

**Step 1** — 計算 batch 均值：
```
mu_B = (1/m) * sum(x_i)
```

**Step 2** — 計算 batch 方差：
```
sigma_B^2 = (1/m) * sum((x_i - mu_B)^2)
```

**Step 3** — 正規化：
```
x_hat_i = (x_i - mu_B) / sqrt(sigma_B^2 + epsilon)
```

**Step 4** — 縮放與平移（可學習參數）：
```
y_i = gamma * x_hat_i + beta
```

其中 gamma（縮放）和 beta（平移）是**可學習參數**，讓網路自行決定最佳的分佈。

### 7.3 使用位置 Placement

常見做法：放在線性變換之後、激活函數之前：

```
h = f(BN(W * x + b))
```

也有研究建議放在激活函數之後，實務中差異不大。

### 7.4 訓練 vs. 推論 Training vs. Inference

| 階段 | 均值 / 方差 |
|------|-----------|
| 訓練 Training | 使用 mini-batch 統計量 |
| 推論 Inference | 使用訓練期間累積的移動平均 (Running Mean/Variance) |

> 這就是為什麼在推論時需要呼叫 `model.eval()` 的原因之一。

### 7.5 BatchNorm 的好處 Benefits

1. **加速收斂**：可以使用更大的學習率
2. **正則化效果**：mini-batch 的隨機統計量引入雜訊，類似 Dropout
3. **減少對初始化的敏感度**：即使初始化不完美，仍能正常訓練
4. **平滑損失面**：讓梯度更穩定

---

## 8. 權重初始化 Weight Initialization

### 8.1 為什麼初始化重要？ Why Initialization Matters?

- **全部設為 0**：所有神經元學到相同的東西（對稱性問題 Symmetry Breaking 失敗）
- **過大的值**：前向傳播信號爆炸 → 梯度爆炸
- **過小的值**：前向傳播信號消失 → 梯度消失

目標：讓每層的**輸出方差 (Variance)** 保持穩定，既不放大也不縮小。

### 8.2 Xavier 初始化（Glorot Initialization）

適用於 **Sigmoid / Tanh** 激活函數。

均勻分佈版本：
```
W ~ Uniform[-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out))]
```

正態分佈版本：
```
W ~ Normal(0, 2/(n_in + n_out))
```

其中 n_in 和 n_out 分別是該層的輸入與輸出維度。

**原理：** 同時考慮前向與反向傳播，讓兩個方向的方差都保持穩定。

### 8.3 He 初始化（Kaiming Initialization）

適用於 **ReLU** 及其變體。

```
W ~ Normal(0, 2/n_in)
```

**原理：** ReLU 會將約一半的神經元輸出設為 0，因此需要將方差放大 2 倍來補償。

### 8.4 PyTorch 中的使用

```python
import torch.nn as nn

# Xavier
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

# He (Kaiming)
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

> PyTorch 的 nn.Linear 預設使用 Kaiming Uniform 初始化。

---

## 9. PyTorch 基礎 PyTorch Fundamentals

### 9.1 Tensor

Tensor 是 PyTorch 的核心資料結構，類似 NumPy 的 ndarray，但支援 GPU 加速與自動微分。

```python
import torch

# 建立 Tensor
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.randn(3, 4)         # 標準正態隨機
z = torch.zeros(2, 3)         # 全零
w = torch.ones(2, 3)          # 全一

# 基本運算
a = x + 1                     # 逐元素加法
b = torch.matmul(y.T, y)      # 矩陣乘法
c = y.sum(dim=0)               # 沿特定維度求和

# GPU 加速
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
```

### 9.2 autograd 自動微分

PyTorch 的 autograd 引擎自動計算梯度：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x + 1       # y = x^3 + 2x + 1

y.backward()                  # 自動反向傳播
print(x.grad)                 # dy/dx = 3x^2 + 2 = 14.0
```

**關鍵概念：**
- `requires_grad=True`：告訴 PyTorch 追蹤此變數的運算
- `.backward()`：觸發反向傳播，計算梯度
- `.grad`：存放計算得到的梯度
- `.detach()`：從計算圖中分離，不再追蹤
- `with torch.no_grad():`：推論時使用，不建構計算圖以節省記憶體

### 9.3 nn.Module

所有神經網路模型都繼承自 nn.Module：

```python
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 使用
model = SimpleMLP(784, 256, 10)
output = model(torch.randn(32, 784))   # batch_size=32
print(output.shape)                     # torch.Size([32, 10])
```

### 9.4 訓練循環 Training Loop

```python
import torch.optim as optim

model = SimpleMLP(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # 1. 前向傳播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 2. 反向傳播
        optimizer.zero_grad()       # 清除舊梯度
        loss.backward()             # 計算梯度
        optimizer.step()            # 更新參數
```

---

## 10. 本週實作預告 Lab Preview

本週 Notebook 包含以下實作：

1. **激活函數視覺化**：繪製六種激活函數的函數圖與導數圖
2. **建構 MLP**：使用 PyTorch 建構簡單 MLP 並進行前向傳播
3. **梯度消失視覺化**：比較深層 Sigmoid 網路與 ReLU 網路的梯度變化
4. **Dropout 效果視覺化**：觀察 Dropout 如何影響模型的預測
5. **BatchNorm 效果比較**：有/無 BatchNorm 對訓練曲線的影響
6. **MNIST 分類器**：在 MNIST / Fashion-MNIST 上訓練完整分類器

---

## 關鍵詞彙 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 感知器 | Perceptron | 最簡單的人工神經元模型 |
| 多層感知器 | Multi-Layer Perceptron (MLP) | 多層堆疊的感知器網路 |
| 前饋網路 | Feedforward Network | 信號單向流動的神經網路 |
| 激活函數 | Activation Function | 引入非線性的函數 |
| 反向傳播 | Backpropagation | 利用鏈式法則計算梯度的演算法 |
| 梯度消失 | Vanishing Gradient | 深層網路梯度指數衰減的現象 |
| 梯度爆炸 | Exploding Gradient | 深層網路梯度指數增長的現象 |
| 正則化 | Regularization | 控制模型複雜度的技術 |
| 批次正規化 | Batch Normalization (BatchNorm) | 對每層輸入做正規化 |
| 權重初始化 | Weight Initialization | 設定網路初始權重的策略 |
| 丟棄法 | Dropout | 訓練時隨機關閉神經元 |
| 早停 | Early Stopping | 驗證損失不再改善時停止訓練 |
| 權重衰減 | Weight Decay | L2 正則化的等效名稱 |
| 萬能近似定理 | Universal Approximation Theorem | 足夠寬的單隱藏層可逼近任何連續函數 |
| 張量 | Tensor | PyTorch 的核心多維陣列結構 |
| 自動微分 | Autograd | 自動計算梯度的引擎 |
| 計算圖 | Computational Graph | 記錄運算的有向圖，用於反向傳播 |
| 死亡 ReLU | Dying ReLU | ReLU 神經元永久輸出 0 的問題 |
| 內部協變量偏移 | Internal Covariate Shift (ICS) | 各層輸入分佈隨訓練改變的現象 |
| 稀疏激活 | Sparse Activation | 部分神經元輸出為零 |

---

## 延伸閱讀 Further Reading

- Goodfellow et al., "Deep Learning" (Ch. 6: Deep Feedforward Networks)
- Ioffe & Szegedy, 2015, "Batch Normalization: Accelerating Deep Network Training"
- He et al., 2015, "Delving Deep into Rectifiers" (He Initialization)
- Glorot & Bengio, 2010, "Understanding the difficulty of training deep feedforward neural networks"
- Hendrycks & Gimpel, 2016, "Gaussian Error Linear Units (GELUs)"
- Ramachandran et al., 2017, "Searching for Activation Functions" (Swish)
- Srivastava et al., 2014, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- PyTorch 官方教學：https://pytorch.org/tutorials/beginner/basics/intro.html
