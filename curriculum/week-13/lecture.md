# 第 13 週：RNN/序列建模（LSTM/GRU；Transformers 概念）
# Week 13: RNN/Sequence Modeling (LSTM/GRU; Transformers Concepts)

## 學習目標 Learning Objectives
1. 理解序列資料 (Sequential Data) 的特性與應用場景
2. 掌握遞迴神經網路 (RNN) 的結構、展開圖與前向傳播過程
3. 了解 RNN 梯度消失 (Vanishing Gradient) 問題的成因與後果
4. 深入理解 LSTM (Long Short-Term Memory) 的三門控機制
5. 認識 GRU (Gated Recurrent Unit) 作為 LSTM 的簡化替代方案
6. 了解雙向 RNN (Bidirectional RNN) 的架構與應用
7. 理解注意力機制 (Attention Mechanism) 的動機與原理
8. 掌握 Transformer 架構的核心組件：Self-Attention、Multi-Head Attention
9. 認識位置編碼 (Positional Encoding) 與 Encoder-Decoder 架構
10. 理解從 RNN 到 Transformer 的技術演進脈絡

---

## 1. 序列資料的特性 Characteristics of Sequential Data

### 1.1 什麼是序列資料？ What is Sequential Data?

序列資料是指資料點之間具有**時間順序**或**位置順序**依賴關係的資料。與傳統的表格資料不同，序列資料中每個元素的意義不僅取決於自身的值，還取決於它在序列中的**位置**以及與其他元素的**上下文關係 (Contextual Relationship)**。

> **核心特性：** 序列中的資料點並非獨立同分布 (i.i.d.)，而是存在時間或空間上的相依性 (Temporal/Spatial Dependency)。

### 1.2 序列資料的類型 Types of Sequential Data

| 類型 | 英文 | 範例 | 特點 |
|------|------|------|------|
| 時間序列 | Time Series | 股票價格、天氣溫度、心電圖 | 等間隔取樣，時間相依 |
| 自然語言 | Natural Language | 句子、文章、對話 | 離散符號序列，語義相依 |
| 語音訊號 | Speech Signal | 語音波形、音素序列 | 連續信號的離散化 |
| 生物序列 | Biological Sequence | DNA/RNA 序列、蛋白質序列 | 字母表有限，位置敏感 |
| 音樂 | Music | MIDI 序列、和弦進行 | 節奏與旋律的時間模式 |
| 事件序列 | Event Sequence | 使用者點擊流、交易記錄 | 不等間隔，行為模式 |

### 1.3 序列建模的輸入/輸出模式 Input/Output Patterns

序列模型可以處理多種不同的輸入/輸出組合：

```
(1) One-to-One         (2) One-to-Many         (3) Many-to-One
    ┌───┐                   ┌───┐                 ┌───┐
    │ y │                   │y1 │y2 │y3│           │ y │
    └─┬─┘                   └─┬──┬──┬─┘           └─┬─┘
      │                       │  │  │               │
    ┌─┴─┐                   ┌─┴──┴──┴─┐         ┌─┬─┴─┬─┐
    │ x │                   │    x    │         │x1│x2│x3│
    └───┘                   └─────────┘         └──┴──┴──┘
  影像分類                   圖片說明生成           情感分析

(4) Many-to-Many (同步)   (5) Many-to-Many (異步，Seq2Seq)
    ┌───┬───┬───┐             ┌───┬───┐
    │y1 │y2 │y3 │             │y1 │y2 │
    └─┬─┴─┬─┴─┬─┘             └─┬──┴─┬─┘
      │   │   │                  │    │
    ┌─┴─┬─┴─┬─┴─┐           ┌─┬─┴──┬─┴─┬─┐
    │x1 │x2 │x3 │           │x1│x2 │x3 │ │
    └───┴───┴───┘           └──┴───┴───┴─┘
  詞性標注 (POS Tagging)     機器翻譯 (Machine Translation)
```

### 1.4 為什麼全連接網路不適用？ Why Fully-Connected Networks Fail?

全連接網路 (Fully-Connected Network, FCN) 處理序列資料面臨三大問題：

1. **固定長度輸入**：FCN 要求輸入維度固定，但序列長度是可變的 (Variable-Length)
2. **無法捕捉順序**：FCN 將輸入視為無序特徵向量，無法區分 "狗咬人" 和 "人咬狗"
3. **參數爆炸**：若以 one-hot 展平長序列，參數量會極其龐大且無法共享時間步之間學到的知識

> 這些限制促使研究者設計出能「記住過去」的神經網路架構 -- 遞迴神經網路 (RNN)。

---

## 2. 遞迴神經網路 (RNN) 結構與展開圖

### 2.1 RNN 的基本概念 Basic Concept

遞迴神經網路 (Recurrent Neural Network, RNN) 的核心思想是引入**隱藏狀態 (Hidden State)**，讓網路能夠在處理序列時「記住」之前的資訊。

**與前饋網路的關鍵區別：** RNN 在隱藏層中加入了**自迴圈連接 (Self-Loop Connection)**，使得當前時間步的計算可以利用前一時間步的隱藏狀態。

### 2.2 RNN 的摺疊與展開表示 Folded vs Unfolded Representation

```svg
<figure class="md-figure">
<svg viewBox="0 0 720 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="RNN 摺疊 vs 展開表示">
  <rect x="0" y="0" width="720" height="320" fill="#ffffff"/>
  <!-- LEFT: folded -->
  <g transform="translate(30,40)">
    <text x="90" y="-12" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">摺疊 Folded</text>
    <!-- x -->
    <g fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"><circle cx="90" cy="220" r="18"/></g>
    <text x="90" y="224" text-anchor="middle" font-size="12" fill="#1e3a5f">x</text>
    <!-- h (with self loop) -->
    <g fill="#fef3c7" stroke="#d97706" stroke-width="1.5"><circle cx="90" cy="130" r="24"/></g>
    <text x="90" y="134" text-anchor="middle" font-size="13" fill="#7f1d1d" font-weight="600">h</text>
    <!-- Self loop on h -->
    <path d="M 112 115 C 145 100 145 145 112 143" fill="none" stroke="#d97706" stroke-width="1.5" marker-end="url(#loopArr)"/>
    <text x="150" y="134" font-size="11" fill="#b45309" font-weight="600">W_hh</text>
    <!-- y -->
    <g fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"><circle cx="90" cy="45" r="18"/></g>
    <text x="90" y="49" text-anchor="middle" font-size="12" fill="#7f1d1d">y</text>
    <!-- Edges x->h->y -->
    <line x1="90" y1="202" x2="90" y2="154" stroke="#374151" stroke-width="1.5" marker-end="url(#foldArr)"/>
    <line x1="90" y1="106" x2="90" y2="63" stroke="#374151" stroke-width="1.5" marker-end="url(#foldArr)"/>
    <defs>
      <marker id="foldArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker>
      <marker id="loopArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#d97706"/></marker>
      <marker id="unfoldArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker>
    </defs>
    <text x="100" y="178" font-size="10" fill="#6b7280">W_xh</text>
    <text x="100" y="88" font-size="10" fill="#6b7280">W_hy</text>
  </g>
  <!-- "展開" arrow between -->
  <g transform="translate(180,160)">
    <path d="M 0 0 L 60 0" stroke="#374151" stroke-width="2" marker-end="url(#unfoldBigArr)"/>
    <defs>
      <marker id="unfoldBigArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker>
    </defs>
    <text x="30" y="-10" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">沿時間展開</text>
    <text x="30" y="22" text-anchor="middle" font-size="10" fill="#6b7280">unfold over t</text>
  </g>
  <!-- RIGHT: unfolded over 3 time steps -->
  <g transform="translate(270,40)">
    <text x="200" y="-12" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">展開 Unfolded（參數共享）</text>
    <!-- time labels (bottom) -->
    <text x="50" y="260" text-anchor="middle" font-size="11" fill="#6b7280">t-1</text>
    <text x="180" y="260" text-anchor="middle" font-size="11" fill="#6b7280">t</text>
    <text x="310" y="260" text-anchor="middle" font-size="11" fill="#6b7280">t+1</text>
    <!-- Inputs -->
    <g fill="#dbeafe" stroke="#2563eb" stroke-width="1.5">
      <circle cx="50" cy="220" r="16"/><circle cx="180" cy="220" r="16"/><circle cx="310" cy="220" r="16"/>
    </g>
    <g font-size="11" fill="#1e3a5f" text-anchor="middle">
      <text x="50" y="224">xₜ₋₁</text><text x="180" y="224">xₜ</text><text x="310" y="224">xₜ₊₁</text>
    </g>
    <!-- Hidden states -->
    <g fill="#fef3c7" stroke="#d97706" stroke-width="1.5">
      <circle cx="50" cy="130" r="22"/><circle cx="180" cy="130" r="22"/><circle cx="310" cy="130" r="22"/>
    </g>
    <g font-size="12" fill="#7f1d1d" text-anchor="middle" font-weight="600">
      <text x="50" y="134">hₜ₋₁</text><text x="180" y="134">hₜ</text><text x="310" y="134">hₜ₊₁</text>
    </g>
    <!-- Output -->
    <g fill="#fee2e2" stroke="#dc2626" stroke-width="1.5">
      <circle cx="50" cy="45" r="16"/><circle cx="180" cy="45" r="16"/><circle cx="310" cy="45" r="16"/>
    </g>
    <g font-size="11" fill="#7f1d1d" text-anchor="middle">
      <text x="50" y="49">yₜ₋₁</text><text x="180" y="49">yₜ</text><text x="310" y="49">yₜ₊₁</text>
    </g>
    <!-- x → h edges -->
    <g stroke="#374151" stroke-width="1.2" fill="none">
      <line x1="50" y1="204" x2="50" y2="152" marker-end="url(#unfoldArr)"/>
      <line x1="180" y1="204" x2="180" y2="152" marker-end="url(#unfoldArr)"/>
      <line x1="310" y1="204" x2="310" y2="152" marker-end="url(#unfoldArr)"/>
    </g>
    <!-- h → y edges -->
    <g stroke="#374151" stroke-width="1.2" fill="none">
      <line x1="50" y1="108" x2="50" y2="63" marker-end="url(#unfoldArr)"/>
      <line x1="180" y1="108" x2="180" y2="63" marker-end="url(#unfoldArr)"/>
      <line x1="310" y1="108" x2="310" y2="63" marker-end="url(#unfoldArr)"/>
    </g>
    <!-- h_{t-1} → h_t → h_{t+1} (time-directed recurrence) -->
    <g stroke="#d97706" stroke-width="1.8" fill="none">
      <line x1="72" y1="130" x2="158" y2="130" marker-end="url(#unfoldArr)"/>
      <line x1="202" y1="130" x2="288" y2="130" marker-end="url(#unfoldArr)"/>
      <!-- From prev (dashed) -->
      <line x1="0" y1="130" x2="28" y2="130" stroke-dasharray="3 3" marker-end="url(#unfoldArr)"/>
      <line x1="332" y1="130" x2="360" y2="130" stroke-dasharray="3 3"/>
    </g>
    <text x="115" y="122" text-anchor="middle" font-size="10" fill="#b45309">W_hh</text>
    <text x="245" y="122" text-anchor="middle" font-size="10" fill="#b45309">W_hh</text>
    <text x="365" y="132" font-size="10" fill="#6b7280">...</text>
  </g>
</svg>
<figcaption>示意圖：RNN 摺疊 vs 展開表示。左側用自迴圈表示 hidden state h 的遞迴；右側沿時間軸展開：同一組權重 (W_xh, W_hh, W_hy) 在所有時間步共享，隱藏狀態 hₜ 同時承載當前輸入 xₜ 與歷史資訊 hₜ₋₁。</figcaption>
</figure>
```

### 2.3 RNN 的數學公式 Mathematical Formulation

在每個時間步 $t$，RNN 執行以下計算：

**隱藏狀態更新：**
$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

**輸出計算：**
$$y_t = W_{hy} \cdot h_t + b_y$$

其中：
- $x_t \in \mathbb{R}^d$：時間步 $t$ 的輸入向量
- $h_t \in \mathbb{R}^n$：時間步 $t$ 的隱藏狀態向量
- $y_t$：時間步 $t$ 的輸出
- $W_{xh} \in \mathbb{R}^{n \times d}$：輸入到隱藏層的權重矩陣
- $W_{hh} \in \mathbb{R}^{n \times n}$：隱藏層到隱藏層的權重矩陣（**參數共享**的關鍵）
- $W_{hy}$：隱藏層到輸出的權重矩陣
- $b_h, b_y$：偏差項 (Bias)
- $\tanh$：雙曲正切激活函數，將值壓縮到 $[-1, 1]$

### 2.4 參數共享 Parameter Sharing

RNN 的一個關鍵特性是**參數共享 (Parameter Sharing)**：$W_{xh}$、$W_{hh}$、$W_{hy}$ 在所有時間步上是相同的。

**好處：**
- 大幅減少參數量（與序列長度無關）
- 能處理任意長度的序列
- 在不同時間位置學到的特徵可以泛化

### 2.5 沿時間反向傳播 Backpropagation Through Time (BPTT)

RNN 的訓練使用**沿時間反向傳播 (BPTT)** 演算法，本質上是將展開後的 RNN 視為一個深度前饋網路來進行反向傳播。

```
損失計算 (Loss Computation)：

  L(t-1)       L(t)        L(t+1)        L_total = Σ L(t)
    ↑            ↑            ↑
  y(t-1)       y(t)        y(t+1)
    ↑            ↑            ↑
  h(t-1) ──→  h(t)  ──→  h(t+1)
    ↑    ←──    ↑    ←──    ↑          ← 梯度反向流動
  x(t-1)       x(t)        x(t+1)       (Gradient Backflow)
```

梯度需要沿時間步逐步傳播回去，這直接導致了下一節要討論的梯度消失問題。

---

## 3. RNN 的梯度消失問題 Vanishing Gradient Problem

### 3.1 問題的根源 Root Cause

考慮梯度從時間步 $T$ 傳播到時間步 $1$ 的過程。根據鏈式法則 (Chain Rule)：

$$\frac{\partial L_T}{\partial h_1} = \frac{\partial L_T}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

由於 $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$，我們有：

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(\cdot)) \cdot W_{hh}$$

連續相乘 $T-1$ 個矩陣 $W_{hh}$：
- 若 $W_{hh}$ 的最大特徵值 $< 1$：梯度**指數衰減** → **梯度消失 (Vanishing Gradient)**
- 若 $W_{hh}$ 的最大特徵值 $> 1$：梯度**指數增長** → **梯度爆炸 (Exploding Gradient)**

### 3.2 後果 Consequences

```
梯度大小 (Gradient Magnitude)
  │
  │ ████                              ← 梯度爆炸 Exploding
  │ ██
  │ █
  │ ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓          ← 理想狀態 Ideal
  │                      ░ ░ ░ ░ ░   ← 梯度消失 Vanishing
  │                            · · ·
  └──────────────────────────────────▶
   t=T  t=T-1  ...   t=3  t=2  t=1    (越早的時間步)
                                       (Earlier Timesteps)
```

**梯度消失的實際影響：**
- 模型無法學到**長程依賴 (Long-Range Dependencies)**
- 例如："The cat, which already ate a lot of food earlier that day, **was** full." — 模型可能無法將 "was" 與遠處的 "cat" 聯繫起來
- 早期時間步的權重幾乎不會被更新

### 3.3 梯度爆炸的緩解 Mitigating Exploding Gradients

梯度爆炸相對容易處理，常用**梯度裁剪 (Gradient Clipping)**：

```python
# 梯度裁剪 Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

但梯度消失問題更加棘手，需要從**架構層面**解決 -- 這就是 LSTM 和 GRU 被發明的動機。

---

## 4. LSTM (Long Short-Term Memory)

### 4.1 核心思想 Core Idea

LSTM 由 Hochreiter & Schmidhuber 於 1997 年提出，其核心思想是引入一條**細胞狀態 (Cell State)** 通道，讓資訊可以在不經過非線性轉換的情況下沿時間流動，從而有效緩解梯度消失問題。

> **類比：** 如果把 RNN 比作一條河流，資訊像水一樣流過每個水壩（非線性激活），逐漸被消耗殆盡。LSTM 則增加了一條**高速公路 (Highway)**，讓重要資訊可以不受阻礙地快速傳遞。

### 4.2 LSTM 的結構 Architecture

LSTM 包含三個門控 (Gate) 和一個細胞狀態：

```
                    ┌─────────────────────────────────────┐
                    │          LSTM Cell                   │
                    │                                     │
  C(t-1) ────────────→ [×]────→ [+] ─────────────────────→ C(t)
                    │   ↑        ↑                        │
                    │   f(t)   [×]                        │
                    │          ↑  ↑                        │
                    │        i(t) C̃(t)                    │
                    │                                     │
  h(t-1) ──────┬───│──→ [σ] [σ] [tanh] [σ] ←── x(t)     │
               │   │     f    i    C̃     o               │
               │   │                     │                │
               │   │                   [×]                │
               │   │                   ↑  ↑               │
               │   │                 o(t) tanh(C(t))      │
               │   │                     │                │
               │   │                     ├────────────────→ h(t)
               │   └─────────────────────┘                │
               └──────────────────────────────────────────┘
```

### 4.3 三個門控機制 Three Gate Mechanisms

#### 遺忘門 Forget Gate ($f_t$)

決定從細胞狀態中**丟棄**哪些資訊。

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- 輸出值在 $[0, 1]$ 之間
- $f_t = 0$：完全遺忘（丟棄該維度的記憶）
- $f_t = 1$：完全保留（保留該維度的記憶）
- **直覺理解：** 讀到新句子時，忘記前一個句子的主語

#### 輸入門 Input Gate ($i_t$ 與 $\tilde{C}_t$)

決定將哪些**新資訊**寫入細胞狀態。

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- $i_t$：門控信號，決定哪些維度要更新
- $\tilde{C}_t$：候選記憶 (Candidate Memory)，即要寫入的新值
- **直覺理解：** 讀到新的主語時，將其存入記憶

#### 細胞狀態更新 Cell State Update

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

- $\odot$ 表示逐元素乘法 (Element-wise Multiplication)
- 先用遺忘門決定要保留多少舊記憶
- 再用輸入門決定要加入多少新記憶
- **這就是「高速公路」：** $C_t$ 可以通過 $f_t \approx 1$ 幾乎不衰減地傳遞梯度

#### 輸出門 Output Gate ($o_t$)

決定細胞狀態中的哪些部分要作為**輸出**。

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

- 先對細胞狀態做 $\tanh$ 壓縮到 $[-1, 1]$
- 再由輸出門決定哪些維度要輸出
- **直覺理解：** 根據當前的語境，決定要展現記憶的哪個面向

### 4.4 LSTM 如何解決梯度消失 How LSTM Solves Vanishing Gradients

**關鍵洞察：** 細胞狀態 $C_t$ 的更新是**加法** (Additive) 而非乘法。梯度沿細胞狀態反向傳播時：

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

只要遺忘門 $f_t$ 接近 1（即「記住」模式），梯度就能幾乎不衰減地流過任意長的時間步。這與 ResNet 中的殘差連接 (Residual Connection) 原理相似。

### 4.5 LSTM 的記憶容量 Memory Capacity

| 記憶類型 | 儲存位置 | 更新方式 | 時間尺度 |
|---------|---------|---------|---------|
| 短期記憶 | 隱藏狀態 $h_t$ | 每步完全重算 | 1-10 步 |
| 長期記憶 | 細胞狀態 $C_t$ | 加法更新，門控遺忘 | 可達數百步 |

---

## 5. GRU (Gated Recurrent Unit)

### 5.1 GRU 的動機 Motivation

GRU 由 Cho et al. 於 2014 年提出，是 LSTM 的**簡化版本**。它將 LSTM 的三個門簡化為兩個門，並**合併了細胞狀態與隱藏狀態**，減少了參數量的同時保留了門控記憶的優勢。

### 5.2 GRU 的結構 Architecture

```
              ┌────────────────────────────────────┐
              │           GRU Cell                  │
              │                                    │
  h(t-1) ────┤──→ [σ]  [σ]                        │
              │     z    r                          │
              │     ↓    ↓                          │
              │     z(t) [×] ──→ [tanh] → h̃(t)     │
              │     │         h(t-1)·r(t)           │
              │     ↓                ↓              │
              │   [1-z(t)]        [z(t)]            │
              │     ↓   ↓         ↓   ↓            │
              │     [×] h(t-1)   [×] h̃(t)          │
              │       ↓            ↓                │
              │       └───[+]──────┘                │
              │            ↓                        │
              │          h(t) ──────────────────────→ h(t)
              └────────────────────────────────────┘
                            ↑
                           x(t)
```

### 5.3 GRU 的數學公式 Mathematical Formulation

#### 重置門 Reset Gate ($r_t$)

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

- 決定要「忽略」多少過去的隱藏狀態
- $r_t \approx 0$：幾乎完全忽略過去（類似 LSTM 遺忘門 = 0）
- 讓模型可以「忘記」不相關的歷史資訊

#### 更新門 Update Gate ($z_t$)

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

- 類似 LSTM 的遺忘門和輸入門的**結合版**
- $z_t$ 同時控制保留多少舊狀態和接收多少新狀態
- $z_t \approx 1$：保留舊狀態（記住）
- $z_t \approx 0$：採用新候選狀態（更新）

#### 候選隱藏狀態 Candidate Hidden State ($\tilde{h}_t$)

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

#### 最終隱藏狀態 Final Hidden State

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 5.4 LSTM vs GRU 比較 Comparison

| 特性 | LSTM | GRU |
|------|------|-----|
| 門控數量 | 3（遺忘、輸入、輸出） | 2（重置、更新） |
| 狀態變數 | 2（$h_t$ 和 $C_t$） | 1（$h_t$） |
| 參數量 | 較多（約 4 倍隱藏維度） | 較少（約 3 倍隱藏維度） |
| 長程依賴 | 優秀 | 良好 |
| 訓練速度 | 較慢 | 較快 |
| 適用場景 | 需要精細記憶控制 | 資料量較少或速度優先 |
| 提出年份 | 1997 | 2014 |

> **經驗法則：** 在大多數任務上，LSTM 和 GRU 的效能差異不大。GRU 因為參數更少，在小資料集上可能更不容易過擬合。建議兩者都嘗試，選擇表現較好的那個。

---

## 6. 雙向 RNN (Bidirectional RNN)

### 6.1 動機 Motivation

標準 RNN 只能利用**過去**的上下文（從左到右），但許多任務需要同時考慮**未來**的上下文。

**範例：** 填入空白 "I am _____ happy because I passed the exam."
- 需要看到 "because I passed the exam" 才能確定填入 "very" 或 "so"

### 6.2 架構 Architecture

雙向 RNN 包含兩個獨立的 RNN，一個正向 (Forward)、一個反向 (Backward)：

```
正向 RNN (Forward):   h→₁ ──→ h→₂ ──→ h→₃ ──→ h→₄
                        ↑       ↑       ↑       ↑
                       x₁      x₂      x₃      x₄
                        ↓       ↓       ↓       ↓
反向 RNN (Backward):  h←₁ ←── h←₂ ←── h←₃ ←── h←₄

輸出:                 [h→₁;h←₁] [h→₂;h←₂] [h→₃;h←₃] [h→₄;h←₄]
                        ↓         ↓         ↓         ↓
                       y₁        y₂        y₃        y₄
```

$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

- 每個時間步的表示是正向和反向隱藏狀態的**拼接 (Concatenation)**
- 輸出維度是單向 RNN 的兩倍
- 可以用 LSTM 或 GRU 作為基礎單元

### 6.3 適用與限制 Applicability and Limitations

| 適用場景 | 不適用場景 |
|---------|-----------|
| 文本分類 (Text Classification) | 即時語音辨識 (Real-time ASR) |
| 命名實體辨識 (NER) | 線上時間序列預測 |
| 機器翻譯的 Encoder | 任何需要因果推斷的任務 |
| 情感分析 (Sentiment Analysis) | 自迴歸語言模型 |

> **限制：** 雙向 RNN 需要完整的序列才能運算，因此無法用於需要即時輸出的任務（無法看到「未來」資訊）。

---

## 7. 注意力機制 (Attention Mechanism) 的動機

### 7.1 Seq2Seq 的瓶頸 The Bottleneck of Seq2Seq

在傳統的 Encoder-Decoder (Seq2Seq) 架構中，Encoder 將整個輸入序列壓縮成一個**固定長度的上下文向量 (Context Vector)**：

```
Encoder:   x₁ → x₂ → x₃ → x₄ → [context vector c]
                                         │
Decoder:                                 ├→ y₁ → y₂ → y₃
```

**問題：**
- 所有輸入資訊必須壓縮到一個固定維度的向量中
- 序列越長，資訊損失越嚴重 -- 稱為**資訊瓶頸 (Information Bottleneck)**
- 實驗證明：輸入序列超過 20-30 個 token 時，翻譯品質急劇下降

### 7.2 注意力機制的核心想法 Core Idea of Attention

> **核心直覺：** 與其把所有資訊壓成一個向量，不如讓 Decoder 在每個時間步**回頭看** Encoder 的所有隱藏狀態，並根據當前需求**選擇性地關注 (Attend to)** 最相關的部分。

```
Encoder 隱藏狀態:    h₁    h₂    h₃    h₄
                     │     │     │     │
注意力權重:         0.1   0.7   0.15  0.05   (加總 = 1.0)
                     │     │     │     │
加權求和:           ──→  context = Σ αᵢ · hᵢ
                              │
Decoder:           s(t-1) + context → s(t) → y(t)
```

### 7.3 注意力的計算步驟 Computation Steps

**Bahdanau Attention (2014)：**

1. **計算對齊分數 (Alignment Score)：**
   $$e_{t,i} = \text{score}(s_{t-1}, h_i)$$
   其中 $s_{t-1}$ 是 Decoder 的前一時間步隱藏狀態，$h_i$ 是 Encoder 第 $i$ 個隱藏狀態

2. **正規化為注意力權重 (Attention Weights)：**
   $$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x} \exp(e_{t,j})} = \text{softmax}(e_t)_i$$

3. **計算上下文向量 (Context Vector)：**
   $$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} \cdot h_i$$

4. **結合上下文與 Decoder 狀態：**
   $$s_t = f(s_{t-1}, y_{t-1}, c_t)$$

### 7.4 注意力分數的計算方式 Score Functions

| 名稱 | 公式 | 特點 |
|------|------|------|
| 加法注意力 (Additive) | $v^T \tanh(W_1 s + W_2 h)$ | Bahdanau 原始版本，表達力強 |
| 乘法注意力 (Multiplicative) | $s^T W h$ | Luong 提出，計算更高效 |
| 點積注意力 (Dot-Product) | $s^T h$ | 最簡單，但要求維度匹配 |
| 縮放點積 (Scaled Dot-Product) | $\frac{s^T h}{\sqrt{d_k}}$ | Transformer 使用，避免大維度的 softmax 飽和 |

---

## 8. Transformer 架構概述

### 8.1 "Attention Is All You Need"

2017 年，Vaswani et al. 提出了 Transformer 架構，其革命性的主張是：**完全捨棄遞迴結構 (Recurrence)**，僅使用注意力機制來處理序列資料。

**Transformer 解決了 RNN 的三大根本限制：**

| RNN 的限制 | Transformer 的解決方案 |
|-----------|---------------------|
| 無法平行化（必須按時間步序計算） | Self-Attention 可以**完全平行化** |
| 長程依賴靠梯度長距離傳播 | 任意兩個位置之間的路徑長度為 $O(1)$ |
| 固定大小的隱藏狀態瓶頸 | 動態關注整個序列 |

### 8.2 Self-Attention 自注意力機制

Self-Attention 的核心想法是讓序列中的**每個位置**都能直接關注序列中的**所有其他位置**，從而捕捉全局依賴關係。

```svg
<figure class="md-figure">
<svg viewBox="0 0 640 440" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Self-Attention 權重熱圖示意">
  <rect x="0" y="0" width="640" height="440" fill="#ffffff"/>
  <defs>
    <linearGradient id="attnGrad" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stop-color="#eff6ff"/>
      <stop offset="50%" stop-color="#60a5fa"/>
      <stop offset="100%" stop-color="#1e3a8a"/>
    </linearGradient>
  </defs>
  <text x="320" y="28" text-anchor="middle" font-size="14" fill="#111827" font-weight="600">Self-Attention 權重矩陣：句子 "The cat sat on the mat ."</text>
  <!-- Tokens along top (Key) -->
  <g font-size="12" fill="#111827" font-family="serif" text-anchor="middle">
    <text x="140" y="70">The</text><text x="195" y="70">cat</text><text x="250" y="70">sat</text>
    <text x="305" y="70">on</text><text x="360" y="70">the</text><text x="415" y="70">mat</text><text x="470" y="70">.</text>
  </g>
  <text x="305" y="52" text-anchor="middle" font-size="11" fill="#6b7280">Key ↑（被關注的位置）</text>
  <!-- Tokens along left (Query) -->
  <g font-size="12" fill="#111827" font-family="serif" text-anchor="end">
    <text x="108" y="105">The</text><text x="108" y="160">cat</text><text x="108" y="215">sat</text>
    <text x="108" y="270">on</text><text x="108" y="325">the</text><text x="108" y="380">mat</text><text x="108" y="430">.</text>
  </g>
  <text x="60" y="250" text-anchor="middle" font-size="11" fill="#6b7280" transform="rotate(-90 60 250)">Query（發出查詢的位置）</text>
  <!-- Grid 7x7 with varying opacity simulating softmax weights -->
  <g stroke="#d1d5db" stroke-width="0.5">
    <!-- Row 1: The -->
    <rect x="115" y="82" width="50" height="40" fill="#1e3a8a" opacity="0.85"/>
    <rect x="170" y="82" width="50" height="40" fill="#60a5fa" opacity="0.35"/>
    <rect x="225" y="82" width="50" height="40" fill="#60a5fa" opacity="0.15"/>
    <rect x="280" y="82" width="50" height="40" fill="#60a5fa" opacity="0.1"/>
    <rect x="335" y="82" width="50" height="40" fill="#60a5fa" opacity="0.55"/>
    <rect x="390" y="82" width="50" height="40" fill="#60a5fa" opacity="0.2"/>
    <rect x="445" y="82" width="50" height="40" fill="#60a5fa" opacity="0.05"/>
    <!-- Row 2: cat -->
    <rect x="115" y="137" width="50" height="40" fill="#60a5fa" opacity="0.6"/>
    <rect x="170" y="137" width="50" height="40" fill="#1e3a8a" opacity="0.9"/>
    <rect x="225" y="137" width="50" height="40" fill="#60a5fa" opacity="0.55"/>
    <rect x="280" y="137" width="50" height="40" fill="#60a5fa" opacity="0.15"/>
    <rect x="335" y="137" width="50" height="40" fill="#60a5fa" opacity="0.1"/>
    <rect x="390" y="137" width="50" height="40" fill="#60a5fa" opacity="0.2"/>
    <rect x="445" y="137" width="50" height="40" fill="#60a5fa" opacity="0.05"/>
    <!-- Row 3: sat -->
    <rect x="115" y="192" width="50" height="40" fill="#60a5fa" opacity="0.15"/>
    <rect x="170" y="192" width="50" height="40" fill="#1e3a8a" opacity="0.7"/>
    <rect x="225" y="192" width="50" height="40" fill="#1e3a8a" opacity="0.85"/>
    <rect x="280" y="192" width="50" height="40" fill="#60a5fa" opacity="0.5"/>
    <rect x="335" y="192" width="50" height="40" fill="#60a5fa" opacity="0.15"/>
    <rect x="390" y="192" width="50" height="40" fill="#60a5fa" opacity="0.35"/>
    <rect x="445" y="192" width="50" height="40" fill="#60a5fa" opacity="0.05"/>
    <!-- Row 4: on -->
    <rect x="115" y="247" width="50" height="40" fill="#60a5fa" opacity="0.1"/>
    <rect x="170" y="247" width="50" height="40" fill="#60a5fa" opacity="0.3"/>
    <rect x="225" y="247" width="50" height="40" fill="#60a5fa" opacity="0.55"/>
    <rect x="280" y="247" width="50" height="40" fill="#1e3a8a" opacity="0.8"/>
    <rect x="335" y="247" width="50" height="40" fill="#60a5fa" opacity="0.2"/>
    <rect x="390" y="247" width="50" height="40" fill="#60a5fa" opacity="0.55"/>
    <rect x="445" y="247" width="50" height="40" fill="#60a5fa" opacity="0.05"/>
    <!-- Row 5: the -->
    <rect x="115" y="302" width="50" height="40" fill="#60a5fa" opacity="0.55"/>
    <rect x="170" y="302" width="50" height="40" fill="#60a5fa" opacity="0.15"/>
    <rect x="225" y="302" width="50" height="40" fill="#60a5fa" opacity="0.1"/>
    <rect x="280" y="302" width="50" height="40" fill="#60a5fa" opacity="0.2"/>
    <rect x="335" y="302" width="50" height="40" fill="#1e3a8a" opacity="0.8"/>
    <rect x="390" y="302" width="50" height="40" fill="#60a5fa" opacity="0.7"/>
    <rect x="445" y="302" width="50" height="40" fill="#60a5fa" opacity="0.05"/>
    <!-- Row 6: mat -->
    <rect x="115" y="357" width="50" height="40" fill="#60a5fa" opacity="0.25"/>
    <rect x="170" y="357" width="50" height="40" fill="#60a5fa" opacity="0.3"/>
    <rect x="225" y="357" width="50" height="40" fill="#60a5fa" opacity="0.4"/>
    <rect x="280" y="357" width="50" height="40" fill="#60a5fa" opacity="0.55"/>
    <rect x="335" y="357" width="50" height="40" fill="#60a5fa" opacity="0.65"/>
    <rect x="390" y="357" width="50" height="40" fill="#1e3a8a" opacity="0.85"/>
    <rect x="445" y="357" width="50" height="40" fill="#60a5fa" opacity="0.1"/>
    <!-- Row 7: . -->
    <rect x="115" y="412" width="50" height="20" fill="#60a5fa" opacity="0.12"/>
    <rect x="170" y="412" width="50" height="20" fill="#60a5fa" opacity="0.1"/>
    <rect x="225" y="412" width="50" height="20" fill="#60a5fa" opacity="0.15"/>
    <rect x="280" y="412" width="50" height="20" fill="#60a5fa" opacity="0.1"/>
    <rect x="335" y="412" width="50" height="20" fill="#60a5fa" opacity="0.15"/>
    <rect x="390" y="412" width="50" height="20" fill="#60a5fa" opacity="0.5"/>
    <rect x="445" y="412" width="50" height="20" fill="#1e3a8a" opacity="0.7"/>
  </g>
  <!-- Color scale on right -->
  <rect x="520" y="82" width="14" height="350" fill="url(#attnGrad)" stroke="#d1d5db"/>
  <text x="542" y="92" font-size="10" fill="#111827">高</text>
  <text x="542" y="260" font-size="10" fill="#6b7280">權重</text>
  <text x="542" y="430" font-size="10" fill="#111827">低</text>
</svg>
<figcaption>示意圖：Self-Attention 權重熱圖。每一列是某個 Query token 對所有 Key token 的 softmax 注意力分布（每列和為 1）。對角線通常最深代表 token 關注自己；"sat" 同時關注 "cat"（主詞）與自己；"mat" 關注 "the"（冠詞）與自己，反映句法依存。實務中 Multi-Head Attention 會並行多個這種熱圖，每個 head 學到不同的語言模式。</figcaption>
</figure>
```

#### Query-Key-Value (QKV) 框架

受資訊檢索啟發，Self-Attention 將每個輸入向量線性投影為三個角色：

- **Query (Q, 查詢)**：「我在找什麼？」 -- 代表當前位置的查詢需求
- **Key (K, 鍵)**：「我有什麼？」 -- 代表每個位置可以提供的索引資訊
- **Value (V, 值)**：「我的內容是什麼？」 -- 代表每個位置的實際內容

```
輸入 X = [x₁, x₂, x₃, ..., xₙ]
          │    │    │         │
          ↓    ↓    ↓         ↓
  Q = XWQ  K = XWK  V = XWV

  Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

#### 計算步驟 Step-by-Step

1. **線性投影：** $Q = XW^Q, \quad K = XW^K, \quad V = XW^V$
2. **計算注意力分數：** $\text{scores} = \frac{QK^T}{\sqrt{d_k}}$
   - $d_k$ 是 Key 的維度，除以 $\sqrt{d_k}$ 是為了防止分數過大導致 softmax 梯度消失
3. **Softmax 正規化：** $\text{weights} = \text{softmax}(\text{scores})$
4. **加權求和：** $\text{output} = \text{weights} \cdot V$

#### 直覺範例 Intuitive Example

句子："The **cat** sat on the **mat** because **it** was tired."

計算 "it" 的 Self-Attention：
```
         The    cat    sat    on    the    mat   because   it    was   tired
  it →  [0.02] [0.45] [0.03] [0.01] [0.01] [0.08] [0.02] [0.10] [0.05] [0.23]
                 ↑↑                                                       ↑
           "it" 主要關注 "cat"                                      也關注 "tired"
```

模型學會了 "it" 最可能指代 "cat"，因此給予 "cat" 最高的注意力權重。

### 8.3 Multi-Head Attention 多頭注意力

**動機：** 單一的注意力頭只能學習一種注意力模式。但語言中存在多種不同的關係（語法、語義、指代等），因此使用**多個注意力頭**，讓每個頭專注於不同類型的關係。

```
Multi-Head Attention:

輸入 X ──┬──→ Head 1 (Q₁, K₁, V₁) → 語法依賴
         ├──→ Head 2 (Q₂, K₂, V₂) → 語義相似
         ├──→ Head 3 (Q₃, K₃, V₃) → 位置接近
         ├──→ ...
         └──→ Head h (Qₕ, Kₕ, Vₕ) → 指代關係
                       │
                       ↓
              [Concat] → WO → 輸出
```

**數學公式：**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- 假設 $d_{model} = 512$，$h = 8$ 個頭
- 每個頭的維度 $d_k = d_v = d_{model} / h = 64$
- 計算量與單頭 $d_{model}$ 維度的注意力相當，但表達能力更強

---

## 9. 位置編碼 (Positional Encoding)

### 9.1 為什麼需要位置編碼？ Why Positional Encoding?

Self-Attention 是**置換不變的 (Permutation Invariant)**：打亂輸入順序，輸出也只是跟著打亂，注意力權重完全相同。這意味著 Transformer **天生不知道序列順序**。

但語言明顯是有序的：
- "狗咬人" ≠ "人咬狗"
- "not bad" ≠ "bad not"

因此必須**人為注入位置資訊**。

### 9.2 正弦位置編碼 Sinusoidal Positional Encoding

Vaswani et al. 使用正弦/餘弦函數生成位置編碼：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- $pos$：位置索引（0, 1, 2, ...）
- $i$：維度索引
- $d_{model}$：模型維度

**設計巧思：**
- 不同維度使用不同頻率的正弦波，低維度頻率高（捕捉近距離關係），高維度頻率低（捕捉遠距離關係）
- 對於任意固定的位移 $k$，$PE_{pos+k}$ 可以表示為 $PE_{pos}$ 的線性函數，使模型能學到相對位置關係
- 可以外推 (Extrapolate) 到訓練時未見過的序列長度

### 9.3 位置編碼的使用方式

$$\text{Input} = \text{Token Embedding} + \text{Positional Encoding}$$

位置編碼向量與詞嵌入向量**逐元素相加**，作為 Transformer 的最終輸入。

```
Token:       "The"    "cat"    "sat"
Embedding:   e₁       e₂       e₃        (學習的詞嵌入)
Position:    PE₀      PE₁      PE₂       (固定的位置編碼)
Input:       e₁+PE₀   e₂+PE₁   e₃+PE₂   (相加後作為輸入)
```

---

## 10. Encoder-Decoder 架構

### 10.1 Transformer 的完整架構 Complete Architecture

```
┌─────────────── Transformer ───────────────┐
│                                           │
│   Encoder (×N)            Decoder (×N)    │
│  ┌──────────────┐       ┌──────────────┐  │
│  │              │       │              │  │
│  │ Feed-Forward │       │ Feed-Forward │  │
│  │ ↑ Add & Norm │       │ ↑ Add & Norm │  │
│  │              │       │              │  │
│  │ Self-Attn    │       │ Cross-Attn   │  │
│  │ ↑ Add & Norm │       │ ↑ Add & Norm │←─│── Encoder 輸出
│  │              │       │              │  │
│  │              │       │Masked Self-Attn│ │
│  │              │       │ ↑ Add & Norm │  │
│  └──────┬───────┘       └──────┬───────┘  │
│         │                      │          │
│   Positional Enc.        Positional Enc.  │
│   + Embedding            + Embedding      │
│         ↑                      ↑          │
│      Input               Output (shifted) │
│    (源語言)               (目標語言)        │
└───────────────────────────────────────────┘
```

### 10.2 各組件說明 Component Description

#### Encoder

- **N 個相同的層堆疊**（原始論文 N=6）
- 每層包含：
  1. **Multi-Head Self-Attention**：讓每個位置關注輸入序列的所有位置
  2. **Feed-Forward Network (FFN)**：兩層全連接，中間用 ReLU，$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
  3. **殘差連接 (Residual Connection)**：$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$

#### Decoder

- 同樣 **N 個相同的層堆疊**
- 每層包含：
  1. **Masked Multi-Head Self-Attention**：加入因果遮罩 (Causal Mask)，確保位置 $i$ 只能看到位置 $\leq i$ 的資訊（防止作弊看到未來的答案）
  2. **Cross-Attention (Encoder-Decoder Attention)**：Query 來自 Decoder，Key 和 Value 來自 Encoder 輸出
  3. **Feed-Forward Network**

#### Add & Norm（殘差連接 + 層正規化）

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

- 殘差連接幫助梯度流動（類似 ResNet）
- 層正規化 (Layer Normalization) 穩定訓練

### 10.3 Masked Attention 遮罩注意力

在 Decoder 中，為了保持自迴歸 (Autoregressive) 特性，必須確保生成第 $t$ 個 token 時只能看到前 $t-1$ 個 token：

```
注意力矩陣（解碼 "I am happy" 時）：

            I    am   happy
    I     [ 1    -∞    -∞  ]     只能看到自己
    am    [ 0.4  0.6   -∞  ]     可以看到 "I" 和自己
    happy [ 0.2  0.3   0.5 ]     可以看到前面所有 token

    -∞ 表示 mask（softmax 後會變成 0）
```

---

## 11. 從 RNN 到 Transformer 的演進

### 11.1 技術演進時間線 Timeline

```
1990 ── Elman Network (Simple RNN)
         │
1997 ── LSTM (Hochreiter & Schmidhuber)
         │   解決梯度消失，能捕捉長程依賴
         │
2014 ── GRU (Cho et al.)
         │   簡化 LSTM，參數更少
         │
2014 ── Seq2Seq with Attention (Bahdanau et al.)
         │   引入注意力機制，突破固定向量瓶頸
         │
2015 ── Attention-based Models 百花齊放
         │   各種注意力變體
         │
2017 ── Transformer (Vaswani et al.)
         │   "Attention Is All You Need"
         │   完全拋棄遞迴，全注意力架構
         │
2018 ── GPT-1 (OpenAI) / BERT (Google)
         │   大規模預訓練語言模型
         │
2019+ ── GPT-2/3/4, T5, PaLM...
         │   規模不斷擴大，能力躍升
         │
2023+ ── 多模態 Transformer（文字+圖片+音訊+影片）
```

### 11.2 關鍵比較 Key Comparison

| 特性 | Vanilla RNN | LSTM/GRU | Transformer |
|------|------------|----------|-------------|
| 長程依賴 | 差 | 良好 | 優秀 |
| 平行化能力 | 無 | 無 | 完全平行 |
| 計算複雜度（序列長度 n） | $O(n)$ | $O(n)$ | $O(n^2)$* |
| 最大路徑長度 | $O(n)$ | $O(n)$ | $O(1)$ |
| 記憶機制 | 隱藏狀態 | 門控 + 細胞狀態 | 全局注意力 |
| 參數效率 | 高 | 中等 | 需要更多資料 |
| 歸納偏置 | 強（順序） | 強（順序） | 弱（需要位置編碼） |

*$O(n^2)$ 是因為每個位置要與所有位置計算注意力分數，這也是 Transformer 在極長序列上的挑戰。

### 11.3 RNN 依然有用嗎？ Are RNNs Still Relevant?

雖然 Transformer 在大多數 NLP 任務上已經取代了 RNN，但 RNN/LSTM 在某些場景仍有優勢：

- **計算資源受限**的邊緣設備 (Edge Devices)
- **極長序列**（Transformer 的 $O(n^2)$ 記憶體需求可能不可行）
- **即時串流處理**（RNN 天然支持逐步處理）
- **小資料集**（RNN 的歸納偏置提供更好的先驗）
- 近年的**狀態空間模型 (State Space Models, SSMs)** 如 Mamba，融合了 RNN 的線性複雜度與 Transformer 的效能

---

## 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 序列資料 | Sequential Data | 具有時間或位置順序依賴的資料 |
| 遞迴神經網路 | Recurrent Neural Network (RNN) | 具有自迴圈連接的神經網路，用於處理序列 |
| 隱藏狀態 | Hidden State | RNN 在每個時間步維護的內部表示向量 |
| 沿時間反向傳播 | Backpropagation Through Time (BPTT) | RNN 的訓練演算法 |
| 梯度消失 | Vanishing Gradient | 梯度在反向傳播中指數衰減的現象 |
| 梯度爆炸 | Exploding Gradient | 梯度在反向傳播中指數增長的現象 |
| 梯度裁剪 | Gradient Clipping | 限制梯度大小以防止梯度爆炸的技術 |
| 長短期記憶 | Long Short-Term Memory (LSTM) | 透過門控機制解決梯度消失的 RNN 變體 |
| 細胞狀態 | Cell State | LSTM 中用於儲存長期記憶的向量 |
| 遺忘門 | Forget Gate | LSTM 中決定遺忘哪些記憶的門控 |
| 輸入門 | Input Gate | LSTM 中決定寫入哪些新資訊的門控 |
| 輸出門 | Output Gate | LSTM 中決定輸出哪些記憶的門控 |
| 門控遞迴單元 | Gated Recurrent Unit (GRU) | LSTM 的簡化版本，使用重置門和更新門 |
| 重置門 | Reset Gate | GRU 中決定忽略多少過去狀態的門控 |
| 更新門 | Update Gate | GRU 中控制新舊狀態混合比例的門控 |
| 雙向 RNN | Bidirectional RNN | 同時考慮正向和反向上下文的 RNN |
| 序列到序列 | Sequence-to-Sequence (Seq2Seq) | 將一個序列轉換為另一個序列的模型架構 |
| 注意力機制 | Attention Mechanism | 讓模型選擇性關注輸入特定部分的技術 |
| 注意力權重 | Attention Weights | 表示對各位置關注程度的正規化分數 |
| 上下文向量 | Context Vector | 注意力加權求和產生的向量 |
| 自注意力 | Self-Attention | 序列內部元素互相關注的注意力機制 |
| 多頭注意力 | Multi-Head Attention | 使用多組注意力頭捕捉不同關係的技術 |
| 查詢/鍵/值 | Query/Key/Value (QKV) | Self-Attention 中的三個投影角色 |
| 縮放點積注意力 | Scaled Dot-Product Attention | Transformer 使用的注意力計算方式 |
| 位置編碼 | Positional Encoding | 為 Transformer 注入序列位置資訊的技術 |
| 編碼器-解碼器 | Encoder-Decoder | 先編碼輸入再解碼輸出的架構模式 |
| 因果遮罩 | Causal Mask | 防止 Decoder 看到未來 token 的遮罩 |
| 殘差連接 | Residual Connection | 將輸入直接加到輸出以幫助梯度流動 |
| 層正規化 | Layer Normalization | 對每層的激活值進行正規化的技術 |
| 參數共享 | Parameter Sharing | 不同時間步使用相同權重矩陣 |
| 長程依賴 | Long-Range Dependency | 序列中相距很遠的元素之間的依賴關係 |
| 狀態空間模型 | State Space Model (SSM) | 結合 RNN 效率與 Transformer 效能的新架構 |

---

## 延伸閱讀 Further Reading

- Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
- Cho, K. et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder."
- Bahdanau, D. et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate."
- Vaswani, A. et al. (2017). "Attention Is All You Need."
- Jay Alammar, "The Illustrated Transformer": https://jalammar.github.io/illustrated-transformer/
- Lilian Weng, "Attention? Attention!": https://lilianweng.github.io/posts/2018-06-24-attention/
- Andrej Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks": http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Christopher Olah, "Understanding LSTM Networks": https://colah.github.io/posts/2015-08-Understanding-LSTMs/
