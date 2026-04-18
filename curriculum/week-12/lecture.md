# 第 12 週：CNN 視覺化（卷積核、特徵圖、CAM/Grad-CAM）
# Week 12: CNN Visualization (Filters, Feature Maps, CAM/Grad-CAM)

## 學習目標 Learning Objectives
1. 理解卷積神經網路 (Convolutional Neural Network, CNN) 的動機與基本結構
2. 掌握卷積運算 (Convolution) 的數學原理與直覺理解
3. 了解卷積核 (Filters/Kernels)、填充 (Padding)、步幅 (Stride)、通道 (Channels) 等核心概念
4. 理解池化層 (Pooling Layer) 的作用與類型
5. 認識經典 CNN 架構：LeNet、AlexNet、VGG、ResNet
6. 掌握 CNN 可視化技術：卷積核視覺化、特徵圖視覺化、CAM、Grad-CAM
7. 了解遷移學習 (Transfer Learning) 的概念與應用

---

## 1. 卷積神經網路的動機 Motivation for CNN

### 1.1 全連接層的問題 Problems with Fully Connected Layers

在第 11 週中，我們學習了全連接神經網路 (Fully Connected Neural Network, FCNN)。然而，當輸入是影像時，FCNN 存在嚴重的問題：

| 問題 | 說明 |
|------|------|
| 參數爆炸 Parameter Explosion | 一張 224x224x3 的影像有 150,528 個輸入。若第一個隱藏層有 1000 個神經元，僅第一層就有 **1.5 億個參數** |
| 忽略空間結構 Ignoring Spatial Structure | 將影像展平 (Flatten) 為一維向量，丟失了像素之間的空間關係 |
| 缺乏平移不變性 No Translation Invariance | 同一物件在影像不同位置會產生完全不同的激活 |

### 1.2 CNN 的核心思想 Core Ideas of CNN

CNN 透過以下三個關鍵特性解決上述問題：

1. **局部連接 (Local Connectivity)**：每個神經元只與輸入的一小塊區域（感受野 Receptive Field）相連，而非全部像素
2. **權重共享 (Weight Sharing)**：同一組卷積核在整張影像上滑動，大幅減少參數量
3. **空間層次 (Spatial Hierarchy)**：淺層學習邊緣 (Edges)，中層學習紋理 (Textures)，深層學習物件部件 (Object Parts)

> "CNN 的關鍵洞見是：影像中的局部特徵（如邊緣、角落）在不同位置具有相似的統計特性，因此可以用同一組濾波器在全圖上偵測。"

### 1.3 CNN 的整體架構 Overall Architecture

```
輸入影像 (Input Image)
  → [卷積層 Conv + 激活 ReLU] × N
  → [池化層 Pooling]
  → [卷積層 Conv + 激活 ReLU] × M
  → [池化層 Pooling]
  → ...
  → 展平 Flatten
  → [全連接層 FC] × K
  → 輸出 Output (分類/回歸)
```

CNN 可以看作兩部分：
- **特徵提取器 (Feature Extractor)**：卷積層 + 池化層，自動學習影像特徵
- **分類器 (Classifier)**：全連接層，根據提取的特徵進行分類

---

## 2. 卷積運算 Convolution Operation

### 2.1 直覺理解 Intuitive Understanding

卷積運算可以想像成一個小窗口（卷積核）在影像上滑動，每到一個位置就與覆蓋區域做加權求和 (Weighted Sum)。這個過程類似於用一個模板在影像中尋找特定的局部模式。

```svg
<figure class="md-figure">
<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="2D 卷積運算滑動示意圖">
  <rect x="0" y="0" width="720" height="300" fill="#ffffff"/>
  <!-- LEFT: input image 5x5 with 3x3 kernel highlighted at position (0,0) -->
  <g transform="translate(20,40)">
    <text x="100" y="-10" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">輸入 Input (5×5)</text>
    <!-- 5x5 grid -->
    <g stroke="#9ca3af" stroke-width="1">
      <!-- cells -->
      <g fill="#ffffff">
        <rect x="0" y="0" width="40" height="40"/><rect x="40" y="0" width="40" height="40"/><rect x="80" y="0" width="40" height="40"/><rect x="120" y="0" width="40" height="40"/><rect x="160" y="0" width="40" height="40"/>
        <rect x="0" y="40" width="40" height="40"/><rect x="40" y="40" width="40" height="40"/><rect x="80" y="40" width="40" height="40"/><rect x="120" y="40" width="40" height="40"/><rect x="160" y="40" width="40" height="40"/>
        <rect x="0" y="80" width="40" height="40"/><rect x="40" y="80" width="40" height="40"/><rect x="80" y="80" width="40" height="40"/><rect x="120" y="80" width="40" height="40"/><rect x="160" y="80" width="40" height="40"/>
        <rect x="0" y="120" width="40" height="40"/><rect x="40" y="120" width="40" height="40"/><rect x="80" y="120" width="40" height="40"/><rect x="120" y="120" width="40" height="40"/><rect x="160" y="120" width="40" height="40"/>
        <rect x="0" y="160" width="40" height="40"/><rect x="40" y="160" width="40" height="40"/><rect x="80" y="160" width="40" height="40"/><rect x="120" y="160" width="40" height="40"/><rect x="160" y="160" width="40" height="40"/>
      </g>
    </g>
    <!-- Kernel overlay: solid box at top-left 3x3 -->
    <rect x="0" y="0" width="120" height="120" fill="#fef08a" opacity="0.55" stroke="#d97706" stroke-width="3"/>
    <!-- Ghost box at one-step-right position (dashed) -->
    <rect x="40" y="0" width="120" height="120" fill="none" stroke="#d97706" stroke-width="1.5" stroke-dasharray="5 3"/>
    <!-- Arrow showing sliding direction -->
    <path d="M 150 60 L 175 60" stroke="#d97706" stroke-width="2" marker-end="url(#slideArr)"/>
    <defs>
      <marker id="slideArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#d97706"/></marker>
    </defs>
    <text x="188" y="64" font-size="11" fill="#b45309">stride=1</text>
    <!-- Sample values in overlapped region -->
    <g font-size="12" fill="#111827" text-anchor="middle">
      <text x="20" y="24">1</text><text x="60" y="24">2</text><text x="100" y="24">3</text><text x="140" y="24">0</text><text x="180" y="24">1</text>
      <text x="20" y="64">0</text><text x="60" y="64">1</text><text x="100" y="64">2</text><text x="140" y="64">3</text><text x="180" y="64">0</text>
      <text x="20" y="104">2</text><text x="60" y="104">1</text><text x="100" y="104">0</text><text x="140" y="104">1</text><text x="180" y="104">2</text>
      <text x="20" y="144">1</text><text x="60" y="144">0</text><text x="100" y="144">1</text><text x="140" y="144">2</text><text x="180" y="144">1</text>
      <text x="20" y="184">0</text><text x="60" y="184">1</text><text x="100" y="184">2</text><text x="140" y="184">0</text><text x="180" y="184">1</text>
    </g>
  </g>
  <!-- Kernel (3x3) in middle -->
  <g transform="translate(280,80)">
    <text x="60" y="-15" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">卷積核 Kernel</text>
    <text x="60" y="-2" text-anchor="middle" font-size="10" fill="#6b7280">(3×3 邊緣偵測)</text>
    <g stroke="#d97706" stroke-width="2" fill="#fef08a">
      <rect x="0" y="0" width="40" height="40"/><rect x="40" y="0" width="40" height="40"/><rect x="80" y="0" width="40" height="40"/>
      <rect x="0" y="40" width="40" height="40"/><rect x="40" y="40" width="40" height="40"/><rect x="80" y="40" width="40" height="40"/>
      <rect x="0" y="80" width="40" height="40"/><rect x="40" y="80" width="40" height="40"/><rect x="80" y="80" width="40" height="40"/>
    </g>
    <g font-size="12" fill="#111827" text-anchor="middle" font-weight="600">
      <text x="20" y="24">-1</text><text x="60" y="24">-1</text><text x="100" y="24">-1</text>
      <text x="20" y="64">-1</text><text x="60" y="64">8</text><text x="100" y="64">-1</text>
      <text x="20" y="104">-1</text><text x="60" y="104">-1</text><text x="100" y="104">-1</text>
    </g>
    <text x="60" y="145" text-anchor="middle" font-size="14" fill="#111827">⊛</text>
  </g>
  <!-- Arrow from input+kernel to output -->
  <g transform="translate(440,150)">
    <path d="M 0 0 L 50 0" stroke="#374151" stroke-width="2" marker-end="url(#convArr)"/>
    <defs>
      <marker id="convArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker>
    </defs>
    <text x="25" y="-8" text-anchor="middle" font-size="11" fill="#6b7280">Σ wx</text>
  </g>
  <!-- RIGHT: output feature map 3x3 -->
  <g transform="translate(520,80)">
    <text x="60" y="-15" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">特徵圖 Feature Map</text>
    <text x="60" y="-2" text-anchor="middle" font-size="10" fill="#6b7280">(3×3)</text>
    <g stroke="#9ca3af" stroke-width="1" fill="#dbeafe">
      <rect x="0" y="0" width="40" height="40"/><rect x="40" y="0" width="40" height="40"/><rect x="80" y="0" width="40" height="40"/>
      <rect x="0" y="40" width="40" height="40"/><rect x="40" y="40" width="40" height="40"/><rect x="80" y="40" width="40" height="40"/>
      <rect x="0" y="80" width="40" height="40"/><rect x="40" y="80" width="40" height="40"/><rect x="80" y="80" width="40" height="40"/>
    </g>
    <!-- First cell highlighted (result of first kernel position) -->
    <rect x="0" y="0" width="40" height="40" fill="#3b82f6" stroke="#1e40af" stroke-width="2"/>
    <g font-size="12" text-anchor="middle">
      <text x="20" y="24" fill="#ffffff" font-weight="600">4</text>
      <text x="60" y="24" fill="#111827">-2</text><text x="100" y="24" fill="#111827">1</text>
      <text x="20" y="64" fill="#111827">0</text><text x="60" y="64" fill="#111827">-3</text><text x="100" y="64" fill="#111827">2</text>
      <text x="20" y="104" fill="#111827">1</text><text x="60" y="104" fill="#111827">-1</text><text x="100" y="104" fill="#111827">0</text>
    </g>
  </g>
</svg>
<figcaption>示意圖：2D 卷積。3×3 卷積核（黃色）在 5×5 輸入上以 stride=1 滑動，每個位置計算 Σ(w·x) 得到輸出特徵圖的一個像素。左側輸入中橘色虛線框代表下一步位置；右側特徵圖左上角的藍色格子對應當前覆蓋區與 kernel 的卷積結果。</figcaption>
</figure>
```

### 2.2 數學定義 Mathematical Definition

對於二維離散卷積 (2D Discrete Convolution)，假設：
- 輸入影像 $I$ 大小為 $H \times W$
- 卷積核 $K$ 大小為 $k_h \times k_w$

輸出特徵圖 (Feature Map) 在位置 $(i, j)$ 的值為：

$$
(I * K)(i, j) = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} I(i + m, j + n) \cdot K(m, n)
$$

> **注意**：嚴格來說，深度學習中的「卷積」其實是**互相關 (Cross-correlation)**，省略了核函數的翻轉步驟。在實務中不影響學習效果，因為卷積核的權重是學習得到的。

### 2.3 卷積運算範例 Convolution Example

假設輸入為 5x5 影像，卷積核為 3x3：

```
輸入 Input (5x5):          卷積核 Kernel (3x3):
┌─────────────────┐        ┌───────────┐
│ 1  0  1  0  1 │        │ 1  0  1 │
│ 0  1  0  1  0 │        │ 0  1  0 │
│ 1  0  1  0  1 │        │ 1  0  1 │
│ 0  1  0  1  0 │        └───────────┘
│ 1  0  1  0  1 │
└─────────────────┘

輸出 Output (3x3):
位置 (0,0): 1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 = 5
...依此類推
```

### 2.4 常見的卷積核效果 Common Kernel Effects

不同的卷積核可以偵測不同的影像特徵：

| 卷積核類型 | 矩陣 | 效果 |
|-----------|------|------|
| 水平邊緣偵測 Horizontal Edge | `[[-1,-1,-1],[0,0,0],[1,1,1]]` | 偵測水平方向的邊緣 |
| 垂直邊緣偵測 Vertical Edge | `[[-1,0,1],[-1,0,1],[-1,0,1]]` | 偵測垂直方向的邊緣 |
| Sobel 濾波器 | `[[-1,0,1],[-2,0,2],[-1,0,1]]` | 偵測垂直邊緣（加權版本） |
| 銳化 Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | 增強影像細節 |
| 高斯模糊 Gaussian Blur | `[[1,2,1],[2,4,2],[1,2,1]] / 16` | 平滑化影像 |

---

## 3. 卷積核的角色 Role of Filters/Kernels

### 3.1 從手工設計到自動學習 From Handcrafted to Learned

傳統影像處理中，卷積核是人工設計的（如 Sobel、Prewitt）。CNN 最大的突破在於：**卷積核的數值是透過反向傳播 (Backpropagation) 自動學習的**。

在訓練過程中：
1. 卷積核初始化為隨機值 (Random Initialization)
2. 前向傳播 (Forward Pass) 計算預測結果
3. 計算損失 (Loss)
4. 反向傳播更新卷積核的權重
5. 經過多次迭代，卷積核「學會」偵測有用的特徵

### 3.2 卷積核的數量與維度 Number and Dimensions

一個卷積層通常包含**多個卷積核**，每個卷積核學習偵測不同的特徵：

```
卷積層參數：
  輸入通道數 (in_channels): C_in
  輸出通道數 (out_channels): C_out（= 卷積核個數）
  卷積核大小 (kernel_size): k × k
  參數總量: C_out × C_in × k × k + C_out（偏置項 Bias）
```

例如，一個卷積層接收 3 通道（RGB）輸入，使用 64 個 3x3 卷積核：
- 參數量 = 64 × 3 × 3 × 3 + 64 = 1,792

### 3.3 多通道卷積 Multi-channel Convolution

當輸入有多個通道（如 RGB 三通道）時，每個卷積核的深度必須與輸入通道數相同：

```
輸入：H × W × C_in
一個卷積核：k × k × C_in
輸出的一個通道：(H-k+1) × (W-k+1) × 1

使用 C_out 個卷積核
→ 輸出：(H-k+1) × (W-k+1) × C_out
```

---

## 4. 填充、步幅與通道 Padding, Stride & Channels

### 4.1 填充 Padding

不使用填充時，每次卷積都會縮小特徵圖尺寸。填充在輸入邊緣補零 (Zero Padding)，以控制輸出尺寸。

| 填充類型 | 說明 | 輸出尺寸 |
|---------|------|---------|
| Valid (無填充) | 不補零 | 縮小 |
| Same (同尺寸) | 補零使輸出大小與輸入相同 | 不變 |
| Full | 補零使所有輸入元素都被完整覆蓋 | 放大 |

**Same Padding 的填充量**：對於 k×k 的卷積核，每邊填充 $p = \lfloor k/2 \rfloor$ 個零。

例如：3×3 卷積核 → 每邊填充 1；5×5 卷積核 → 每邊填充 2。

### 4.2 步幅 Stride

步幅控制卷積核每次滑動的像素數。預設步幅為 1，增大步幅可以降低特徵圖的解析度。

```
步幅 = 1：逐步滑動，輸出較大
步幅 = 2：跳一步滑動，輸出約為原來的一半
```

### 4.3 輸出尺寸公式 Output Size Formula

$$
O = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
$$

其中：
- $H$：輸入高度（或寬度）
- $p$：填充量 (Padding)
- $k$：卷積核大小 (Kernel Size)
- $s$：步幅 (Stride)

**範例**：輸入 32×32，卷積核 3×3，padding=1，stride=1
$$
O = \left\lfloor \frac{32 + 2 \times 1 - 3}{1} \right\rfloor + 1 = 32
$$

### 4.4 通道的概念 Channel Concept

通道 (Channels) 在不同階段有不同含義：
- **輸入層**：對應影像的色彩通道（RGB=3、灰階=1）
- **隱藏層**：對應不同卷積核學到的特徵（也稱 Feature Maps）
- 隨著網路加深，通道數通常增加（如 64 → 128 → 256 → 512），而空間尺寸逐漸縮小

---

## 5. 池化層 Pooling Layer

### 5.1 池化的目的 Purpose of Pooling

池化層的主要功能：
1. **降維 (Downsampling)**：減少特徵圖的空間尺寸，降低計算量
2. **增強魯棒性 (Robustness)**：對微小的平移具有不變性 (Translation Invariance)
3. **擴大感受野 (Receptive Field)**：讓後續卷積層能「看到」更大範圍的輸入

### 5.2 Max Pooling（最大池化）

取池化窗口中的最大值。這是最常用的池化方式。

```svg
<figure class="md-figure">
<svg viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Max Pooling vs Average Pooling 示意圖">
  <rect x="0" y="0" width="680" height="280" fill="#ffffff"/>
  <!-- INPUT 4x4 (shared reference) -->
  <g transform="translate(20,40)">
    <text x="80" y="-12" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">輸入特徵圖 (4×4)</text>
    <g stroke="#9ca3af" stroke-width="1" fill="#ffffff">
      <rect x="0" y="0" width="40" height="40"/><rect x="40" y="0" width="40" height="40"/><rect x="80" y="0" width="40" height="40"/><rect x="120" y="0" width="40" height="40"/>
      <rect x="0" y="40" width="40" height="40"/><rect x="40" y="40" width="40" height="40"/><rect x="80" y="40" width="40" height="40"/><rect x="120" y="40" width="40" height="40"/>
      <rect x="0" y="80" width="40" height="40"/><rect x="40" y="80" width="40" height="40"/><rect x="80" y="80" width="40" height="40"/><rect x="120" y="80" width="40" height="40"/>
      <rect x="0" y="120" width="40" height="40"/><rect x="40" y="120" width="40" height="40"/><rect x="80" y="120" width="40" height="40"/><rect x="120" y="120" width="40" height="40"/>
    </g>
    <!-- 2x2 window partitions (colored backgrounds) -->
    <rect x="0" y="0" width="80" height="80" fill="#dbeafe" opacity="0.5"/>
    <rect x="80" y="0" width="80" height="80" fill="#fef3c7" opacity="0.5"/>
    <rect x="0" y="80" width="80" height="80" fill="#fce7f3" opacity="0.5"/>
    <rect x="80" y="80" width="80" height="80" fill="#d1fae5" opacity="0.5"/>
    <!-- Values -->
    <g font-size="13" fill="#111827" text-anchor="middle">
      <text x="20" y="25">1</text><text x="60" y="25">3</text><text x="100" y="25">2</text><text x="140" y="25">1</text>
      <text x="20" y="65" font-weight="600" fill="#1e40af">5</text><text x="60" y="65">2</text><text x="100" y="65">1</text><text x="140" y="65" font-weight="600" fill="#b45309">3</text>
      <text x="20" y="105">4</text><text x="60" y="105" font-weight="600" fill="#be185d">8</text><text x="100" y="105" font-weight="600" fill="#047857">6</text><text x="140" y="105">2</text>
      <text x="20" y="145">3</text><text x="60" y="145">1</text><text x="100" y="145">4</text><text x="140" y="145">5</text>
    </g>
  </g>
  <!-- Max Pool output (top-right) -->
  <g transform="translate(280,40)">
    <text x="40" y="-12" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">Max Pool (2×2)</text>
    <g stroke="#6b7280" stroke-width="1.5">
      <rect x="0" y="0" width="40" height="40" fill="#dbeafe"/>
      <rect x="40" y="0" width="40" height="40" fill="#fef3c7"/>
      <rect x="0" y="40" width="40" height="40" fill="#fce7f3"/>
      <rect x="40" y="40" width="40" height="40" fill="#d1fae5"/>
    </g>
    <g font-size="15" fill="#111827" text-anchor="middle" font-weight="700">
      <text x="20" y="26" fill="#1e40af">5</text>
      <text x="60" y="26" fill="#b45309">3</text>
      <text x="20" y="66" fill="#be185d">8</text>
      <text x="60" y="66" fill="#047857">6</text>
    </g>
    <text x="40" y="100" text-anchor="middle" font-size="10" fill="#6b7280">每格取窗口最大值</text>
    <text x="40" y="116" text-anchor="middle" font-size="10" fill="#6b7280">保留最強激活</text>
  </g>
  <!-- Avg Pool output -->
  <g transform="translate(420,40)">
    <text x="40" y="-12" text-anchor="middle" font-size="13" fill="#111827" font-weight="600">Avg Pool (2×2)</text>
    <g stroke="#6b7280" stroke-width="1.5">
      <rect x="0" y="0" width="40" height="40" fill="#dbeafe"/>
      <rect x="40" y="0" width="40" height="40" fill="#fef3c7"/>
      <rect x="0" y="40" width="40" height="40" fill="#fce7f3"/>
      <rect x="40" y="40" width="40" height="40" fill="#d1fae5"/>
    </g>
    <g font-size="12" fill="#111827" text-anchor="middle" font-weight="600">
      <text x="20" y="26">2.75</text>
      <text x="60" y="26">1.75</text>
      <text x="20" y="66">4.00</text>
      <text x="60" y="66">4.25</text>
    </g>
    <text x="40" y="100" text-anchor="middle" font-size="10" fill="#6b7280">每格取窗口平均值</text>
    <text x="40" y="116" text-anchor="middle" font-size="10" fill="#6b7280">平滑化輸出</text>
  </g>
  <!-- Arrows -->
  <g stroke="#374151" stroke-width="1.5" fill="none">
    <path d="M 200 80 L 280 60" marker-end="url(#poolArr)"/>
    <path d="M 200 100 L 420 60" marker-end="url(#poolArr)"/>
  </g>
  <defs>
    <marker id="poolArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker>
  </defs>
  <!-- Footer note -->
  <text x="340" y="240" text-anchor="middle" font-size="11" fill="#6b7280">池化 (Pooling) 降低空間維度、擴大感受野、並對微小平移保持不變（translation invariance）</text>
</svg>
<figcaption>示意圖：Max vs Average Pooling。輸入 4×4 被分成四個 2×2 色塊；Max Pooling 取每塊最大值（5, 3, 8, 6）保留最強激活；Average Pooling 取每塊平均值得到較平滑但保留整體強度的表徵。兩者皆將空間維度降為原 1/4。</figcaption>
</figure>
```

**直覺**：Max Pooling 保留最強的特徵激活，相當於保留「最像某特徵」的回應。

### 5.3 Average Pooling（平均池化）

取池化窗口中的平均值。常用於網路末端（Global Average Pooling, GAP）。

```
輸入 (4×4):              Avg Pool (2×2, stride=2):
┌──────────────┐          ┌──────────┐
│ 1  3  2  1 │          │ 2.75  1.75 │
│ 5  2  1  3 │   →      │ 4.00  4.25 │
│ 4  8  6  2 │          └──────────┘
│ 3  1  4  5 │
└──────────────┘
```

### 5.4 Global Average Pooling (GAP)

對每個通道的整張特徵圖取平均，輸出為 1×1×C。GAP 常被用來取代全連接層，大幅減少參數量。

```
輸入：H × W × C
GAP 輸出：1 × 1 × C（即 C 維向量）
```

ResNet、GoogLeNet 等現代架構普遍採用 GAP。

### 5.5 Max Pooling vs Average Pooling

| 特性 | Max Pooling | Average Pooling |
|------|------------|----------------|
| 保留資訊 | 最強激活 | 平均特徵 |
| 常見位置 | 隱藏層之間 | 網路末端 (GAP) |
| 梯度特性 | 僅最大值位置有梯度 | 所有位置均分梯度 |
| 適用場景 | 紋理/邊緣偵測 | 全局特徵摘要 |

---

## 6. 特徵圖的意義 Meaning of Feature Maps

### 6.1 什麼是特徵圖？ What is a Feature Map?

特徵圖是卷積核在輸入上滑動後產生的二維輸出。每個卷積核產生一張特徵圖，代表輸入中某種特定模式的空間分布。

```
輸入影像 → 卷積核_1 → 特徵圖_1（偵測水平邊緣）
         → 卷積核_2 → 特徵圖_2（偵測垂直邊緣）
         → 卷積核_3 → 特徵圖_3（偵測對角線）
         ...
         → 卷積核_N → 特徵圖_N
```

### 6.2 不同深度的特徵圖 Feature Maps at Different Depths

CNN 的一個重要特性是**層次化特徵學習 (Hierarchical Feature Learning)**：

| 層級 | 學習的特徵 | 例子 |
|------|---------|------|
| 淺層 (Layer 1-2) | 低階特徵 Low-level Features | 邊緣 (Edges)、顏色梯度 (Color Gradients) |
| 中層 (Layer 3-4) | 中階特徵 Mid-level Features | 紋理 (Textures)、形狀片段 (Shape Parts) |
| 深層 (Layer 5+) | 高階特徵 High-level Features | 物件部件 (Object Parts)、語義概念 (Semantic Concepts) |

> 這就是為什麼深度學習被稱為「表徵學習 (Representation Learning)」—— 網路自動學習從低階到高階的特徵表示。

### 6.3 感受野 Receptive Field

感受野是指特徵圖上一個元素「看到」的原始輸入區域大小。隨著層數加深，感受野會逐漸增大：

- 第 1 層（3x3 卷積）：感受野 = 3x3
- 第 2 層（3x3 卷積）：感受野 = 5x5
- 第 3 層（3x3 卷積）：感受野 = 7x7

這就是為什麼深層能學習更「全局」的特徵。

---

## 7. 經典架構簡介 Classic CNN Architectures

### 7.1 LeNet-5 (1998, Yann LeCun)

CNN 的先驅架構，用於手寫數字辨識 (MNIST)。

```
Input(32×32×1) → Conv5×5(6) → Pool → Conv5×5(16) → Pool → FC(120) → FC(84) → Output(10)
```

- 參數量：約 60K
- 貢獻：證明了卷積網路在影像辨識上的有效性

### 7.2 AlexNet (2012, Alex Krizhevsky)

在 ImageNet 競賽 (ILSVRC) 上取得突破性成績，開啟了深度學習的黃金時代。

```
Input(224×224×3) → Conv11×11(96) → Pool → Conv5×5(256) → Pool
  → Conv3×3(384) → Conv3×3(384) → Conv3×3(256) → Pool → FC(4096) → FC(4096) → Output(1000)
```

- 參數量：約 60M
- 關鍵創新：使用 ReLU、Dropout、GPU 訓練、資料增強 (Data Augmentation)

### 7.3 VGGNet (2014, Oxford)

證明了網路深度的重要性。統一使用 3×3 卷積核。

```
核心設計原則：
- 所有卷積核均為 3×3
- 每經過一次池化，通道數翻倍：64 → 128 → 256 → 512 → 512
- VGG-16: 13 卷積層 + 3 全連接層
- VGG-19: 16 卷積層 + 3 全連接層
```

- 參數量：VGG-16 約 138M
- 貢獻：證明多個小卷積核堆疊可以取代大卷積核（兩個 3×3 等效於一個 5×5 的感受野，但參數更少且非線性更強）

### 7.4 ResNet (2015, Microsoft)

透過殘差連接 (Residual Connection / Skip Connection) 解決了深層網路的退化問題 (Degradation Problem)。

```
殘差區塊 (Residual Block):

輸入 x ─────────────────────┐
  │                          │ (Skip Connection)
  → Conv → BN → ReLU         │
  → Conv → BN                │
  │                          │
  └──── + ←──────────────────┘
         │
      ReLU
         │
     輸出 F(x) + x
```

- 原理：學習殘差映射 $F(x) = H(x) - x$ 比學習直接映射 $H(x)$ 更容易
- 版本：ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- 貢獻：使訓練超過 100 層的網路成為可能

### 7.5 架構演進總結 Architecture Evolution Summary

| 架構 | 年份 | 深度 | Top-5 Error (ImageNet) | 關鍵創新 |
|------|------|------|----------------------|---------|
| LeNet-5 | 1998 | 5 | — | 卷積 + 池化 |
| AlexNet | 2012 | 8 | 16.4% | ReLU, Dropout, GPU |
| VGGNet | 2014 | 16/19 | 7.3% | 統一 3×3 卷積 |
| GoogLeNet | 2014 | 22 | 6.7% | Inception Module |
| ResNet | 2015 | 152 | 3.6% | 殘差連接 |

---

## 8. CNN 可視化技術 CNN Visualization Techniques

理解 CNN「看到了什麼」是可解釋性 AI (Explainable AI, XAI) 的重要課題。以下介紹四種核心可視化技術。

### 8.1 卷積核視覺化 Filter/Kernel Visualization

**目標**：直接觀察學到的卷積核權重。

**方法**：將卷積核的權重值映射為灰階或彩色影像。

```python
# 概念程式碼
filters = model.conv1.weight.data  # 形狀：[out_ch, in_ch, kH, kW]
# 對每個 filter 正規化到 [0, 1] 範圍
# 以灰階或彩色圖像顯示
```

**觀察重點**：
- 第一層卷積核通常學到邊緣偵測器 (Edge Detectors)、色彩斑塊 (Color Blobs)
- 更深層的卷積核因為是高維的，較難直接視覺化

### 8.2 中間層特徵圖視覺化 Intermediate Feature Map Visualization

**目標**：觀察每一層卷積後的輸出，了解 CNN 在不同階段「看到」什麼。

**方法**：
1. 輸入一張影像
2. 使用 Hook 機制擷取中間層的輸出
3. 將各通道的特徵圖可視化

```python
# 概念程式碼
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
output = model(input_image)
feature_maps = activation['conv1']  # 形狀：[1, C, H, W]
```

**觀察重點**：
- 淺層特徵圖保留大量空間細節，能看到邊緣、紋理
- 深層特徵圖更抽象，空間解析度更低，但語義更豐富
- 某些通道可能對特定物件部位有強烈回應

### 8.3 Class Activation Mapping (CAM)

**目標**：找出 CNN 做決策時最關注的影像區域。

**前提條件**：模型最後一個卷積層後面必須接 Global Average Pooling (GAP) + 全連接層。

**原理**：
1. 取最後一個卷積層的特徵圖 $f_k(x, y)$（第 $k$ 個通道）
2. 取分類層中對類別 $c$ 的權重 $w_k^c$
3. CAM 公式：

$$
M_c(x, y) = \sum_{k} w_k^c \cdot f_k(x, y)
$$

4. 將 $M_c$ 上採樣 (Upsample) 到原圖大小，疊加為熱度圖 (Heatmap)

**限制**：
- 要求模型必須使用 GAP 結構
- 需要修改模型架構才能套用
- 只能對最後一層特徵圖做可視化

### 8.4 Grad-CAM：梯度加權類別激活映射 Gradient-weighted Class Activation Mapping

**目標**：克服 CAM 的結構限制，適用於任何 CNN 架構。

**原理**：
1. 前向傳播得到目標類別 $c$ 的分數 $y^c$
2. 計算 $y^c$ 對目標卷積層特徵圖 $A^k$ 的梯度 $\frac{\partial y^c}{\partial A^k}$
3. 對梯度做 Global Average Pooling，得到每個通道的重要性權重：

$$
\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{ij}}
$$

4. 加權求和並通過 ReLU：

$$
L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)
$$

5. 上採樣到原圖尺寸，生成熱度圖

**為什麼用 ReLU？** 我們只關注對目標類別有「正面影響」的特徵，負值代表對其他類別更重要。

**Grad-CAM 的優勢**：
- 不限制模型架構（適用於 VGG、ResNet、Inception 等）
- 可針對任何卷積層生成可視化
- 無需修改或重新訓練模型
- 可解釋任何目標類別（不只是預測類別）

### 8.5 CAM vs Grad-CAM 比較

| 特性 | CAM | Grad-CAM |
|------|-----|----------|
| 架構要求 | 必須有 GAP | 任何 CNN |
| 是否需修改模型 | 是 | 否 |
| 可視化層 | 僅最後卷積層 | 任意卷積層 |
| 計算方式 | FC 層權重 | 梯度 |
| 解析度 | 較低 | 較低（需上採樣） |
| 變體 | — | Grad-CAM++, Score-CAM, Eigen-CAM |

### 8.6 進階可視化技術（補充）

- **Guided Backpropagation**：修改 ReLU 的反向傳播，只保留正梯度，產生更細緻的像素級歸因
- **Guided Grad-CAM**：Grad-CAM 熱度圖 × Guided Backpropagation，結合類別鑑別力與高解析度
- **Score-CAM**：不使用梯度，以每個通道的前向傳播分數為權重，避免梯度噪聲問題
- **Grad-CAM++**：改進 Grad-CAM，使用高階梯度，更好地處理多個同類別物件

---

## 9. 遷移學習 Transfer Learning

### 9.1 什麼是遷移學習？ What is Transfer Learning?

遷移學習是指將在一個任務（通常是大型資料集如 ImageNet）上預訓練 (Pre-trained) 的模型，應用到另一個相關但不同的任務上。

### 9.2 為什麼遷移學習有效？ Why Does It Work?

1. **特徵的通用性 (Feature Generality)**：淺層學到的邊緣、紋理等特徵具有高度通用性
2. **資料效率 (Data Efficiency)**：目標任務的資料量可能很少，但預訓練模型已學到豐富的特徵表示
3. **訓練效率 (Training Efficiency)**：微調 (Fine-tuning) 比從頭訓練快得多

### 9.3 遷移學習的策略 Transfer Learning Strategies

| 策略 | 說明 | 適用場景 |
|------|------|---------|
| **特徵提取 Feature Extraction** | 凍結所有卷積層，只訓練分類器 | 目標資料少，任務相似 |
| **微調 Fine-tuning** | 解凍部分/全部卷積層，以較低學習率訓練 | 目標資料適中，任務有差異 |
| **逐層解凍 Gradual Unfreezing** | 從最後一層逐漸解凍到前面的層 | 避免破壞預訓練權重 |

### 9.4 實務建議 Practical Tips

```
目標資料量大 + 任務相似   → 微調整個網路（較低學習率）
目標資料量大 + 任務不同   → 微調較多層或從頭訓練
目標資料量小 + 任務相似   → 凍結卷積層，只訓練分類器
目標資料量小 + 任務不同   → 凍結淺層，微調深層
```

---

## 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 卷積神經網路 | Convolutional Neural Network (CNN) | 專門處理網格結構資料（如影像）的神經網路 |
| 卷積核/濾波器 | Filter / Kernel | 在影像上滑動並提取特徵的小矩陣 |
| 特徵圖 | Feature Map | 卷積核作用於輸入後的輸出 |
| 感受野 | Receptive Field | 特徵圖一個元素對應的原始輸入區域 |
| 填充 | Padding | 在輸入邊緣補值（通常為零） |
| 步幅 | Stride | 卷積核每次滑動的像素數 |
| 通道 | Channel | 影像的色彩通道或特徵圖的數量 |
| 池化 | Pooling | 下採樣操作（Max / Average） |
| 全局平均池化 | Global Average Pooling (GAP) | 對整張特徵圖取平均 |
| 殘差連接 | Residual Connection / Skip Connection | 讓輸入繞過某些層直接加到輸出 |
| 類別激活映射 | Class Activation Mapping (CAM) | 利用 GAP 權重定位 CNN 關注區域 |
| 梯度加權類別激活映射 | Gradient-weighted CAM (Grad-CAM) | 利用梯度定位 CNN 關注區域 |
| 遷移學習 | Transfer Learning | 將預訓練模型應用到新任務 |
| 微調 | Fine-tuning | 以較低學習率繼續訓練預訓練模型 |
| 特徵提取 | Feature Extraction | 凍結預訓練模型，僅訓練分類器 |
| 資料增強 | Data Augmentation | 透過變換（旋轉、翻轉等）擴增訓練資料 |
| 反向傳播 | Backpropagation | 透過鏈式法則計算梯度並更新權重 |
| 批次正規化 | Batch Normalization (BatchNorm) | 對每個 mini-batch 做正規化以穩定訓練 |

---

## 延伸閱讀 Further Reading

- Zeiler & Fergus (2014). "Visualizing and Understanding Convolutional Networks"
- Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Zhou et al. (2016). "Learning Deep Features for Discriminative Localization" (CAM 原始論文)
- He et al. (2016). "Deep Residual Learning for Image Recognition" (ResNet)
- Yosinski et al. (2014). "How transferable are features in deep neural networks?"
- PyTorch 官方教學 — CNN：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- PyTorch 官方教學 — Transfer Learning：https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
