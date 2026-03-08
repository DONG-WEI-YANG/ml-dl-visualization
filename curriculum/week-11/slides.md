# 第 11 週投影片：神經網路基礎 — 激活函數、正則化與 BatchNorm 可視化

---

## Slide 1: 本週主題
# 神經網路基礎
### Neural Network Basics
- 從感知器到多層網路
- 激活函數家族
- 正則化技術
- BatchNorm 的威力
- PyTorch 初體驗

---

## Slide 2: 從大腦到人工智慧
### 生物神經元 vs. 人工神經元
```
生物神經元                    人工神經元 (感知器)
┌─────────────┐              ┌─────────────┐
│ 樹突 → 接收信號 │    <->     │ 輸入 x1, x2, ...│
│ 突觸 → 信號強度 │    <->     │ 權重 w1, w2, ...│
│ 細胞體 → 整合  │    <->     │ 加權求和 Swx+b  │
│ 閾值 → 激發?   │    <->     │ 激活函數 f(z)   │
│ 軸突 → 傳遞    │    <->     │ 輸出 y_hat     │
└─────────────┘              └─────────────┘
```
- 1943 McCulloch-Pitts 模型 → 1958 Rosenblatt 感知器

---

## Slide 3: 感知器的數學
### Perceptron
```
z = w1*x1 + w2*x2 + ... + wn*xn + b
y_hat = step(z)
```

**能做的：** AND, OR（線性可分）
**不能做的：** XOR（非線性可分）

> Minsky & Papert (1969) 的證明 → 第一次 AI 寒冬

---

## Slide 4: 多層感知器 MLP
### 堆疊多層 → 解決非線性問題
```
輸入層       隱藏層 1      隱藏層 2      輸出層
(4 neurons)  (8 neurons)  (6 neurons)  (3 neurons)

  O ──┐                               ┌── O
  O ──┼──→ O ──┐                 ┌──→ O
  O ──┼──→ O ──┼──→ O ──┐  ┌──→ O
  O ──┘──→ O ──┼──→ O ──┼──┘
         → O ──┼──→ O ──┘
         → O ──┼──→ O
         → O ──┘──→ O
         → O       → O
```
- 萬能近似定理：理論上可以逼近任何連續函數
- 深度 = 隱藏層數，寬度 = 每層神經元數

---

## Slide 5: 為什麼需要激活函數？
### 沒有激活函數 = 線性疊加 = 等於單層
```
Layer2(Layer1(x)) = W2(W1*x + b1) + b2 = W'*x + b'
```

**關鍵洞察：** 激活函數引入**非線性**，讓網路能學習複雜模式！

---

## Slide 6: Sigmoid
### sigma(z) = 1 / (1 + exp(-z))
- 輸出：(0, 1)
- 導數最大值：0.25
- 用途：二元分類輸出層

| 優點 | 缺點 |
|:---:|:---:|
| 輸出可解釋為機率 | 梯度消失 |
| 平滑可微 | 非零中心 |
| | 計算 exp 較慢 |

---

## Slide 7: Tanh
### tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
- 輸出：(-1, 1) — **零中心！**
- 導數最大值：1
- 相當於 Sigmoid 的平移縮放版

| 優點 | 缺點 |
|:---:|:---:|
| 零中心 | 仍有梯度消失 |
| 導數更大 (max=1) | 兩端飽和 |

---

## Slide 8: ReLU — 深度學習的功臣
### ReLU(z) = max(0, z)
- 2012 年 AlexNet 採用 → 深度學習革命

| 優點 | 缺點 |
|:---:|:---:|
| 計算極快 | 死亡 ReLU |
| 正半軸梯度=1 | 非零中心 |
| 稀疏激活 | z=0 不可微 |
| 收斂速度快 6x | |

> **Dead ReLU：** 一旦 z < 0，梯度永遠為 0，神經元「永久死亡」

---

## Slide 9: Leaky ReLU 與變體
### LeakyReLU(z) = max(alpha*z, z)，alpha = 0.01
- 負半軸保留微小梯度 → 避免死亡 ReLU

**變體家族：**
- **PReLU**：alpha 可學習
- **RReLU**：alpha 隨機取值
- **ELU**：alpha*(exp(z) - 1) 當 z < 0

---

## Slide 10: GELU — Transformer 的最愛
### GELU(z) = z * Phi(z)
- BERT、GPT 系列的預設激活函數
- 平滑版 ReLU：用機率「柔性門控」
- 直覺：「z 越大越有信心保留，z 越小越傾向丟棄」

---

## Slide 11: Swish — 自動搜尋的結果
### Swish(z) = z * sigma(z)
- Google Brain 2017 — NAS (Neural Architecture Search) 發現
- 非單調：在 z < 0 區域有小凹陷
- beta -> inf 退化為 ReLU
- 深層 CNN 中常優於 ReLU

---

## Slide 12: 激活函數選擇指南
| 場景 | 推薦 |
|:---:|:---:|
| 隱藏層（預設） | **ReLU** |
| 死亡 ReLU 問題 | **Leaky ReLU / PReLU** |
| Transformer | **GELU** |
| 深層 CNN | **Swish** |
| 二元分類輸出 | **Sigmoid** |
| 多元分類輸出 | **Softmax** |
| 回歸輸出 | **無（線性）** |

---

## Slide 13: 反向傳播 — 追責的藝術
### Backpropagation = 鏈式法則 (Chain Rule)
```
前向傳播 →→→→→→→→→→→→→→→→→→→→→→→
  x  →  h1  →  h2  →  y_hat  →  Loss
←←←←←←←←←←←←←←←←←←←←←←← 反向傳播
  dL/dx <- dL/dh1 <- dL/dh2 <- dL/dy_hat
```
- 每個權重對誤差的「責任」= 梯度
- PyTorch autograd 自動完成！

---

## Slide 14: 梯度消失 — 深層 Sigmoid 的惡夢
### 10 層 Sigmoid 網路的梯度衰減
```
梯度 ~ 0.25 * 0.25 * ... * 0.25 (10 layers) = 0.25^10 ≈ 1e-6
```

- 淺層幾乎學不到東西！
- 解法：ReLU、殘差連接、BatchNorm

---

## Slide 15: 梯度爆炸 — 另一個極端
### 權重過大 → 梯度指數增長
- Loss 劇烈震盪、出現 NaN
- 解法：梯度裁剪 (Gradient Clipping)、適當初始化

---

## Slide 16: 正則化 — 對抗過擬合
### 三大武器

| 技術 | 原理 | 效果 |
|:---:|:---:|:---:|
| **L1** | lambda * sum(abs(w)) | 稀疏權重 → 特徵選擇 |
| **L2** | lambda * sum(w^2) | 權重收縮 → 平滑模型 |
| **Dropout** | 隨機關閉神經元 | 集成效果 → 防共適應 |

---

## Slide 17: Dropout 視覺化
### 訓練時：隨機關閉 (p=0.5)
```
完整網路         Dropout 後
O → O → O       O → O → O
O → O → O       O → X → O
O → O → O   →   O → O → X
O → O → O       O → X → O
```
- 每次 forward = 不同的子網路
- 測試時：全部啟用（權重已自動縮放）
- 效果類似「多模型投票」

---

## Slide 18: 早停 Early Stopping
### 最簡單也最有效的正則化
```
Loss
 |   \          Training Loss
 |    \____________________________
 |     \ /
 |      X <- Best Model (保存這個！)
 |     / \
 |    /   \   Validation Loss
 └────────────────→ Epoch
         ^
    Stop Here (patience=10)
```

---

## Slide 19: Batch Normalization
### 每層輸入都做正規化
```
x_hat = (x - mu_B) / sqrt(sigma_B^2 + epsilon)
y = gamma * x_hat + beta
```

- mu_B, sigma_B：mini-batch 統計量
- gamma, beta：**可學習參數**（讓網路自己決定最佳分佈）

---

## Slide 20: BatchNorm 的四大好處
1. **加速收斂** — 可以用更大學習率
2. **正則化效果** — mini-batch 雜訊 ≈ 輕量 Dropout
3. **減少初始化敏感度** — 初始化不完美也能訓練
4. **平滑損失面** — 梯度更穩定（Santurkar et al., 2018）

> 訓練時用 batch 統計量；推論時用移動平均 → 記得 `model.eval()`！

---

## Slide 21: 權重初始化
### 打破對稱性 + 維持方差穩定

| 方法 | 適用激活函數 | 方差 |
|:---:|:---:|:---:|
| **Xavier** | Sigmoid / Tanh | 2 / (n_in + n_out) |
| **He** | ReLU 家族 | 2 / n_in |

- 全零初始化 → 所有神經元學同樣的東西（災難！）
- 過大 → 梯度爆炸
- 過小 → 梯度消失

---

## Slide 22: PyTorch 快速入門
### 三大核心概念
```python
# 1. Tensor — 多維陣列 + GPU 加速
x = torch.randn(32, 784)

# 2. autograd — 自動微分
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()       # x.grad = 6.0

# 3. nn.Module — 模型建構基底
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10))
    def forward(self, x):
        return self.layers(x)
```

---

## Slide 23: PyTorch 訓練循環
### 四步驟口訣：「前 → 算 → 反 → 更」
```python
for epoch in range(epochs):
    for x, y in train_loader:
        pred = model(x)           # 1. 前向傳播
        loss = criterion(pred, y) # 2. 算損失
        optimizer.zero_grad()     # (清梯度)
        loss.backward()           # 3. 反向傳播
        optimizer.step()          # 4. 更新參數
```

---

## Slide 24: 本週 Demo
### 互動式視覺化
1. 六種激活函數的函數圖 + 導數圖並排
2. 深層網路梯度消失現象（Sigmoid vs. ReLU）
3. Dropout 隨機遮罩的效果
4. 有/無 BatchNorm 的訓練曲線對比
5. MNIST 手寫數字分類器

---

## Slide 25: 本週作業
1. 實作自訂激活函數並視覺化
2. 建構 MLP 分類 Fashion-MNIST
3. 實驗不同正則化組合的效果
4. 分析 BatchNorm 對訓練的影響

---

## Slide 26: 下週預告
### Week 12: CNN 視覺化
- 卷積運算 (Convolution)
- 卷積核 (Filters) 與特徵圖 (Feature Maps)
- CAM / Grad-CAM 解釋 CNN 決策
- 經典架構：LeNet → VGG → ResNet
