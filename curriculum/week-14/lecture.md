# 第 14 週：深度學習訓練技巧
# Week 14: Deep Learning Training Techniques

## 學習目標 Learning Objectives
1. 掌握學習率排程策略 (Learning Rate Scheduling)：Step Decay、Cosine Annealing、Warmup、OneCycleLR
2. 理解早停 (Early Stopping) 的原理與實作方式
3. 了解資料增強 (Data Augmentation) 的常用技術（影像與文字）
4. 認識梯度裁剪 (Gradient Clipping) 與混合精度訓練 (Mixed Precision Training)
5. 掌握學習率搜尋 (LR Finder) 方法，找到最佳學習率範圍
6. 能整合多種訓練技巧，提升模型效能與訓練穩定性

---

## 1. 學習率排程 Learning Rate Scheduling

### 1.1 為什麼需要調整學習率？ Why Adjust the Learning Rate?

學習率 (Learning Rate, LR) 是深度學習訓練中最重要的超參數 (Hyperparameter) 之一。它控制每次參數更新的步幅大小：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L$$

其中 $\eta$ 就是學習率。

**固定學習率的問題：**

- **太大 (Too Large)**：訓練不穩定，損失震盪甚至發散 (Diverge)
- **太小 (Too Small)**：收斂速度極慢，可能陷入局部最小值 (Local Minimum)
- **無法兼顧**：訓練初期需要較大的步幅來快速探索，後期需要較小的步幅來精細調整

> **核心思想：** 在訓練過程中動態調整學習率，讓模型在不同階段使用不同的學習速度，以達到更好的收斂效果。

### 1.2 Step Decay（階梯衰減）

最簡單的學習率排程策略。每隔固定的 Epoch 數，將學習率乘以一個衰減因子 (Decay Factor)。

**公式：**
$$\eta_t = \eta_0 \times \gamma^{\lfloor t / S \rfloor}$$

其中：
- $\eta_0$：初始學習率 (Initial Learning Rate)
- $\gamma$：衰減因子（通常為 0.1 或 0.5）
- $S$：步進間隔（如每 30 個 Epoch）
- $t$：當前 Epoch

```
學習率 LR
  │
  │ ────────────
  │             │
  │             │────────────
  │                         │
  │                         │────────────
  │                                     │
  └──────────────────────────────────────▶ Epoch
  0          30          60          90
```

**PyTorch 實作：**

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()
```

**MultiStepLR** 允許在指定的多個 Epoch 降低學習率：

```python
from torch.optim.lr_scheduler import MultiStepLR

scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
```

**優缺點：**
- 優點：簡單直觀，經典方法（ResNet 原始論文即使用此策略）
- 缺點：需要手動設定衰減時機，不夠平滑

### 1.3 Exponential Decay（指數衰減）

每個 Epoch 學習率按固定比例衰減，產生平滑的衰減曲線。

**公式：**
$$\eta_t = \eta_0 \times \gamma^t$$

```python
from torch.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(optimizer, gamma=0.95)
```

```
學習率 LR
  │╲
  │ ╲
  │  ╲
  │   ╲
  │    ╲╲
  │      ╲╲╲
  │         ╲╲╲╲╲───────────
  └──────────────────────────▶ Epoch
```

**注意：** $\gamma$ 的選擇非常關鍵。如果 $\gamma$ 太小（如 0.8），學習率會衰減得太快；太大（如 0.999）則效果不明顯。

### 1.4 Cosine Annealing（餘弦退火）

學習率按餘弦函數從初始值降到最小值，在訓練後期形成平緩的衰減。

**公式：**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}} \pi\right)\right)$$

其中 $T_{max}$ 是總 Epoch 數，$\eta_{min}$ 是最小學習率。

```
學習率 LR
  │
  │ ╲
  │   ╲
  │     ╲
  │       ╲
  │         ╲╲
  │            ╲╲╲
  │                ╲╲╲╲╲╲
  │                        ╲╲╲╲╲╲╲╲╲╲
  └────────────────────────────────────▶ Epoch
```

**PyTorch 實作：**

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

**Cosine Annealing with Warm Restarts (SGDR)**：

Loshchilov & Hutter (2017) 提出在學習率降到最小值後，重新「回彈」到較大值，形成週期性的學習率曲線。這有助於跳出局部最小值。

```
學習率 LR
  │╲     ╱╲     ╱╲
  │ ╲   ╱  ╲   ╱  ╲
  │  ╲ ╱    ╲ ╱    ╲
  │   ╲      ╲      ╲
  └──────────────────────▶ Epoch
```

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

### 1.5 Warmup + Cosine（熱身 + 餘弦退火）

**Warmup（熱身）** 是在訓練最初幾個 Epoch，從一個很小的學習率**線性增加**到目標學習率，然後再開始衰減。

**為什麼需要 Warmup？**

1. **穩定初始訓練**：模型參數通常以隨機值初始化，此時的梯度方向不可靠。直接用大學習率更新，可能導致初期訓練不穩定
2. **BatchNorm 與 Adaptive Optimizers 的統計量**：這些元件需要幾個 Batch 來累積可靠的統計資訊
3. **大批量訓練的必要性**：Goyal et al. (2017) 在訓練 ImageNet 時發現，大 Batch Size 搭配 Warmup 可以使用更大的學習率

**Warmup + Cosine Decay 曲線：**

```
學習率 LR
  │        ╲
  │      ╱   ╲
  │    ╱       ╲
  │  ╱           ╲╲
  │╱                ╲╲╲
  │                     ╲╲╲╲╲╲╲
  └──────────────────────────────▶ Epoch
  │warmup│      cosine decay      │
```

**手動實作：**

```python
import math

def warmup_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        # 線性 Warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine Decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

**使用 `LambdaLR` 包裝：**

```python
from torch.optim.lr_scheduler import LambdaLR

warmup_epochs = 5
total_epochs = 100
base_lr = 0.1

scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch:
    (epoch + 1) / warmup_epochs if epoch < warmup_epochs
    else 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
)
```

### 1.6 OneCycleLR

Smith & Topin (2019) 提出的超收斂 (Super-Convergence) 策略。核心思想是在整個訓練過程中只使用**一個**學習率週期：先從小到大、再從大到小。

**三階段：**

1. **上升階段 (Ascending)**：學習率從 `max_lr / div_factor` 線性增加到 `max_lr`（約佔 30% 訓練時間）
2. **下降階段 (Descending)**：學習率從 `max_lr` 餘弦衰減到 `max_lr / div_factor`（約佔 70% 訓練時間）
3. **退火階段 (Annihilation)**：學習率進一步降到極小值 `max_lr / final_div_factor`（最後幾個 Epoch）

```
學習率 LR
  │       ╱╲
  │      ╱  ╲
  │     ╱    ╲
  │    ╱      ╲
  │   ╱        ╲╲
  │  ╱            ╲╲╲
  │ ╱                 ╲╲╲╲
  │╱                        ╲╲╲╲╲──
  └──────────────────────────────────▶ Epoch
  │  上升  │     下降     │退火│
```

**PyTorch 實作：**

```python
from torch.optim.lr_scheduler import OneCycleLR

optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,       # 上升階段比例
    div_factor=25,        # 初始 LR = max_lr / 25
    final_div_factor=1e4  # 最終 LR = max_lr / (25 * 10000)
)

for epoch in range(100):
    for batch in train_loader:
        train_step(...)
        scheduler.step()  # 注意：OneCycleLR 在每個 Batch 後更新
```

**OneCycleLR 的優勢：**
- 通常可以使用比傳統方法**更大的學習率**
- 訓練所需的 Epoch 數更少
- 具有隱式的正則化效果（大學習率本身就是一種正則化）

### 1.7 學習率排程策略比較

| 策略 Strategy | 適用場景 Use Case | 複雜度 | 超參數數量 |
|--------------|-----------------|--------|-----------|
| Step Decay | 經典 CNN 訓練、初學者 | 低 | 2 (step_size, gamma) |
| Exponential Decay | 需要平滑衰減的場景 | 低 | 1 (gamma) |
| Cosine Annealing | 通用，尤其長時間訓練 | 中 | 1-2 (T_max, eta_min) |
| Warmup + Cosine | 大模型、大 Batch 訓練 | 中 | 3 (warmup, total, lr) |
| OneCycleLR | 追求快速收斂的場景 | 中 | 3-4 (max_lr, pct_start, ...) |
| Warm Restarts | 需要跳出局部最優的場景 | 中 | 2 (T_0, T_mult) |

---

## 2. 早停 Early Stopping

### 2.1 原理 Principle

早停是防止過擬合 (Overfitting) 最直接有效的正則化技術之一。其核心思想是：

> **當驗證集 (Validation Set) 上的效能不再改善時，停止訓練。**

```
損失 Loss
  │
  │ ╲                    ╱╱╱╱
  │   ╲               ╱╱╱
  │     ╲           ╱╱
  │       ╲       ╱╱   ← 驗證損失 Validation Loss
  │         ╲   ╱╱
  │          ╲╱╱  ← 最佳點 Best Point
  │           ╲
  │            ╲╲
  │              ╲╲╲
  │                 ╲╲╲╲╲╲╲╲  ← 訓練損失 Training Loss
  └──────────────────────────────▶ Epoch
               ↑
          早停點 Early Stop
```

**過擬合的視覺信號：** 驗證損失開始上升（或驗證準確率開始下降），而訓練損失持續下降。兩者之間的差距就是**泛化差距 (Generalization Gap)**。

### 2.2 實作關鍵參數

| 參數 Parameter | 說明 Description | 典型值 |
|----------------|-----------------|--------|
| `monitor` | 監控的指標 | `val_loss` 或 `val_accuracy` |
| `patience` | 容忍幾個 Epoch 無改善 | 5-20 |
| `min_delta` | 最小改善量，低於此值不算改善 | 1e-4 |
| `restore_best_weights` | 停止後是否回復到最佳權重 | `True` |
| `mode` | `min`（監控 loss）或 `max`（監控 accuracy） | 視指標而定 |

### 2.3 PyTorch 手動實作

```python
class EarlyStopping:
    """早停機制，當驗證損失不再改善時停止訓練"""
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # 有改善：重置計數器，儲存最佳權重
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            # 無改善：增加計數器
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)

# 使用方式
early_stopping = EarlyStopping(patience=10)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

    scheduler.step()
```

### 2.4 實務建議

1. **Patience 的選擇**：太小可能過早停止（訓練尚未充分），太大則無法有效防止過擬合。建議設為 5-20 個 Epoch
2. **搭配學習率排程**：早停與學習率排程可以同時使用。注意觀察是否因為學習率衰減導致看似「不再改善」
3. **監控指標的選擇**：
   - 回歸問題：監控 `val_loss`（MSE 或 MAE）
   - 分類問題：可監控 `val_loss` 或 `val_accuracy`，但 `val_loss` 通常更敏感
4. **儲存最佳模型 (Model Checkpointing)**：即使不使用早停，也應該在訓練過程中定期儲存驗證指標最佳的模型

---

## 3. 資料增強 Data Augmentation

### 3.1 什麼是資料增強？ What is Data Augmentation?

資料增強是在訓練過程中，對原始資料施加隨機變換 (Random Transformations)，以產生更多樣化的訓練樣本。這是一種隱式的正則化方法 (Implicit Regularization)，能有效降低過擬合風險。

> **核心思想：** 透過人工製造合理的資料變異 (Variation)，讓模型學會對不相關的變化保持不變性 (Invariance)。

### 3.2 影像資料增強 Image Data Augmentation

#### 3.2.1 基礎幾何變換 Basic Geometric Transformations

| 方法 Method | 說明 Description | 參數 Parameters |
|------------|-----------------|----------------|
| 水平翻轉 Horizontal Flip | 左右鏡像 | 機率 p=0.5 |
| 垂直翻轉 Vertical Flip | 上下鏡像 | 機率 p=0.5 |
| 隨機旋轉 Random Rotation | 旋轉任意角度 | 角度範圍，如 [-15, 15] |
| 隨機裁切 Random Crop | 隨機位置裁切子區域 | 裁切後大小 |
| 隨機縮放 Random Resized Crop | 先裁切再縮放 | 縮放範圍 |
| 仿射變換 Random Affine | 旋轉 + 平移 + 縮放 + 剪切 | 多種參數 |

```python
import torchvision.transforms as T

transform_train = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]),
])
```

#### 3.2.2 色彩變換 Color Transformations

| 方法 Method | 說明 Description |
|------------|-----------------|
| 色彩抖動 Color Jitter | 隨機調整亮度 (Brightness)、對比度 (Contrast)、飽和度 (Saturation)、色調 (Hue) |
| 灰階化 Grayscale | 以一定機率轉為灰階 |
| 高斯模糊 Gaussian Blur | 加入模糊效果 |
| 隨機擦除 Random Erasing | 隨機遮蓋部分區域 |

```python
transform_train = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomGrayscale(p=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]),
    T.RandomErasing(p=0.2),
])
```

#### 3.2.3 Mixup

Zhang et al. (2018) 提出的方法，將兩張不同的訓練影像以隨機比例混合：

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

其中 $\lambda \sim \text{Beta}(\alpha, \alpha)$，通常 $\alpha = 0.2$ 或 $1.0$。

**效果：** 標籤空間也做了平滑處理 (Label Smoothing)，模型不會對任何單一類別過度自信。

```python
def mixup_data(x, y, alpha=0.2):
    """對一個 Batch 的資料進行 Mixup"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 的損失計算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

#### 3.2.4 CutMix

Yun et al. (2019) 提出的方法，與 Mixup 不同，CutMix 是**將一張圖的某個矩形區域替換為另一張圖的對應區域**：

$$\tilde{x} = M \odot x_i + (1 - M) \odot x_j$$

其中 $M$ 是一個二值遮罩 (Binary Mask)，標籤的混合比例等於遮罩中非零區域的面積比。

```python
def cutmix_data(x, y, alpha=1.0):
    """對一個 Batch 的資料進行 CutMix"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # 產生隨機矩形區域
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)  # 調整 lambda
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
```

**Mixup vs CutMix 比較：**

| 特性 | Mixup | CutMix |
|------|-------|--------|
| 混合方式 | 像素級混合（全圖） | 區域級替換（矩形） |
| 視覺效果 | 產生半透明疊加 | 保持局部清晰 |
| 強項 | 標籤平滑、泛化 | 局部特徵學習 |
| 常見 $\alpha$ | 0.2 - 1.0 | 1.0 |

#### 3.2.5 進階方法

- **AutoAugment** (Cubuk et al., 2019)：使用強化學習搜尋最佳增強策略
- **RandAugment** (Cubuk et al., 2020)：簡化版 AutoAugment，只需兩個超參數 (N, M)
- **TrivialAugment** (Muller & Hutter, 2021)：更簡化，無超參數

```python
# RandAugment 使用
from torchvision.transforms import RandAugment

transform_train = T.Compose([
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]),
])
```

### 3.3 文字資料增強 Text Data Augmentation

文字增強比影像增強更有挑戰性，因為對文字的修改很容易改變語意。

#### 3.3.1 同義詞替換 Synonym Replacement

隨機選擇句中的非停用詞 (Non-Stop Words)，替換為同義詞。

```python
# 使用 nlpaug 套件
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
text = "The quick brown fox jumps over the lazy dog"
augmented = aug.augment(text)
# 可能結果: "The fast brown fox leaps over the lazy dog"
```

#### 3.3.2 回翻譯 Back-Translation

將文字翻譯成另一語言再翻譯回來，產生不同的表達方式。

```
原文 (中文) → 英文翻譯 → 再翻回中文
"這家餐廳的食物非常好吃" → "The food at this restaurant is delicious" → "這家餐廳的食物很美味"
```

#### 3.3.3 其他文字增強方法

| 方法 | 說明 | 適用場景 |
|------|------|---------|
| 隨機插入 Random Insertion | 在隨機位置插入同義詞 | 短文本分類 |
| 隨機刪除 Random Deletion | 以一定機率刪除每個詞 | 抗雜訊訓練 |
| 隨機交換 Random Swap | 隨機交換兩個詞的位置 | 語序不敏感的任務 |
| EDA (Easy Data Augmentation) | 上述方法的組合 | 通用文字分類 |

### 3.4 資料增強的注意事項

1. **只對訓練集做增強**：測試集和驗證集不應做資料增強（除了 TTA，Test-Time Augmentation）
2. **保持語意不變**：增強後的資料應仍屬於原始類別
3. **領域知識很重要**：例如醫學影像中，垂直翻轉可能不合理
4. **過度增強可能有害**：太強的增強可能讓模型難以學習
5. **線上 vs 離線增強**：
   - **線上 (Online)**：每次讀取時隨機變換（推薦，PyTorch 預設）
   - **離線 (Offline)**：預先生成增強資料儲存（佔空間但速度快）

---

## 4. 梯度裁剪 Gradient Clipping

### 4.1 為什麼需要梯度裁剪？

在深度網路（特別是 RNN/LSTM）中，反向傳播可能產生**梯度爆炸 (Gradient Explosion)**——梯度值變得非常大，導致參數更新過猛、訓練不穩定。

### 4.2 兩種裁剪方式

#### 按值裁剪 Clip by Value

將每個梯度元素限制在 $[-\text{clip\_value}, +\text{clip\_value}]$ 之間：

$$g_i = \text{clip}(g_i, -v, v)$$

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

#### 按範數裁剪 Clip by Norm（推薦）

計算所有梯度的全局 L2 範數 (Global Norm)，如果超過閾值 (Max Norm) 就等比例縮放：

$$\hat{g} = \begin{cases} g & \text{if } \|g\| \leq \text{max\_norm} \\ \frac{\text{max\_norm}}{\|g\|} g & \text{if } \|g\| > \text{max\_norm} \end{cases}$$

```python
# 推薦做法：在 optimizer.step() 之前調用
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**為什麼按範數裁剪更好？** 按值裁剪會改變梯度的方向；按範數裁剪只改變梯度的大小，保持了梯度的方向性。

### 4.3 實務建議

- **常用 max_norm 值**：0.5 - 5.0，依任務而定
- **RNN/LSTM 訓練**：幾乎必備
- **Transformer 訓練**：通常使用 max_norm=1.0
- **監控梯度範數**：記錄每步的梯度範數可以幫助診斷訓練問題

```python
# 監控梯度範數
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm: {total_norm:.4f}")
```

---

## 5. 混合精度訓練 Mixed Precision Training 簡介

### 5.1 什麼是混合精度？

傳統深度學習使用 FP32（32 位元浮點數）進行計算。混合精度訓練 (Mixed Precision Training) 是在訓練過程中同時使用 FP16（16 位元）和 FP32，以加速訓練並減少記憶體使用。

**為什麼可以用 FP16？**

- 現代 GPU（如 NVIDIA Tensor Cores）對 FP16 運算有硬體加速
- 大多數梯度計算用 FP16 就足夠了
- 關鍵的累加運算（如梯度更新）仍保持 FP32 以維持數值穩定

### 5.2 核心機制

1. **FP16 前向傳播**：用 FP16 做矩陣運算（快！）
2. **FP32 損失計算**：損失用 FP32 以避免溢出
3. **損失縮放 (Loss Scaling)**：將損失乘以一個大數再反向傳播，防止 FP16 的小梯度被截斷為零（Underflow）
4. **FP32 參數更新**：保持一份 FP32 的主參數 (Master Weights) 用於更新

### 5.3 PyTorch AMP 實作

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

for data, target in train_loader:
    optimizer.zero_grad()

    # 自動混合精度：前向傳播用 FP16
    with autocast('cuda'):
        output = model(data)
        loss = criterion(output, target)

    # 反向傳播：先縮放損失，再用 FP32 更新
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 5.4 效益

| 指標 | FP32 | Mixed Precision |
|------|------|-----------------|
| 訓練速度 | 基準 | 快 1.5x - 3x |
| GPU 記憶體 | 基準 | 減少 ~50% |
| 模型精度 | 基準 | 幾乎相同 |

**注意：** 混合精度訓練需要支援 FP16 的 GPU（NVIDIA Volta 以後的架構）。

---

## 6. 學習率搜尋 LR Range Test / LR Finder

### 6.1 原理 Principle

Smith (2017) 提出的方法，用於快速找到最佳學習率範圍。做法是：

1. 從一個很小的學習率開始（如 $10^{-7}$）
2. 每個 Batch 逐步增加學習率（通常按指數增長）
3. 記錄每個學習率對應的損失值
4. 畫出 LR vs Loss 曲線

```
損失 Loss
  │
  │                              ╱
  │                            ╱╱
  │                          ╱╱
  │                        ╱╱
  │ ──────────╲          ╱╱
  │             ╲╲     ╱╱
  │               ╲╲ ╱╱
  │                ╲╲╱   ← 最低點
  │                 ↑
  │            最佳 LR 範圍
  └──────────────────────────────▶ Learning Rate (log scale)
  10⁻⁷    10⁻⁵    10⁻³    10⁻¹   10¹
```

### 6.2 如何解讀 LR Finder 曲線？

- **最佳學習率**：損失下降最快的位置（梯度最陡處），通常取**最低點左邊一段**的值
- **經驗法則**：選取最低損失對應學習率的 **1/10** 作為最大學習率
- **右側陡升**：表示學習率太大，訓練開始發散

### 6.3 實作

```python
def lr_finder(model, train_loader, optimizer, criterion,
              start_lr=1e-7, end_lr=10, num_iter=100):
    """學習率搜尋器"""
    lrs = []
    losses = []
    best_loss = float('inf')

    # 儲存初始狀態
    init_state = model.state_dict().copy()
    init_optim_state = optimizer.state_dict().copy()

    # 計算每步的學習率乘數
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)

    lr = start_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for i, (data, target) in enumerate(train_loader):
        if i >= num_iter:
            break

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())

        # 如果損失爆炸，提前停止
        if loss.item() > best_loss * 4:
            break
        if loss.item() < best_loss:
            best_loss = loss.item()

        # 增加學習率
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 恢復初始狀態
    model.load_state_dict(init_state)
    optimizer.load_state_dict(init_optim_state)

    return lrs, losses
```

### 6.4 使用 `torch-lr-finder` 套件

```python
# pip install torch-lr-finder
from torch_lr_finder import LRFinder

model = ...
optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
criterion = nn.CrossEntropyLoss()

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
lr_finder.plot()  # 畫出 LR vs Loss 曲線
lr_finder.reset()  # 恢復模型和優化器狀態
```

---

## 7. 實務訓練技巧清單 Practical Training Checklist

### 7.1 訓練前 Before Training

- [ ] **資料檢查**：確認資料標註正確、分布合理
- [ ] **基準模型 (Baseline)**：先用簡單配置建立基準
- [ ] **學習率搜尋**：使用 LR Finder 確定學習率範圍
- [ ] **資料增強策略**：根據任務選擇合理的增強方法
- [ ] **程式碼驗證**：用少量資料跑一個完整的 train-eval 迴圈，確認沒有 Bug

### 7.2 訓練中 During Training

- [ ] **監控指標**：同時追蹤訓練損失和驗證損失
- [ ] **學習率排程**：使用 Warmup + Cosine 或 OneCycleLR
- [ ] **早停**：設定合理的 patience（通常 5-20 Epoch）
- [ ] **梯度裁剪**：特別是 RNN 或 Transformer 模型
- [ ] **混合精度**：在支援的硬體上開啟 AMP
- [ ] **定期儲存 Checkpoint**：至少儲存驗證指標最佳的模型

### 7.3 訓練後 After Training

- [ ] **載入最佳模型**：使用驗證集上表現最好的 Checkpoint
- [ ] **測試集評估**：只評估一次，報告最終結果
- [ ] **超參數紀錄**：記錄所有超參數以確保可重現性 (Reproducibility)
- [ ] **錯誤分析 (Error Analysis)**：檢視模型犯錯的樣本，尋找改善方向

### 7.4 常見陷阱 Common Pitfalls

| 陷阱 Pitfall | 症狀 Symptom | 解決方案 Solution |
|-------------|-------------|------------------|
| 學習率太大 | Loss 震盪或 NaN | 降低 LR，使用 LR Finder |
| 學習率太小 | 收斂極慢 | 增加 LR，使用 Warmup |
| Batch Size 太大 | 泛化能力差 | 搭配 Warmup + 更大的 LR |
| 沒有資料增強 | 過擬合嚴重 | 加入合理的增強策略 |
| 資料洩漏 | 測試結果過好 | 檢查增強是否只用在訓練集 |
| 梯度爆炸 | Loss 突然變 NaN | 梯度裁剪 + 降低 LR |
| 忘記 `model.eval()` | 推論結果不穩定 | 推論時呼叫 `model.eval()` |

### 7.5 推薦的訓練配方 Recommended Recipe

對於一般的影像分類任務（如 CIFAR-10, ImageNet），以下是一個穩健的起始配置：

```python
# 優化器：SGD with Momentum 或 AdamW
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# 學習率排程：OneCycleLR 或 Warmup + Cosine
scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=100,
                       steps_per_epoch=len(train_loader))

# 資料增強：RandAugment + CutMix/Mixup
transform_train = T.Compose([
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# 早停：patience=10
early_stopping = EarlyStopping(patience=10)

# 梯度裁剪：max_norm=1.0
clip_grad_norm_(model.parameters(), max_norm=1.0)

# 混合精度：開啟 AMP
scaler = GradScaler('cuda')
```

---

## 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 學習率 | Learning Rate (LR) | 控制參數更新步幅的超參數 |
| 學習率排程 | Learning Rate Scheduling | 在訓練中動態調整學習率的策略 |
| 階梯衰減 | Step Decay | 每隔固定 Epoch 降低學習率 |
| 指數衰減 | Exponential Decay | 每個 Epoch 按固定比例衰減學習率 |
| 餘弦退火 | Cosine Annealing | 按餘弦函數平滑衰減學習率 |
| 熱身 | Warmup | 訓練初期逐步增加學習率 |
| 超收斂 | Super-Convergence | 使用特殊 LR 策略大幅縮短訓練時間 |
| 早停 | Early Stopping | 驗證指標不再改善時停止訓練 |
| 資料增強 | Data Augmentation | 對訓練資料施加隨機變換以增加多樣性 |
| 混入 | Mixup | 將兩張影像以隨機比例混合的增強方法 |
| 剪切混入 | CutMix | 將一張圖的矩形區域替換為另一張圖 |
| 梯度裁剪 | Gradient Clipping | 限制梯度大小以防止梯度爆炸 |
| 梯度爆炸 | Gradient Explosion | 梯度值過大導致訓練不穩定的現象 |
| 混合精度訓練 | Mixed Precision Training | 同時使用 FP16 和 FP32 的訓練方式 |
| 損失縮放 | Loss Scaling | 混合精度中放大損失以防止梯度下溢 |
| 學習率搜尋 | LR Range Test / LR Finder | 逐步增加 LR 以找到最佳範圍的方法 |
| 泛化差距 | Generalization Gap | 訓練效能與驗證/測試效能之間的差距 |
| 回翻譯 | Back-Translation | 翻譯成另一語言再翻回來的文字增強法 |
| 隨機增強 | RandAugment | 簡化的自動增強策略 |
| 測試時增強 | Test-Time Augmentation (TTA) | 推論時對輸入做增強再平均預測結果 |

---

## 延伸閱讀 Further Reading

- Smith, L.N., "Cyclical Learning Rates for Training Neural Networks" (2017)
- Smith, L.N. & Topin, N., "Super-Convergence: Very Fast Training Using Large Learning Rates" (2019)
- Loshchilov, I. & Hutter, F., "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017)
- Zhang, H. et al., "mixup: Beyond Empirical Risk Minimization" (2018)
- Yun, S. et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (2019)
- Goyal, P. et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)
- Micikevicius, P. et al., "Mixed Precision Training" (2018)
- PyTorch 官方文件 — Learning Rate Scheduler: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- PyTorch 官方文件 — Automatic Mixed Precision: https://pytorch.org/docs/stable/amp.html
