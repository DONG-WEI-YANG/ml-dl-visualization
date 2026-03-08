# 第 14 週投影片：深度學習訓練技巧
# Week 14 Slides: Deep Learning Training Techniques

---

## Slide 1: 本週主題
# 深度學習訓練技巧
### Deep Learning Training Techniques
- 學習率排程 (LR Scheduling)
- 早停 (Early Stopping)
- 資料增強 (Data Augmentation)
- 梯度裁剪 (Gradient Clipping)
- 混合精度訓練 (Mixed Precision)
- 學習率搜尋 (LR Finder)

---

## Slide 2: 學習率的重要性
### 學習率 = 訓練的油門
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla L$$

| 學習率 | 效果 |
|:------:|:----:|
| 太大 | 震盪、發散 |
| 太小 | 收斂極慢 |
| 剛好 | 快速且穩定 |

**問題：** 固定的「剛好」學習率在訓練全程都適用嗎？

---

## Slide 3: 動態學習率的直覺
### 開車比喻
- **訓練初期** = 在高速公路上 → 大油門（大 LR）
- **訓練後期** = 進入停車場 → 小油門（小 LR）
- **Warmup** = 剛發動 → 先暖車再加速

> 動態調整學習率 = 在正確的時機用正確的速度

---

## Slide 4: Step Decay
### 階梯衰減
$$\eta_t = \eta_0 \times \gamma^{\lfloor t / S \rfloor}$$

```
LR │────────
   │        │────────
   │                 │────────
   └──────────────────────────▶ Epoch
```

- 每隔 S 個 Epoch，LR 乘以 $\gamma$（如 0.1）
- 經典方法：ResNet 在 Epoch 30, 60, 90 降低 LR
- **優點：** 簡單直觀
- **缺點：** 不夠平滑，需手動設定時機

---

## Slide 5: Exponential Decay
### 指數衰減
$$\eta_t = \eta_0 \times \gamma^t$$

```
LR │╲
   │ ╲
   │  ╲╲
   │    ╲╲╲╲
   │        ╲╲╲╲╲╲╲───
   └──────────────────▶ Epoch
```

- 每個 Epoch 衰減固定比例
- **$\gamma$ 的選擇很關鍵：** 0.95 vs 0.99 差異巨大

---

## Slide 6: Cosine Annealing
### 餘弦退火
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$

```
LR │╲
   │  ╲
   │    ╲╲
   │       ╲╲╲
   │           ╲╲╲╲╲╲╲╲
   └────────────────────▶ Epoch
```

- 平滑衰減，後期衰減速度放緩
- 目前最流行的 Schedule 之一
- **變體：** Cosine Annealing with Warm Restarts (SGDR)

---

## Slide 7: Warm Restarts (SGDR)
### 週期性重啟

```
LR │╲    ╱╲      ╱╲
   │ ╲  ╱  ╲    ╱  ╲
   │  ╲╱    ╲  ╱    ╲
   │         ╲╱      ╲
   └─────────────────────▶ Epoch
```

- 學習率降到底後「回彈」到高值
- 有助於跳出局部最小值 (Local Minimum)
- `T_mult` 可控制每次週期的長度倍增

---

## Slide 8: Warmup + Cosine
### 熱身 + 餘弦衰減

```
LR │      ╲
   │    ╱   ╲
   │  ╱       ╲╲
   │╱            ╲╲╲╲╲
   └──────────────────▶ Epoch
   │warm│ cosine decay │
```

**為什麼需要 Warmup？**
1. 初始參數隨機 → 梯度不可靠 → 大 LR 會讓模型「亂跑」
2. BatchNorm 需要累積統計量
3. 大 Batch Size 訓練的必備技巧

---

## Slide 9: OneCycleLR
### 超收斂 Super-Convergence

```
LR │     ╱╲
   │    ╱  ╲
   │   ╱    ╲
   │  ╱      ╲╲
   │ ╱          ╲╲╲
   │╱               ╲╲──
   └──────────────────────▶ Epoch
   │上升│   下降   │退火│
```

- **三階段：** 上升 → 下降 → 退火
- 可以使用更大的學習率、更少的 Epoch
- 大 LR 本身就是一種正則化
- **注意：** 每個 Batch 後更新（不是每個 Epoch）

---

## Slide 10: LR Schedule 總比較
### 各種策略一覽

| 策略 | 適用場景 | 難度 |
|:----:|:--------:|:----:|
| Step Decay | 經典 CNN、入門 | 低 |
| Exponential | 平滑衰減 | 低 |
| Cosine Annealing | 通用、長訓練 | 中 |
| Warmup + Cosine | 大模型/大 Batch | 中 |
| OneCycleLR | 快速收斂 | 中 |
| Warm Restarts | 逃出局部最優 | 中 |

**2024+ 趨勢：** Warmup + Cosine 或 OneCycleLR 是主流

---

## Slide 11: 早停 Early Stopping
### 過擬合的「剎車」

```
Loss│                    ╱ Val Loss
    │  ╲              ╱╱
    │    ╲          ╱╱
    │      ╲╲    ╱╱
    │        ╲╲╱╱ ← 最佳點
    │          ╲╲
    │            ╲╲╲ Train Loss
    └──────────────────────▶ Epoch
              ↑
         Early Stop
```

- 驗證損失不再改善 → 停止訓練
- 回復到最佳權重 (Restore Best Weights)
- **Patience = 容忍多少個 Epoch 無改善**

---

## Slide 12: Early Stopping 實作要點
### 關鍵參數

| 參數 | 說明 | 建議值 |
|:----:|:----:|:------:|
| patience | 容忍 Epoch 數 | 5-20 |
| min_delta | 最小改善量 | 1e-4 |
| monitor | 監控指標 | val_loss |
| restore_best | 回復最佳權重 | True |

**重點提醒：**
- 搭配 Model Checkpointing 使用
- 不要把 patience 設太小（可能過早停止）
- 搭配 LR Schedule 時注意交互作用

---

## Slide 13: 資料增強的動機
### Data Augmentation — 為什麼？

```
 訓練資料     增強後
 ┌───┐      ┌───┬───┬───┬───┐
 │   │  ──▶ │翻轉│旋轉│裁切│色彩│
 │ 1 │      │   │   │   │   │
 │張 │      │ = │ 4 │ + │張 │
 └───┘      └───┴───┴───┴───┘
```

- **更多樣化** → 減少過擬合
- **增加不變性** → 翻轉不變、旋轉不變...
- **不需要更多真實資料** → 省成本！

---

## Slide 14: 影像增強 — 基礎方法
### Geometric & Color Transformations

| 方法 | 效果 |
|:----:|:----:|
| 水平翻轉 Flip | 左右鏡像 |
| 隨機旋轉 Rotation | 旋轉 +/-15 度 |
| 隨機裁切 Crop | 隨機區域裁切 |
| 色彩抖動 Color Jitter | 亮度/對比度/飽和度變化 |
| 隨機擦除 Erasing | 遮蓋部分區域 |
| 高斯模糊 Blur | 加入模糊 |

**重點：** 只對訓練集做增強！

---

## Slide 15: Mixup & CutMix
### 進階增強方法

**Mixup** (Zhang et al., 2018)：
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$
→ 兩張圖**像素級混合**

**CutMix** (Yun et al., 2019)：
→ 用另一張圖的**矩形區域**替換

| | Mixup | CutMix |
|:----:|:-----:|:------:|
| 混合方式 | 全圖半透明 | 局部替換 |
| 保留細節 | 低 | 高 |
| 標籤 | 按比例混合 | 按面積比 |

---

## Slide 16: 自動增強策略
### AutoAugment → RandAugment → TrivialAugment

| 方法 | 超參數 | 特點 |
|:----:|:------:|:----:|
| AutoAugment | 搜尋出的策略 | 效果好但搜尋成本高 |
| RandAugment | N (數量), M (強度) | 簡單且效果接近 |
| TrivialAugment | 無 | 最簡單，零超參數 |

```python
transform = T.Compose([
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(mean, std)
])
```

---

## Slide 17: 文字增強
### Text Data Augmentation

| 方法 | 說明 |
|:----:|:----:|
| 同義詞替換 | "快速" → "迅速" |
| 回翻譯 | 中→英→中 |
| 隨機插入 | 插入同義詞 |
| 隨機刪除 | 隨機移除詞 |
| 隨機交換 | 交換詞的位置 |

**挑戰：** 文字修改容易改變語意！
需要比影像增強更加謹慎

---

## Slide 18: 梯度裁剪
### Gradient Clipping — 防止梯度爆炸

**按範數裁剪（推薦）：**
- 計算全局梯度 L2 範數
- 若超過閾值 → 等比例縮放
- **保持梯度方向不變**

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0
)
optimizer.step()
```

**何時需要？** RNN, LSTM, Transformer 訓練

---

## Slide 19: 混合精度訓練
### Mixed Precision Training

| | FP32 | FP16 |
|:--:|:----:|:----:|
| 精度 | 高 | 夠用 |
| 速度 | 基準 | 快 1.5-3x |
| 記憶體 | 大 | 小 ~50% |

**核心流程：**
1. FP16 前向傳播（快）
2. FP32 損失計算（穩）
3. Loss Scaling + 反向傳播
4. FP32 參數更新（準）

```python
with autocast('cuda'):
    output = model(data)
    loss = criterion(output, target)
```

---

## Slide 20: LR Finder
### 學習率搜尋

```
Loss│                     ╱
    │                   ╱╱
    │────────╲        ╱╱
    │          ╲╲   ╱╱
    │            ╲╲╱
    │             ↑
    │        最佳 LR
    └────────────────────▶ LR (log)
    10⁻⁷  10⁻⁵  10⁻³  10⁻¹
```

**步驟：**
1. 從極小 LR 開始，逐步指數增加
2. 記錄每步的 Loss
3. 找到 Loss 下降最快的區域
4. 取最低點 LR 的 **1/10** 作為 max_lr

---

## Slide 21: 訓練技巧全景圖
### 組合使用的威力

```
┌──────────────────────────────────────────┐
│           完整訓練流程                      │
│                                          │
│  LR Finder → 找最佳 LR                   │
│      ↓                                   │
│  Data Augmentation → 增加多樣性            │
│      ↓                                   │
│  Warmup + Cosine / OneCycleLR            │
│      ↓                                   │
│  Gradient Clipping → 穩定訓練             │
│      ↓                                   │
│  Mixed Precision → 加速訓練               │
│      ↓                                   │
│  Early Stopping → 防止過擬合              │
│      ↓                                   │
│  Best Model Checkpoint → 最終模型         │
└──────────────────────────────────────────┘
```

---

## Slide 22: 實務訓練配方
### Recommended Recipe

```python
# 1. 優化器：AdamW
optimizer = AdamW(model.parameters(),
                  lr=1e-3, weight_decay=0.01)

# 2. 排程：OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=1e-3,
                       epochs=100, ...)

# 3. 增強：RandAugment
transform = RandAugment(num_ops=2, magnitude=9)

# 4. 早停：patience=10
early_stopping = EarlyStopping(patience=10)

# 5. 梯度裁剪
clip_grad_norm_(model.parameters(), max_norm=1.0)

# 6. 混合精度
scaler = GradScaler('cuda')
```

---

## Slide 23: 常見陷阱
### Common Pitfalls

| 陷阱 | 症狀 | 解方 |
|:----:|:----:|:----:|
| LR 太大 | Loss = NaN | LR Finder |
| LR 太小 | 收斂極慢 | 增加 LR + Warmup |
| 無增強 | 嚴重過擬合 | 加入增強策略 |
| 忘記 eval() | 推論不穩 | model.eval() |
| 梯度爆炸 | 突然 NaN | 梯度裁剪 |
| 過早停止 | 欠訓練 | 增加 patience |

---

## Slide 24: 本週實作
### 動手做！
1. 各種 LR Schedule 曲線視覺化
2. 比較不同 LR Schedule 的訓練效果
3. Early Stopping 實作
4. 影像增強效果視覺化
5. LR Finder 實作
6. CIFAR-10 完整訓練實驗

---

## Slide 25: 本週作業
### Assignment
- 在 CIFAR-10 上訓練 CNN
- 分別實驗：固定 LR vs Cosine vs OneCycleLR
- 加入 Early Stopping 與資料增強
- 比較結果並撰寫分析報告

---

## Slide 26: 下週預告
### Week 15: 模型評估與偏誤檢測
- 混淆矩陣深入分析
- 公平性指標 (Fairness Metrics)
- 穩健性測試 (Robustness Testing)
- AI 倫理案例分析
