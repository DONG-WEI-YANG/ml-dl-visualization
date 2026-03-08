# 第 12 週投影片：CNN 視覺化（卷積核、特徵圖、CAM/Grad-CAM）

---

## Slide 1: 本週主題
# CNN 視覺化
### Convolutional Neural Network Visualization
- 卷積核 Filters & 特徵圖 Feature Maps
- CAM & Grad-CAM
- 理解 CNN 的「眼睛」看到了什麼

---

## Slide 2: 回顧 — 從全連接到卷積
### 全連接層的三大問題
1. **參數爆炸** — 224x224x3 影像 + 1000 神經元 = 1.5 億參數
2. **忽略空間結構** — 展平影像丟失像素鄰居關係
3. **缺乏平移不變性** — 貓在左上角 vs 右下角 = 完全不同的輸入

### CNN 的解決方案
- 局部連接 Local Connectivity
- 權重共享 Weight Sharing
- 空間層次 Spatial Hierarchy

---

## Slide 3: 卷積運算的直覺
### 卷積 = 滑動窗口 + 加權求和
```
  輸入影像           卷積核 (3x3)        特徵圖
 ┌─────────┐       ┌─────┐          ┌───────┐
 │ · · · · │  ×    │ w w w │   →     │ · · · │
 │ · · · · │       │ w w w │         │ · · · │
 │ · · · · │       │ w w w │         │ · · · │
 │ · · · · │       └─────┘          └───────┘
 └─────────┘
```
- 每個位置：逐元素相乘再求和
- 不同卷積核 → 偵測不同模式

---

## Slide 4: 卷積核的效果
### 經典手工卷積核 vs 學習到的卷積核
| 類型 | 效果 |
|------|------|
| 邊緣偵測 | 找出影像中的輪廓 |
| 銳化 | 增強細節 |
| 模糊 | 平滑化雜訊 |

**CNN 的突破**：卷積核不再需要手工設計，透過反向傳播**自動學習**

---

## Slide 5: Padding 與 Stride
### 填充 Padding
- Valid：不補零 → 輸出縮小
- Same：補零 → 輸出不變

### 步幅 Stride
- Stride=1：逐步滑動
- Stride=2：跳步滑動 → 尺寸減半

### 輸出尺寸公式
$$O = \lfloor (H + 2p - k) / s \rfloor + 1$$

---

## Slide 6: 多通道卷積
### RGB 影像的卷積
```
輸入: H × W × 3 (RGB)
卷積核: k × k × 3   (深度 = 輸入通道數)
────────────────────
一個卷積核 → 一張特徵圖

使用 64 個卷積核
→ 輸出: H' × W' × 64
```
- 每個卷積核跨越**所有**輸入通道
- 輸出通道數 = 卷積核數量

---

## Slide 7: 池化層
### Max Pooling vs Average Pooling
```
 輸入 (4×4)           Max Pool (2×2)     Avg Pool (2×2)
┌──────────┐          ┌─────┐            ┌─────┐
│ 1 3 2 1 │          │ 5 3 │            │2.75 1.75│
│ 5 2 1 3 │    →     │ 8 6 │            │4.00 4.25│
│ 4 8 6 2 │          └─────┘            └─────┘
│ 3 1 4 5 │
└──────────┘
```
- **Max Pooling**：保留最強激活（隱藏層間最常用）
- **Average Pooling**：保留平均特徵
- **Global Average Pooling (GAP)**：整張圖取一個平均值 → 取代 FC 層

---

## Slide 8: 特徵圖的層次
### CNN 學到的是什麼？
| 淺層 Shallow | 中層 Middle | 深層 Deep |
|:---:|:---:|:---:|
| 邊緣 Edges | 紋理 Textures | 物件部件 Parts |
| 顏色梯度 | 形狀片段 | 語義概念 |
| 高解析度 | 中解析度 | 低解析度 |

> 淺層 = 「像素級特徵」
> 深層 = 「語義級特徵」

---

## Slide 9: 經典 CNN 架構演進
### 從 LeNet 到 ResNet
```
1998  LeNet-5     5 層    手寫數字
  │
2012  AlexNet     8 層    ImageNet 突破 ← 深度學習元年
  │
2014  VGGNet     16/19層  統一 3×3 卷積
  │
2014  GoogLeNet   22 層   Inception Module
  │
2015  ResNet      152 層  殘差連接 ← 解決深度退化
```

---

## Slide 10: ResNet 殘差連接
### 為什麼需要 Skip Connection？
```
        x ────────────────┐
        │                 │  (捷徑)
     Conv→BN→ReLU         │
     Conv→BN              │
        │                 │
        └──── + ←─────────┘
              │
           ReLU
              │
          F(x) + x
```
- 學習殘差 F(x) = H(x) - x 更容易
- 梯度可以直接流過 Skip Connection
- 使 100+ 層訓練成為可能

---

## Slide 11: 為什麼要視覺化 CNN？
### CNN 是「黑盒子」嗎？
- 問題：CNN 做了正確的預測，但**為什麼**？
- 它真的學到了有意義的特徵？
- 還是只是利用了資料集的偏差 (Dataset Bias)？

### 視覺化的價值
1. **除錯 Debugging**：發現模型關注錯誤的區域
2. **信任 Trust**：向使用者解釋預測依據
3. **理解 Understanding**：了解不同層學到了什麼
4. **改進 Improvement**：指導模型設計與資料收集

---

## Slide 12: 技術一 — 卷積核視覺化
### 直接觀察學到的權重
```python
filters = model.conv1.weight.data
# 將每個 filter 正規化到 [0, 1]
# 以網格方式顯示
```
- 第一層：能看到邊緣偵測器、色彩斑塊
- 深層：維度太高，難以直接解讀

---

## Slide 13: 技術二 — 特徵圖視覺化
### 中間層輸出的「快照」
```python
# 使用 PyTorch Hook 機制
def hook_fn(module, input, output):
    feature_maps.append(output)

model.layer1.register_forward_hook(hook_fn)
```
- 觀察每個通道的激活模式
- 對比淺層 vs 深層的差異
- 找出哪些通道對特定物件有回應

---

## Slide 14: 技術三 — CAM
### Class Activation Mapping
```
最後卷積層的特徵圖    GAP 後的 FC 權重
     f_k(x,y)    ×     w_k^c
         │                │
         └───── 加權求和 ───┘
                  │
            CAM 熱度圖 M_c(x,y)
```
$$M_c(x,y) = \sum_k w_k^c \cdot f_k(x,y)$$

**限制**：模型必須有 GAP + FC 結構

---

## Slide 15: 技術四 — Grad-CAM
### 突破 CAM 的架構限制
**核心步驟**：
1. 前向傳播 → 目標類別分數 $y^c$
2. 反向傳播 → 對目標層的梯度
3. 全局平均梯度 → 通道重要性權重 $\alpha_k^c$
4. 加權組合 + ReLU → 熱度圖

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c \cdot A^k\right)$$

**優勢**：適用於任何 CNN 架構，無需修改模型

---

## Slide 16: Grad-CAM 實例解讀
### 一張熱度圖勝過千言萬語
```
原圖                 Grad-CAM 疊加圖
┌──────────┐        ┌──────────┐
│          │        │  ▓▓▓     │  紅色 = 高關注
│   🐱     │  →     │  ▓█▓     │  藍色 = 低關注
│          │        │          │
└──────────┘        └──────────┘

預測：貓 (Cat) - 信心度 95%
模型關注：貓的臉部與耳朵
```
- 可驗證模型是否關注正確區域
- 可對比不同類別的關注區域

---

## Slide 17: CAM vs Grad-CAM 比較
| 特性 | CAM | Grad-CAM |
|:---:|:---:|:---:|
| 架構限制 | 需要 GAP | 任何 CNN |
| 需修改模型 | 是 | 否 |
| 可選層 | 僅最後卷積層 | 任意卷積層 |
| 計算基礎 | FC 權重 | 梯度 |
| 變體 | — | Grad-CAM++, Score-CAM |

---

## Slide 18: 遷移學習 Transfer Learning
### 站在巨人的肩膀上
```
ImageNet 預訓練 (1000 類, 120 萬張圖)
         │
    ┌────┴────┐
    │ 特徵提取器 │  ← 凍結 (Freeze)
    │ Conv 層  │
    └────┬────┘
    ┌────┴────┐
    │ 新分類器  │  ← 訓練 (Train)
    │ FC 層    │
    └─────────┘
         │
    目標任務 (例：醫學影像分類)
```

---

## Slide 19: 遷移學習策略
### 根據資料量與任務相似度選擇
| | 任務相似 | 任務不同 |
|:---:|:---:|:---:|
| **資料多** | 微調全部層 | 微調多數層 / 從頭訓練 |
| **資料少** | 凍結卷積層，只訓練分類器 | 凍結淺層，微調深層 |

**實務建議**：
- 微調時學習率要比從頭訓練低 10-100 倍
- 逐層解凍 (Gradual Unfreezing) 效果通常更穩定

---

## Slide 20: 今日實作預告
### Notebook 實作內容
1. 用 PyTorch 建構簡單 CNN → 訓練於 CIFAR-10
2. 卷積核視覺化 — 觀察第一層學到的模式
3. 特徵圖提取 — 比較淺層 vs 深層
4. Grad-CAM 實作 — 生成熱度圖解釋預測
5. 進階：使用 Pretrained ResNet 進行遷移學習

---

## Slide 21: 本週作業
### Assignment
1. 訓練一個 CNN 並視覺化不同層的特徵圖
2. 使用 Grad-CAM 分析模型預測
3. 撰寫觀察報告：模型在哪些類別表現好/差？為什麼？
4. 進階挑戰：遷移學習 + 不同資料集

**繳交期限**：下週上課前

---

## Slide 22: 下週預告
### Week 13: RNN/序列建模
- 遞迴神經網路 (RNN) 基礎
- LSTM & GRU — 長短期記憶
- Transformer 概念入門
- 注意力機制 (Attention) 視覺化
