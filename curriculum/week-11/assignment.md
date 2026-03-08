# 第 11 週作業：神經網路基礎實作
# Week 11 Assignment: Neural Network Basics Implementation

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 至課程平台
**總分 Total:** 100 分

---

## 作業一：激活函數探索（20 分）

### 任務 Task
實作並視覺化一個**課堂未介紹**的激活函數，並與 ReLU 進行比較。

**可選激活函數（任選一個）：**
- ELU (Exponential Linear Unit)
- Mish ($x \cdot \tanh(\text{softplus}(x))$)
- Softplus ($\ln(1 + e^x)$)
- SELU (Scaled ELU)

**要求：**
1. (5 分) 寫出該激活函數的數學公式與導數
2. (5 分) 使用 Matplotlib 繪製**函數圖與導數圖並排**（雙子圖 subplots）
3. (5 分) 與 ReLU 在同一張圖上比較，標註關鍵差異
4. (5 分) 用文字說明：這個激活函數解決了 ReLU 的什麼問題？適用於什麼場景？

**提示：** 參考 PyTorch 文件中的 `torch.nn.functional` 模組。

---

## 作業二：MLP 分類器 — Fashion-MNIST（30 分）

### 任務 Task
使用 PyTorch 建構一個 MLP 分類器，在 Fashion-MNIST 資料集上達到 **88% 以上的測試準確率**。

**要求：**

1. (5 分) **資料載入與前處理**
   - 使用 `torchvision.datasets.FashionMNIST` 載入資料
   - 將影像展平 (Flatten) 為 784 維向量
   - 適當的正規化 (Normalize)

2. (10 分) **模型架構**
   - 至少 2 個隱藏層
   - 每層使用適當的激活函數
   - 使用 `nn.Module` 定義模型（不使用 `nn.Sequential`）
   - 在模型定義中加入中文/英文註解

3. (10 分) **訓練流程**
   - 使用 `CrossEntropyLoss` 損失函數
   - 使用 Adam 或 SGD 優化器
   - 訓練至少 10 個 epoch
   - 記錄每個 epoch 的訓練損失與測試準確率
   - 繪製訓練曲線（損失 vs. epoch、準確率 vs. epoch）

4. (5 分) **結果分析**
   - 報告最終測試準確率
   - 隨機抽取 10 張測試圖片，顯示圖片與模型預測
   - 針對預測錯誤的案例，分析可能原因

---

## 作業三：正則化實驗（30 分）

### 任務 Task
在作業二的基礎上，實驗不同正則化技術對模型效能的影響。

**要求：**

1. (10 分) **實驗設計** — 訓練以下 4 個模型變體：

   | 模型 | 正則化 |
   |------|--------|
   | Model A | 無正則化 (Baseline) |
   | Model B | Dropout (p=0.3) |
   | Model C | L2 正則化 (weight_decay=1e-4) |
   | Model D | Dropout + L2 |

2. (10 分) **視覺化比較**
   - 在同一張圖上繪製 4 個模型的**訓練損失曲線**
   - 在同一張圖上繪製 4 個模型的**測試準確率曲線**
   - 圖表需包含圖例 (Legend)、軸標籤 (Axis Labels)、標題 (Title)

3. (10 分) **分析與討論**
   - 哪個模型的訓練損失最低？哪個模型的測試準確率最高？
   - 觀察訓練損失與測試準確率的差距 (Gap)，哪個模型過擬合最嚴重？
   - 你認為 Dropout 和 L2 正則化的效果有何不同？為什麼？

---

## 作業四：BatchNorm 效果分析（20 分）

### 任務 Task
實驗 Batch Normalization 對訓練過程的影響。

**要求：**

1. (8 分) **模型對比** — 建構兩個相同架構的 MLP，唯一差異是是否加入 `nn.BatchNorm1d`：

   ```python
   # Model without BN
   # Linear → ReLU → Linear → ReLU → Linear

   # Model with BN
   # Linear → BatchNorm → ReLU → Linear → BatchNorm → ReLU → Linear
   ```

2. (7 分) **訓練對比**
   - 使用**相同的超參數**（學習率、batch size、epoch 數）
   - 嘗試使用較大學習率（如 0.01），觀察有/無 BatchNorm 的穩定性差異
   - 繪製兩個模型的訓練損失曲線與測試準確率曲線

3. (5 分) **觀察與結論**
   - BatchNorm 是否加速了收斂？量化說明（例如：達到 85% 準確率所需的 epoch 數）
   - 較大學習率下，BatchNorm 是否讓訓練更穩定？
   - 結合課堂知識，解釋你觀察到的現象

---

## 加分題 Bonus（+10 分）

### 選項 A：梯度流視覺化（+5 分）
建構一個 10 層的網路（分別使用 Sigmoid 和 ReLU 激活函數），在一次 forward + backward pass 後，繪製每一層梯度的平均絕對值 (Mean Absolute Gradient) 的長條圖，直觀展示梯度消失現象。

### 選項 B：自訂學習率排程（+5 分）
為作業二的模型加入學習率排程器 (Learning Rate Scheduler)，比較以下策略的訓練曲線：
- 固定學習率 (Constant LR)
- StepLR（每 5 epoch 衰減 0.1 倍）
- CosineAnnealingLR

---

## 繳交清單 Submission Checklist

- [ ] Notebook (.ipynb) 包含所有程式碼與輸出
- [ ] 每個作業區塊有清楚的 Markdown 標題
- [ ] 圖表有標題、軸標籤、圖例
- [ ] 分析文字以 Markdown Cell 撰寫（非程式碼註解）
- [ ] 程式碼有適當註解
- [ ] Notebook 可從頭到尾順利執行（Restart & Run All）

---

## 注意事項 Notes
- 可使用 Google Colab 的免費 GPU 加速訓練
- Fashion-MNIST 每張圖片為 28x28 灰階影像，共 10 個類別
- 確保隨機種子 (Random Seed) 的設定以利重現結果
- 鼓勵嘗試不同超參數，但需記錄所有實驗設定
