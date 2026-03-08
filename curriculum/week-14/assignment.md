# 第 14 週作業：深度學習訓練技巧實驗
# Week 14 Assignment: Deep Learning Training Techniques Experiment

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 至課程平台

---

## 作業一：學習率排程比較實驗（25%）

在 CIFAR-10 資料集上訓練一個簡單的 CNN（可使用課堂提供的模型），比較以下三種學習率策略的訓練效果：

1. **固定學習率 (Constant LR)**：lr = 0.01
2. **Cosine Annealing**：初始 lr = 0.01, eta_min = 1e-6
3. **OneCycleLR**：max_lr = 0.01, pct_start = 0.3

**要求：**
- 每種策略訓練 30 個 Epoch
- 繪製三種策略的學習率變化曲線（一張圖，三條線）
- 繪製三種策略的訓練損失 (Train Loss) 與驗證損失 (Val Loss) 曲線
- 記錄每種策略的最終測試準確率 (Test Accuracy)
- 以 1-2 段文字分析哪種策略表現最好，並推測原因

## 作業二：Early Stopping 實作（20%）

1. 實作一個 `EarlyStopping` 類別，包含以下功能：
   - 支援 `patience` 參數設定
   - 支援 `min_delta` 最小改善量
   - 支援 `restore_best_weights` 功能
   - 提供 `early_stop` 布林值判斷是否觸發

2. 使用你的 `EarlyStopping` 類別訓練模型：
   - 設定 patience = 5，max_epochs = 100
   - 記錄實際停止的 Epoch 數
   - 繪製含早停標記的訓練/驗證損失曲線

3. 比較 patience = 3, 5, 10, 20 對最終模型效能的影響

## 作業三：資料增強效果視覺化（20%）

1. 選取 CIFAR-10 中的一張影像，展示以下增強效果（每種方法展示 4 張變化後的圖）：
   - 水平翻轉 (Horizontal Flip)
   - 隨機旋轉 (Random Rotation, +/-15 度)
   - 色彩抖動 (Color Jitter)
   - 隨機裁切 + 縮放 (Random Resized Crop)
   - 隨機擦除 (Random Erasing)

2. 比較以下三種配置的訓練效果（訓練 20 個 Epoch）：
   - **無增強 (No Augmentation)**：僅 Normalize
   - **基礎增強 (Basic)**：Flip + Rotation + Crop
   - **完整增強 (Full)**：基礎 + ColorJitter + Erasing

3. 繪製三種配置的訓練/驗證損失曲線，分析資料增強對過擬合的影響

## 作業四：LR Finder 實作與應用（15%）

1. 實作 LR Finder（或使用 `torch-lr-finder` 套件）
2. 在你的 CNN 模型上執行 LR Range Test
3. 繪製 LR vs Loss 曲線
4. 根據曲線選出建議的最大學習率
5. 使用找到的學習率搭配 OneCycleLR 訓練模型，報告測試準確率

## 作業五：綜合訓練實驗（20%）

將本週學到的所有技巧組合在一起，在 CIFAR-10 上訓練你的最佳模型：

**基準 (Baseline)：**
- 固定 LR = 0.01
- 無資料增強
- 無早停
- 訓練 50 Epoch

**完整版 (Full Tricks)：**
- 使用 LR Finder 找到的學習率
- OneCycleLR 或 Warmup + Cosine 排程
- 資料增強（你認為最佳的組合）
- Early Stopping (patience=10)
- 梯度裁剪 (max_norm=1.0)

**要求：**
- 記錄兩個版本的最終測試準確率
- 繪製兩個版本的訓練/驗證曲線
- 撰寫 2-3 段分析文字，說明每個技巧的貢獻
- 列出你最終使用的所有超參數 (Hyperparameters)

---

## 加分題 Bonus（+10%）

實作 Mixup 或 CutMix 增強方法，在你的完整版模型上加入此技巧，並分析是否進一步提升了效能。

---

## 評分標準 Grading Criteria

- **程式碼正確執行 Code Execution:** 40%
  - 程式碼能正確執行並產出結果
  - 使用合理的訓練流程
- **視覺化品質 Visualization Quality:** 25%
  - 圖表清晰、標籤完整、配色適當
  - 包含圖例 (Legend)、標題 (Title)、軸標籤 (Axis Labels)
- **分析與反思 Analysis & Reflection:** 25%
  - 對實驗結果有合理的分析
  - 能解釋不同技巧的效果
  - 思路清晰、論述有據
- **程式碼品質 Code Quality:** 10%
  - 程式碼結構清晰、有適當註解
  - 使用函數封裝重複邏輯

## 提示 Hints

1. 若沒有 GPU，可以使用 Google Colab 的免費 GPU
2. CIFAR-10 影像大小為 32x32，訓練速度較快
3. CNN 模型不需太複雜，3-5 層卷積即可
4. 建議先完成作業一、二，再做三、四，最後整合為作業五
5. 可參考課堂 Notebook 的程式碼作為起點
