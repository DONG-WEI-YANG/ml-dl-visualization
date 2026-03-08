# 第 14 週教師手冊
# Week 14 Teacher Guide

## 時間分配 Time Allocation（90 分鐘）

| 時段 | 分鐘 | 活動 | 說明 |
|------|:---:|------|------|
| 開場 | 5 | 回顧上週 + 本週導覽 | 回顧 RNN/Transformer，帶出「如何訓練得更好？」 |
| 理論 (1) | 15 | 學習率排程 | Slide 2-10，重點在 Cosine 和 OneCycleLR |
| 理論 (2) | 10 | 早停 + 資料增強 | Slide 11-17，搭配圖片展示增強效果 |
| 實作 (1) | 25 | Notebook 前半段 | LR Schedule 曲線 + 訓練比較 + Early Stopping |
| 實作 (2) | 20 | Notebook 後半段 | 資料增強視覺化 + LR Finder + CIFAR-10 實驗 |
| 理論 (3) | 5 | 梯度裁剪 + 混合精度 | Slide 18-19，簡要帶過 |
| 總結 | 10 | 訓練技巧清單 + 作業說明 | Slide 21-25，強調整合運用 |

## 教學重點 Key Teaching Points

### 1. 學習率排程是「必修」不是「選修」
- 強調現代深度學習幾乎不會使用固定學習率
- 用開車比喻：高速公路（大 LR）→ 停車場（小 LR）
- 讓學生透過 Notebook 親眼看到不同 Schedule 的差異

### 2. 從直覺出發，再進入公式
- 先展示「為什麼固定 LR 不好」的實驗結果
- 再解釋每種 Schedule 的動機與公式
- 避免一開始就塞太多數學

### 3. 資料增強是「免費的午餐」
- 不需要更多真實資料就能提升效能
- 用 Notebook 展示增強後的影像，讓學生直觀理解
- 強調只對**訓練集**做增強（常見錯誤！）

### 4. 本週是「工具箱」——重點在組合使用
- 每個技巧單獨用效果有限
- 組合使用才能發揮最大效益
- 最後的 CIFAR-10 實驗是整合練習

## 教學策略 Teaching Strategies

### 互動環節建議

1. **LR 猜猜看（5 分鐘）：**
   - 展示三條不同的訓練曲線（Loss vs Epoch）
   - 讓學生猜測分別用了哪種 LR Schedule
   - 答案揭曉後討論為什麼

2. **增強前後配對（3 分鐘）：**
   - 展示原始影像和增強後影像
   - 問學生：「人類還認得出這是什麼嗎？」
   - 引出「增強不能改變語意」的原則

3. **小組討論（5 分鐘）：**
   - 「如果你只能選三個技巧用在你的模型上，你會選哪三個？為什麼？」
   - 讓學生發表後，教師總結最常見的「標配」組合

### 程式碼展示技巧

- **Live Coding 順序：**
  1. 先跑一個「什麼技巧都沒有」的基準模型
  2. 逐步加入技巧，每加一個就跑一次，觀察效果
  3. 最後展示全部技巧組合的結果

- **避免的做法：**
  - 不要一次把所有程式碼都貼出來
  - 不要跳過訓練過程（讓學生看到進度條在跑）
  - 不要用太複雜的模型（簡單 CNN 即可，重點在技巧）

## 檢核點 Checkpoints

- [ ] 學生能說出至少 3 種 LR Schedule 的名稱與特點
- [ ] 學生能解釋 Warmup 的必要性
- [ ] 學生理解 Early Stopping 的 patience 概念
- [ ] 學生能區分 Mixup 與 CutMix 的差異
- [ ] 學生成功在 Notebook 中跑出 LR Finder 曲線
- [ ] 學生能在 CIFAR-10 上比較有無技巧的效能差異
- [ ] 學生了解梯度裁剪的用途

## AI 助教設定 AI Tutor Configuration

本週 AI 助教設定為「引導式問答」模式：

- **第一層提示：** 提醒學生回顧講義中的對應段落
- **第二層提示：** 給出相關的 PyTorch API 名稱（如 `CosineAnnealingLR`）
- **第三層提示：** 提供程式碼框架（有 TODO 的版本）
- **不直接給出：** 完整的訓練迴圈程式碼或分析文字

### 助教常見回應範例

**學生問：「OneCycleLR 的 pct_start 該設多少？」**
- Level 1：「想想看，pct_start 控制的是上升階段的比例。你覺得讓學習率花多少比例的時間來增加是合理的？」
- Level 2：「原始論文建議 0.3，也就是 30% 的訓練時間用來增加學習率。你可以試試不同的值。」
- Level 3：提供 `OneCycleLR(optimizer, max_lr=0.1, epochs=100, steps_per_epoch=len(loader), pct_start=0.3)` 的範例

**學生問：「為什麼加了 Early Stopping 反而效果變差？」**
- Level 1：「Early Stopping 的 patience 設了多少？」
- Level 2：「如果 patience 太小，模型可能還沒充分學習就被停止了。試試把 patience 從 3 改成 10。」
- Level 3：「另外也要確認 min_delta 的設定。如果設太大，正常的小幅改善會被忽略。」

## 常見問題與排除 Troubleshooting

### Q1: OneCycleLR 報錯 "Tried to step X times"
**原因：** OneCycleLR 需要在**每個 Batch** 後調用 `scheduler.step()`，而非每個 Epoch。
**解法：** 確認 `steps_per_epoch` 設定正確，且 `scheduler.step()` 放在 Batch 迴圈內。

### Q2: 加入資料增強後訓練速度變慢
**原因：** 線上增強會增加 CPU 負擔（特別是複雜的增強如 RandAugment）。
**解法：**
- 增加 DataLoader 的 `num_workers`
- 使用更快的增強庫（如 Albumentations）
- 在 Colab 上這是正常現象，影響不大

### Q3: LR Finder 的曲線太不平滑
**原因：** 每個 Batch 的 Loss 波動大。
**解法：** 對 Loss 做指數平滑 (Exponential Smoothing)：
```python
smoothed_loss = beta * smoothed_loss + (1 - beta) * loss
```

### Q4: 混合精度在 CPU 上報錯
**原因：** `autocast('cuda')` 需要 CUDA GPU。
**解法：** 在 CPU 上不使用混合精度，或在 Colab 上切換到 GPU Runtime。

### Q5: 學生的測試準確率差異很大
**原因：** 沒有設定隨機種子 (Random Seed)。
**解法：** 提醒學生加入：
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

### Q6: Early Stopping 和 LR Schedule 衝突
**原因：** 學習率大幅降低時，Loss 可能暫時不改善，觸發早停。
**解法：** 將 patience 設大一些（如 15-20），或監控驗證**準確率**而非 Loss。

## 備課提醒 Preparation Notes

### 課前準備
- [ ] 在有 GPU 的環境中預先跑過所有 Notebook Cell（確認沒有 Bug）
- [ ] 準備好「無技巧 vs 有技巧」的對比結果截圖（以防現場跑不完）
- [ ] 確認 Colab 的 GPU 可用（提前開啟 Runtime）
- [ ] 準備幾張直觀的增強效果圖片（CIFAR-10 的解析度太低可能不夠清晰，可額外準備 ImageNet 的高解析度範例）

### 時間管理
- 本週內容多，**嚴格控制理論時間**，確保留夠實作時間
- 梯度裁剪和混合精度**簡要帶過即可**，不需花太多時間
- 如果時間不足，可將 LR Finder 的實作留給學生課後完成

### 銜接下週
- 本週的訓練技巧是建立「好模型」的基礎
- 下週（Week 15）將聚焦「如何評估模型」——不只看準確率，還要看公平性和穩健性
- 可預告：「有了好的訓練技巧，模型準確率提升了。但準確率高就代表模型好嗎？」

## 延伸資源 Extra Resources（供教師參考）

- Andrej Karpathy, "A Recipe for Training Neural Networks" (blog post)
  - 經典的訓練技巧文章，可推薦給進階學生
- PyTorch 官方 Training Reference Scripts
  - 展示了生產級的訓練配置
- fast.ai 的 1cycle policy 實踐
  - Leslie Smith 的 OneCycleLR 在 fast.ai 中的實踐案例
- timm (PyTorch Image Models) 的訓練配置
  - 開源專案中的實際訓練 recipe
