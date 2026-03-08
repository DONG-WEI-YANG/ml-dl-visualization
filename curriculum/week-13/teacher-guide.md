# 第 13 週教師手冊
# Week 13 Teacher Guide

## 時間分配 Time Allocation（90 分鐘）

| 時段 | 分鐘 | 活動 | 說明 |
|------|:---:|------|------|
| 回顧 | 5 | Week 12 CNN 重點回顧 | 從空間特徵過渡到序列特徵 |
| 理論一 | 20 | 序列資料 + RNN + 梯度消失 | Slide 2-7，建立問題意識 |
| 理論二 | 20 | LSTM + GRU + 雙向 RNN | Slide 8-14，門控機制重點 |
| 實作一 | 15 | Notebook: RNN/LSTM 正弦波預測 | 學生動手，教師巡場 |
| 理論三 | 15 | Attention + Transformer 概念 | Slide 15-22，概念為主 |
| 實作二 | 10 | Notebook: Self-Attention 實作 | 快速體驗 Attention 計算 |
| 總結 | 5 | 回顧 + 作業說明 | Slide 23-26 |

---

## 教學重點 Key Teaching Points

### 1. 建立「序列思維」
- 第一步是讓學生理解為什麼 FC 和 CNN 不適合序列資料
- 使用「狗咬人 vs 人咬狗」的例子直觀說明順序的重要性
- 強調序列模型的核心挑戰：**如何在有限的記憶中保留長距離的資訊**

### 2. RNN 到 LSTM 的邏輯鏈
- 不要直接教 LSTM，先讓學生理解 RNN 的**梯度消失為什麼是致命的**
- 建議流程：
  1. 先展示 Vanilla RNN 在長序列上的失敗案例
  2. 用直覺解釋「連乘效應」（乘以小於 1 的數 100 次會趨近 0）
  3. 引出 LSTM 的「高速公路」類比
  4. 逐門講解遺忘門、輸入門、輸出門的功能

### 3. LSTM 門控值的直覺
- **遺忘門**：用「讀到句號時，前一句的細節可以淡化」來解釋
- **輸入門**：用「讀到新名詞時，將其存入記憶」來解釋
- **輸出門**：用「回答問題時，從記憶中選取相關部分」來解釋
- 建議在黑板上畫完整的 LSTM 資料流圖，讓學生逐步跟著走

### 4. GRU 作為對照
- GRU 不需要花太多時間，重點在於與 LSTM 的**差異對比**
- 強調「更新門 = 遺忘門 + 輸入門的結合」
- 經驗法則：兩者效能差異通常不大，建議都試

### 5. Attention 與 Transformer（概念為主）
- 這部分進階學生可能感興趣，但**不要求所有人完全理解**
- 重點傳達三個核心思想：
  1. **注意力 = 選擇性關注**（而非壓縮成一個向量）
  2. **Self-Attention = 序列內部互相關注**（捕捉全局依賴）
  3. **Transformer = 全注意力 + 位置編碼**（拋棄遞迴）
- 可以用「考試時翻課本」的類比解釋注意力：每次回答問題時，你會翻到最相關的那一頁

---

## 教學策略 Teaching Strategies

### 視覺化教學建議
1. **RNN 展開圖動畫**：建議用動畫或逐步繪圖展示 RNN 如何在時間步上展開
2. **梯度消失示意**：可以用漸層色條表示梯度大小隨時間衰減
3. **LSTM 資料流**：用不同顏色的箭頭區分三個門和細胞狀態的路徑
4. **注意力熱力圖**：展示機器翻譯中 Source-Target 的對齊圖

### 互動活動建議
1. **「傳話遊戲」類比（5 分鐘）**：
   - 安排 10 位學生排成一列
   - 第一位學生看一段句子，向後傳遞
   - 觀察經過多人傳遞後資訊的失真 → 這就是 RNN 的梯度消失
   - 然後給每位學生一張紙可以寫下來（= LSTM 的細胞狀態）

2. **門控值猜測遊戲（5 分鐘）**：
   - 展示一段文本，如："小明去了學校。他在那裡學了數學。[MASK] 很喜歡數學。"
   - 問學生：此時 LSTM 的遺忘門對「小明」應該開還是關？
   - 引導學生理解門控的直覺

### 程式碼教學注意事項
- PyTorch 的 `nn.LSTM` 接口初學者容易混淆：
  - 輸入形狀 `(seq_len, batch, input_size)` vs `batch_first=True` 的差異
  - 輸出包含 `(output, (h_n, c_n))` 三個部分
  - 建議先用 `batch_first=True` 降低混淆
- LSTM 門控值提取需要使用 `LSTMCell` 手動迴圈，建議在 Notebook 中提供模板

---

## 檢核點 Checkpoints

- [ ] 學生能解釋 RNN 與 FC 網路處理序列的差異
- [ ] 學生能用自己的話說明梯度消失問題
- [ ] 學生能描述 LSTM 的三個門各自的功能
- [ ] 學生能在 PyTorch 中建構並訓練一個 LSTM 模型
- [ ] 學生能理解 Self-Attention 的 Q/K/V 概念
- [ ] 學生能說出 Transformer 相較 RNN 的至少兩個優勢

---

## AI 助教設定 AI Tutor Configuration

本週助教設定為「進階引導模式」：
- 允許解釋概念性問題（如門控機制的直覺理解）
- 對 PyTorch 實作問題提供語法層級的幫助
- 對思考題**不直接給答案**，改為反問引導
  - 學生問：「為什麼 Transformer 比 RNN 好？」
  - 助教回：「你覺得 RNN 的計算是序列化的，這對 GPU 利用率有什麼影響？」
- 對門控值視覺化的解讀，提供結構化的思考框架
- 進階學生可引導閱讀原始論文

### 分層提示策略 (Hint Ladder) 本週範例

**問題：LSTM 門控值視覺化作業遇到困難**
1. Level 1 (概念)：「你知道 `nn.LSTM` 和 `nn.LSTMCell` 的差別嗎？提取門控值需要用哪個？」
2. Level 2 (方向)：「LSTMCell 需要你手動寫 for 迴圈處理每個時間步，這樣你就能在每步收集門控值了。」
3. Level 3 (框架)：「嘗試這個結構：初始化 h, c → for t in range(seq_len): h, c = lstm_cell(x[t], (h, c))，在迴圈中收集 h 和 c。」
4. Level 4 (範例)：提供 Notebook 中的程式碼片段參考

---

## 常見問題與排除 Troubleshooting

### Q1: LSTM 輸入維度錯誤
**錯誤訊息：** `Expected input size (seq_len, batch, input_size) but got...`
- 確認是否使用 `batch_first=True`
- 確認輸入形狀是否為 `(batch, seq_len, features)`
- 常見錯誤：忘記加 feature 維度，如 `(batch, seq_len)` 應改為 `(batch, seq_len, 1)`

### Q2: Loss 不收斂
- 檢查學習率是否太大（建議從 0.001 開始）
- 確認資料是否正確正規化 (Normalization)
- 確認目標值的設定是否正確（如預測下一步 vs 預測同步）
- 嘗試增加隱藏維度或層數

### Q3: CUDA out of memory
- 減小 batch_size
- 減小隱藏維度
- 確認是否有張量未正確 detach 導致計算圖累積

### Q4: 門控值提取失敗
- 建議使用 `nn.LSTMCell` 而非 `nn.LSTM`
- 替代方案：使用 `register_forward_hook` 捕獲中間值
- 注意 PyTorch 的 LSTM 內部實作是 fused kernel，直接存取門控值較困難

### Q5: Self-Attention 維度不匹配
- 確認 Q, K 的最後一維相同（$d_k$）
- 矩陣乘法 `Q @ K.T` 中注意轉置的維度
- 若使用 batch 維度，注意 `torch.bmm` vs `@` 的差異

### Q6: 學生對 Transformer 感到畏懼
- 強調本週 Transformer 只需理解**概念**，不需要完整實作
- 引導學生先理解 Self-Attention 的直覺（像 Google 搜尋：Query 是搜尋關鍵字，Key 是網頁標題，Value 是網頁內容）
- 推薦 Jay Alammar 的 "The Illustrated Transformer" 視覺化教學

---

## 差異化教學建議 Differentiated Instruction

### 對基礎較弱的學生
- 聚焦在 RNN 基本結構和 LSTM 的直覺理解
- 作業三（門控值視覺化）可以提供更多程式碼模板
- 思考題可以允許更短的回答或以口頭方式替代
- Transformer 部分僅要求理解「什麼是 Attention」的概念

### 對進階學生
- 鼓勵嘗試更複雜的序列任務（如文本生成、字元級語言模型）
- 引導閱讀 "Attention Is All You Need" 原始論文
- 可挑戰完整的 Transformer Encoder 實作
- 鼓勵探索最新的 SSM (State Space Model) 架構

---

## 與前後週次的銜接 Connection with Other Weeks

### 與前週 (Week 12 CNN) 的銜接
- 回顧 CNN 是如何處理「空間特徵」→ 本週處理「時間/序列特徵」
- 可以提及 CNN 也可以用於序列（1D CNN for text），但 RNN 更自然
- 殘差連接 (Residual Connection) 的概念在 LSTM 和 Transformer 中都有體現

### 與後週 (Week 14 訓練技巧) 的銜接
- 本週的模型訓練可能遇到的問題（如學習率選擇、過擬合）→ 下週會深入討論
- 提前預告：下週的學習率排程和早停技術可以改善本週的模型效果
- 梯度裁剪也是一種訓練技巧

### 與 Week 17 (LLM) 的銜接
- 本週的 Transformer 概念是 Week 17 LLM 的基礎
- 讓學生建立期待：理解 Transformer 後，就能理解 GPT/BERT 等大語言模型的架構

---

## 備課提醒 Preparation Notes

- 提前在課程環境中測試 PyTorch 的 LSTM/GRU 是否可正常運行
- 準備一份 Notebook 的「教師版」（含完整答案），以備學生求助時參考
- 準備好「傳話遊戲」所需的句子紙條（約 30 字的句子）
- 確認 Google Colab 上 PyTorch 版本是否支持本週使用的 API
- 建議提前練習在黑板上繪製 LSTM 細胞圖（複雜但重要）
- 準備 Jay Alammar 的 Illustrated Transformer 連結作為課後資源分發
- 本週內容較多，可根據班級程度調整 Transformer 部分的深度
