# 第 17 週教師手冊
# Week 17 Teacher Guide

## 時間分配 Time Allocation（90 分鐘）

| 時段 | 分鐘 | 活動 | 說明 |
|------|:---:|------|------|
| 開場 | 5 | 回顧與本週導入 | 從 Week 16 MLOps 連結到 LLM 應用 |
| 理論一 | 20 | LLM 概述 + Transformer 回顧 | Slide 1-6，重點在 Decoder-Only |
| 理論二 | 15 | 嵌入 + 語義搜尋 | Slide 7-11，現場 Demo 嵌入視覺化 |
| 理論三 | 15 | RAG 架構 | Slide 12-16，畫流程圖互動 |
| 實作 | 25 | Notebook 實作 | 嵌入視覺化 + 語義搜尋 + RAG |
| 理論四 | 5 | Prompt Engineering + 負責任使用 | Slide 17-24，可融入實作 |
| 總結 | 5 | 回顧 + 作業說明 + 期末專題預告 | Slide 25-26 |

---

## 教學重點 Key Teaching Points

### 核心觀念（務必傳達）
1. **LLM 的本質**：本質上是「下一個 Token 的預測器」。所有看似智慧的行為（推理、翻譯、撰寫）都源自大規模的模式學習。提醒學生不要過度神化也不要過度輕視。
2. **嵌入的直覺**：向量空間中的距離 = 語義距離。這個直覺是理解語義搜尋和 RAG 的基礎。
3. **RAG 的價值**：LLM 有知識截斷和幻覺問題，RAG 用「先查再答」的方式有效緩解。強調這是目前產業界最常用的 LLM 應用模式。
4. **Prompt Engineering 不是魔法**：好的提示設計是有系統和原則的工程實踐，不是靠運氣或「咒語」。
5. **負責任使用**：技術能力伴隨著責任。LLM 的輸出需要人類的判斷和監督。

### 教學策略
- **從熟悉到陌生**：先回顧 Transformer（Week 13-14 已學），再延伸到 LLM
- **類比教學**：
  - 嵌入 → 「把文字放到一個有座標的圖書館，相似的書放在附近」
  - RAG → 「開卷考試：先翻課本找資料，再根據找到的資料作答」
  - Prompt Engineering → 「和一個非常聰明但需要明確指示的助手溝通」
- **Demo 驅動**：理論講解後立刻用 Notebook 展示效果

---

## 教學流程詳細說明 Detailed Teaching Flow

### 開場（5 分鐘）

**連結前週**：
> "上週我們學了 MLOps，知道如何將模型部署到生產環境。今天我們要學習目前 AI 領域最熱門的主題——大型語言模型 (LLM)。你們日常使用的 ChatGPT、Claude、Gemini 背後就是 LLM。"

**引起動機**：
- 現場提問：「你們使用 ChatGPT 或 Claude 時，有沒有遇到它『信心滿滿地說錯』的情況？」
- 引出幻覺問題 → 帶出 RAG 的必要性

### 理論一：LLM 概述 + Transformer 回顧（20 分鐘）

**要點**：
1. LLM 的三個「大」：參數大、資料大、算力大
2. 快速回顧 Self-Attention（不需深入數學，重點在直覺）
3. **重點講解三種架構**：
   - 用表格比較 Encoder-Only / Decoder-Only / Encoder-Decoder
   - 強調「為什麼現在主流 LLM 都是 Decoder-Only」
4. Token 與上下文窗口的概念

**互動建議**：
- 請學生猜測：GPT-4 有多少參數？（答案：據傳約 1.8T，以 MoE 架構實現）
- 用因果遮罩的矩陣圖示解釋為什麼 Decoder-Only 適合生成

**常見誤解提醒**：
- 「LLM 理解語言」→ 更準確地說，LLM 學到了語言的統計規律
- 「參數越多越好」→ 不一定，還取決於訓練資料品質和訓練方法

### 理論二：嵌入 + 語義搜尋（15 分鐘）

**要點**：
1. 文字嵌入的概念：文字 → 向量
2. 餘弦相似度（可用手勢示範兩個向量的夾角）
3. 語義搜尋 vs 關鍵字搜尋（用具體例子展示差異）

**現場 Demo**（建議預先準備好 Notebook 結果）：
- 展示嵌入的 t-SNE/UMAP 降維圖，讓學生看到聚類效果
- 輸入一個查詢，展示語義搜尋找到的結果

**互動建議**：
- 讓學生各想一個搜尋查詢，現場測試語義搜尋的效果
- 問學生：「用關鍵字搜尋『汽車保養方法』能找到包含『轎車維修技巧』的文件嗎？語義搜尋呢？」

### 理論三：RAG 架構（15 分鐘）

**要點**：
1. LLM 的三大限制（知識截斷、幻覺、無專有知識）→ 引出 RAG 的動機
2. RAG 的三個階段：資料準備 → 查詢 → 生成
3. 文件切割策略（重點講解固定大小 + 重疊、遞迴分割）
4. 向量資料庫的概念（不需深入內部實作，重點在使用方式）

**互動建議**：
- 在白板上畫 RAG 流程圖，每畫一步就問學生「這一步在做什麼？」
- 開卷考試的類比：「如果你可以帶一本 1000 頁的課本進考場，但考試只有 60 分鐘，你會怎麼找答案？」→ 先看目錄/索引（= 向量搜尋），找到相關章節（= 檢索），然後根據內容作答（= 生成）

### 實作時間（25 分鐘）

**重點 Notebook 單元**：

| 優先序 | 單元 | 時間 | 說明 |
|:---:|------|:---:|------|
| 1 | 嵌入生成與視覺化 | 8 min | 讓學生看到嵌入的聚類效果 |
| 2 | 語義搜尋實作 | 7 min | 體驗語義搜尋與關鍵字搜尋的差異 |
| 3 | 簡易 RAG 系統 | 7 min | 跑通完整 RAG 流程 |
| 4 | Prompt Engineering 實驗 | 3 min | 快速展示 Zero-shot vs Few-shot |

**巡場重點**：
- 確認學生能成功安裝 `sentence-transformers`（如果網路有問題，提供離線模型）
- 如果有學生已有 API Key，協助他們執行 LLM API 呼叫
- 對沒有 API Key 的學生，說明可以用模擬方式完成作業

### 理論四 + 總結（10 分鐘）

**Prompt Engineering 快速講解**：
- Zero-shot vs Few-shot：用一個簡單的分類例子展示
- CoT：用數學題展示「一步一步想」的威力
- System Prompt：展示一個設計良好 vs 設計粗糙的 System Prompt 的差異

**負責任使用**：
- 強調學術誠信：使用 AI 工具需要標註
- 強調批判性思維：永遠不要盲目信任 LLM 的輸出
- 提到隱私風險：不要把敏感資料傳給 LLM API

**總結 + 作業說明**：
- 回顧今天的四大主題：LLM → 嵌入 → RAG → Prompt Engineering
- 說明作業要求與繳交方式
- **預告 Week 18 專題展示**：提醒學生準備期末專題

---

## 檢核點 Checkpoints

- [ ] 學生能解釋 Decoder-Only 架構與 Encoder-Only 的差異
- [ ] 學生能用自己的話解釋文字嵌入和餘弦相似度
- [ ] 學生理解 RAG 的三個階段（資料準備→查詢→生成）
- [ ] 學生能區分 Zero-shot、Few-shot 和 CoT 提示策略
- [ ] 學生成功在 Notebook 中生成嵌入並視覺化
- [ ] 學生理解 LLM 幻覺的風險及緩解策略
- [ ] 學生了解負責任使用 LLM 的原則

---

## AI 助教設定 AI Tutor Configuration

本週助教設定為「進階探索模式」：
- 可以回答 LLM、嵌入、RAG 的深入問題
- 引導學生思考 Prompt Engineering 的設計原則，而非直接給答案
- 提醒學生 LLM 輸出的不確定性——「這是我的分析，但建議你驗證一下」
- 鼓勵學生嘗試不同的提示策略，觀察輸出差異
- 對作業相關問題，採用引導式提問（如：「你覺得 chunk_size 設太大會有什麼影響？」）

---

## 常見問題與排除 Troubleshooting

### Q1: sentence-transformers 安裝失敗
```bash
# 方法一：使用 pip
pip install sentence-transformers

# 方法二：如果 torch 版本衝突
pip install sentence-transformers --no-deps
pip install transformers tokenizers huggingface-hub

# 方法三：使用 conda
conda install -c conda-forge sentence-transformers
```
- 如果學校網路慢，提前下載模型到本地共享資料夾
- 備用方案：使用 `scikit-learn` 的 `TfidfVectorizer` 作為簡易替代

### Q2: 沒有 LLM API Key
- **免費方案**：
  - Google Gemini API 有免費額度
  - Hugging Face Inference API 有免費方案（速度較慢）
  - Groq 提供免費的開源模型 API
- **模擬方案**：在 Notebook 中提供模擬 LLM 回應的函式，學生可以用模擬方式完成 RAG 流程
- **教師提供**：考慮申請教育用 API 額度，在課堂上提供臨時 Key（課後失效）

### Q3: Chroma 安裝問題
```bash
pip install chromadb
# 如果 sqlite3 版本過低（常見於舊版 Linux）
pip install pysqlite3-binary
```
- 備用方案：直接使用 NumPy 手動計算餘弦相似度（Notebook 中有提供此方案）

### Q4: 嵌入模型下載太慢
- 使用 Hugging Face 鏡像站：
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```
- 預先下載模型到本地，讓學生從區域網路載入

### Q5: 學生問「LLM 是如何訓練的？」
- 簡要解釋預訓練 (Pre-training) + 微調 (Fine-tuning) + RLHF
- 不需深入 RLHF 的細節，但可以提到「人類回饋」讓模型更安全、更有用
- 指向延伸閱讀資源

### Q6: 學生問「嵌入維度為什麼是 384/768/1536？」
- 與 Transformer 的隱藏層維度 (Hidden Dimension) 有關
- 更高維度通常能捕捉更豐富的語義，但計算成本也更高
- 在多數任務上，384 維已足夠使用

---

## 備課提醒 Preparation Notes

### 課前準備
- [ ] 確認教室網路暢通（嵌入模型和 API 需要網路）
- [ ] 預先在教師電腦下載好 `all-MiniLM-L6-v2` 模型
- [ ] 準備好 Notebook 的預執行版本（含所有輸出），以備網路問題
- [ ] 如計畫提供 API Key，提前申請並設定用量限制
- [ ] 準備 2-3 個有趣的 RAG Demo 案例（如用課程教材建構 RAG）

### 教學材料
- [ ] Notebook 中的所有程式碼在教師電腦上測試通過
- [ ] 投影片中的程式碼範例可直接複製到 Notebook 執行
- [ ] 準備一份 API Key 安全使用的提醒文件（可投影）

### 心理準備
- 本週主題對學生來說可能最「接地氣」（因為日常已在使用 LLM），但理論基礎容易被忽略
- 強調「知其然也要知其所以然」——理解 LLM 的原理才能更好地使用它
- 期末前倒數第二週，學生可能已有期末壓力，課程節奏可適度調整
- 鼓勵學生將本週學到的 RAG 或 Prompt Engineering 融入期末專題

### 與期末專題的銜接
- 提醒學生 Week 18 是專題展示
- LLM/RAG/嵌入可以是專題的一部分（如建構某個領域的 RAG 問答系統）
- 提供幾個可能的專題方向：
  1. 用課程教材建構 RAG 問答系統
  2. 比較不同嵌入模型在特定領域的表現
  3. 設計一個特定用途的 Prompt Engineering 工具
  4. 結合本學期學到的 ML/DL 技術，搭配 LLM 做應用
