# 第 17 週投影片：LLM 與嵌入應用（檢索增強、提示工程基礎）

---

## Slide 1: 本週主題
# LLM 與嵌入應用
### 檢索增強生成 (RAG) 與提示工程 (Prompt Engineering)
- 從 Transformer 到 LLM
- 文字嵌入與語義搜尋
- RAG 架構全解析
- Prompt Engineering 實戰技巧

---

## Slide 2: 學習路線圖
### 第 11-17 週 深度學習旅程
```
Week 11: 神經網路基礎
Week 12: CNN 視覺化
Week 13-14: RNN / Transformer
Week 15: 訓練技巧
Week 16: 模型評估 & MLOps
Week 17: LLM 與嵌入應用 ← 你在這裡
Week 18: 專題展示
```
> 今天我們站在巨人的肩膀上，用 Transformer 的力量解決真實世界的問題。

---

## Slide 3: 什麼是 LLM？
### Large Language Model 大型語言模型
| 維度 | 規模 |
|:-:|:-:|
| 參數量 | 數十億至數千億 |
| 訓練資料 | 數兆個 Token |
| 計算量 | 數千 GPU × 數月 |

核心思想：**在海量文本上學習語言規律，湧現多種能力**

---

## Slide 4: LLM 演化簡史
```
N-gram → Word2Vec → ELMo → BERT → GPT → ChatGPT → Claude
(1990s)  (2013)    (2018) (2018)  (2018) (2022)    (2023-)
```
關鍵轉折點：**2017 年 Transformer 論文 "Attention Is All You Need"**

---

## Slide 5: 三種 Transformer 架構
| | Encoder-Only | Decoder-Only | Encoder-Decoder |
|:-:|:-:|:-:|:-:|
| 代表 | BERT | GPT, Claude | T5, BART |
| 注意力 | 雙向 | 因果（單向） | 雙向 + 因果 |
| 任務 | 理解 | **生成** | 序列轉換 |
| 主流? | | **目前主流 LLM** | |

---

## Slide 6: 為什麼 Decoder-Only 勝出？
### 因果注意力 (Causal Attention)
```
Token1  [1 0 0 0]   ← 只看自己
Token2  [1 1 0 0]   ← 看 1, 2
Token3  [1 1 1 0]   ← 看 1, 2, 3
Token4  [1 1 1 1]   ← 看全部
```
- 天然適合**自迴歸生成**（逐 Token 預測）
- 預訓練目標簡單：Next Token Prediction
- 規模擴大 → 理解能力自然湧現

---

## Slide 7: Token 與上下文窗口
### Tokenization 示意
```
"深度學習" → ["深度", "學習"]     (2 Tokens)
"Deep learning" → ["Deep", " learning"]  (2 Tokens)
```
- 上下文窗口 (Context Window) = LLM 一次能看多少 Token
- Claude 3.5 Sonnet：200K Token
- GPT-4o：128K Token

---

## Slide 8: 文字嵌入 Text Embeddings
### 將文字映射為向量
```
"貓是寵物"     → [0.21, -0.53, 0.87, ...]  ─┐ 相近
"狗是好夥伴"   → [0.19, -0.48, 0.82, ...]  ─┘

"量子力學"     → [-0.72, 0.31, -0.15, ...] ← 遙遠
```
語義相似 → 向量距離近
語義不同 → 向量距離遠

---

## Slide 9: 餘弦相似度 Cosine Similarity
$$\text{cos}(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$$

| 值 | 意義 |
|:-:|:-:|
| 1.0 | 完全相同方向 |
| 0.0 | 正交（無關） |
| -1.0 | 完全相反 |

> 比歐幾里得距離更適合高維向量：只看「方向」不看「長度」

---

## Slide 10: 語義搜尋 vs. 關鍵字搜尋
| | 關鍵字搜尋 | 語義搜尋 |
|:-:|:-:|:-:|
| "汽車保養" 能否找到 "轎車維修"？ | 不能 | **能** |
| 匹配方式 | 字面匹配 | 語義理解 |
| 核心技術 | TF-IDF / BM25 | Embedding + ANN |

---

## Slide 11: 語義搜尋流程
### 兩個階段
```
離線：文件 → 切割 → 嵌入 → 向量資料庫
線上：查詢 → 嵌入 → 向量搜尋 → Top-K 結果
```

---

## Slide 12: 為什麼需要 RAG？
### LLM 的四大痛點
1. **知識截斷** — 不知道訓練截止後的事
2. **幻覺** — 可能自信地說錯
3. **無專有知識** — 不知道你公司的內部文件
4. **無法引用** — 說不出答案從哪來

> RAG = 先搜尋，再回答

---

## Slide 13: RAG 架構全景
```
[資料準備]
  文件 → Chunking → Embedding → Vector DB

[查詢]
  問題 → Embedding → 搜尋 → Top-K 段落

[生成]
  System Prompt + 段落 + 問題 → LLM → 帶引用的回答
```

---

## Slide 14: 文件切割策略 Chunking
| 策略 | 特色 |
|:-:|:-:|
| 固定大小 | 簡單但可能斷句 |
| 重疊切割 | 減少上下文丟失 |
| 句子分割 | 保持句子完整 |
| 語義分割 | 品質最佳，計算量大 |
| 遞迴分割 | **推薦：平衡品質與效率** |

建議參數：chunk_size=500, overlap=100

---

## Slide 15: 向量資料庫 Vector Database
| 名稱 | 特色 | 場景 |
|:-:|:-:|:-:|
| **Chroma** | 輕量 Python 原生 | **學習、原型** |
| Faiss | 高效 ANN | 大規模搜尋 |
| Pinecone | 全託管 | 企業生產 |
| pgvector | PostgreSQL 擴充 | SQL 整合 |

---

## Slide 16: RAG 程式碼概念
```python
# 1. 檢索
results = collection.query(
    query_texts=["使用者問題"],
    n_results=5
)
# 2. 組合提示
prompt = f"根據資料回答：{context}\n問題：{question}"
# 3. 生成
response = llm.generate(prompt)
```

---

## Slide 17: 提示工程 Prompt Engineering
### 提示的六大元素
```
┌─ 系統提示 (角色、規則)
├─ 上下文 (參考資料)
├─ 範例 (Few-shot)
├─ 指令 (任務描述)
├─ 輸入 (使用者內容)
└─ 輸出格式 (JSON、Markdown...)
```

---

## Slide 18: Zero-shot vs. Few-shot
### Zero-shot（零範例）
```
翻譯：「深度學習」→ ?
```

### Few-shot（提供範例）
```
範例1: CNN → 卷積神經網路
範例2: RNN → 遞迴神經網路
問題: GAN → ?
```

**Few-shot 讓 LLM 透過上下文學習 (ICL) 掌握模式**

---

## Slide 19: 思維鏈 Chain-of-Thought
### 讓 LLM「一步一步想」
```
不使用 CoT:
  23 - 20 + 6 = ?  →  "9"（有時答錯）

使用 CoT:
  「讓我們一步一步思考。」
  1. 一開始 23 顆
  2. 用掉 20 顆：23 - 20 = 3
  3. 買了 6 顆：3 + 6 = 9
  → 答案正確率大幅提升
```

---

## Slide 20: 系統提示設計原則
### 五大核心元素
1. **角色定義** — "你是資深 Python 工程師"
2. **行為準則** — "不確定時說明"
3. **知識範圍** — "只回答 ML 相關問題"
4. **輸出格式** — "以 Markdown 呈現"
5. **安全限制** — "不生成有害內容"

---

## Slide 21: LLM API 使用
### Anthropic Claude
```python
client = anthropic.Anthropic(api_key=os.getenv("KEY"))
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "..."}]
)
```
### OpenAI GPT
```python
client = OpenAI(api_key=os.getenv("KEY"))
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "..."}]
)
```

---

## Slide 22: Temperature 效果
| 溫度 | 特性 | 適用 |
|:-:|:-:|:-:|
| 0.0 | 確定、一致 | 事實問答 |
| 0.3 | 低創意 | 程式碼 |
| 0.7 | 適度創意 | 一般對話 |
| 1.0 | 高度多樣 | 創意寫作 |

> Temperature 越高 → 越有創意，但幻覺風險也越高

---

## Slide 23: 幻覺 Hallucination
### LLM 可能「自信地說錯」
| 類型 | 範例 |
|:-:|:-:|
| 事實性 | 虛構的論文引用 |
| 忠實性 | 忽略提供的參考資料 |
| 一致性 | 前後自相矛盾 |

### 緩解策略
RAG / 降低 Temperature / 要求引用來源 / 人工審查

---

## Slide 24: 負責任使用 LLM
### 五大原則
1. **透明** — 告知使用者這是 AI 生成
2. **可驗證** — 提供資訊來源
3. **隱私** — 不傳送敏感資料
4. **公平** — 注意偏見
5. **問責** — 使用者對結果負責

### 學術誠信
- 標註 AI 輔助
- 理解而非複製
- 保持批判性思維

---

## Slide 25: 今日實作預告
### Notebook 實作內容
1. 文字嵌入生成與 t-SNE/UMAP 視覺化
2. 語義搜尋實作
3. 簡易 RAG 系統搭建
4. Prompt Engineering 實驗
5. LLM API 呼叫

---

## Slide 26: 本週重點回顧
### Key Takeaways
1. **LLM** = 大規模 Transformer（主要 Decoder-Only）+ 海量訓練
2. **嵌入** = 文字 → 向量，捕捉語義
3. **RAG** = 檢索 + 增強 + 生成，解決知識截斷與幻覺
4. **Prompt Engineering** = Zero/Few-shot, CoT, System Prompt
5. **負責任使用** = 透明、可驗證、保持批判性思維

### 下週預告
第 18 週：專題展示 — 將整學期所學整合成完整專案！
