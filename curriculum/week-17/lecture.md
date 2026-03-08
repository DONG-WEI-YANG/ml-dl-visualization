# 第 17 週：LLM 與嵌入應用（檢索增強、提示工程基礎）
# Week 17: LLM & Embedding Applications (RAG, Prompt Engineering Basics)

## 學習目標 Learning Objectives
1. 理解大型語言模型 (Large Language Model, LLM) 的基本原理與發展脈絡
2. 回顧 Transformer 架構，區分 Encoder-Decoder 與 Decoder-Only 架構
3. 掌握文字嵌入 (Text Embeddings) 的概念與向量空間表示
4. 了解語義搜尋 (Semantic Search) 的原理與實作方法
5. 理解檢索增強生成 (Retrieval-Augmented Generation, RAG) 的完整架構
6. 掌握提示工程 (Prompt Engineering) 的基礎技巧
7. 學會使用 LLM API（Anthropic Claude / OpenAI GPT）進行應用開發
8. 認識 LLM 的限制與負責任使用原則

---

## 1. 大型語言模型概述 Overview of Large Language Models

### 1.1 什麼是 LLM？ What is an LLM?

大型語言模型 (Large Language Model, LLM) 是一種基於深度學習的語言模型，使用大量文本資料進行預訓練 (Pre-training)，能夠理解和生成自然語言。「大型」主要體現在三個維度：

| 維度 | 說明 | 範例 |
|------|------|------|
| 參數量 Parameters | 模型的可學習權重數量 | GPT-3: 175B, Claude 3: 未公開, Llama 3: 70B |
| 訓練資料量 Training Data | 預訓練所使用的文本語料規模 | 數兆 (Trillion) 個 Token |
| 計算量 Compute | 訓練所需的算力 (FLOPs) | 數千 GPU 運算數週至數月 |

> "LLM 的核心能力來自於在海量文本上學習語言的統計規律，從而湧現出 (Emergent) 推理、翻譯、摘要、程式撰寫等多種能力。"

### 1.2 LLM 的發展歷程 Evolution of LLMs

```
統計語言模型 (Statistical LM)
  └── N-gram 模型 (1990s-2000s)
        └── 神經語言模型 (Neural LM)
              ├── Word2Vec / GloVe (2013-2014) — 靜態詞嵌入
              ├── ELMo (2018) — 上下文相關嵌入
              └── Transformer 時代 (2017-)
                    ├── BERT (2018) — Encoder-Only, 雙向理解
                    ├── GPT 系列 (2018-) — Decoder-Only, 自迴歸生成
                    ├── T5 (2020) — Encoder-Decoder, 文字到文字
                    ├── ChatGPT / GPT-4 (2022-2023)
                    ├── Claude 系列 (2023-)
                    └── 開源模型：Llama, Mistral, Gemma...
```

### 1.3 LLM 的核心能力 Core Capabilities

LLM 展現了多種令人驚豔的能力，其中許多是**湧現能力 (Emergent Abilities)**——只在模型規模超過某個閾值後才出現：

| 能力 | 說明 | 範例 |
|------|------|------|
| 文本生成 Text Generation | 根據提示生成連貫文本 | 文章撰寫、故事創作 |
| 語言理解 Language Understanding | 理解文本的語義與意圖 | 情感分析、意圖識別 |
| 推理 Reasoning | 進行邏輯推理與問題解決 | 數學題、邏輯謎題 |
| 翻譯 Translation | 跨語言轉換 | 中英互譯 |
| 程式撰寫 Coding | 生成和理解程式碼 | 函式撰寫、程式除錯 |
| 摘要 Summarization | 壓縮長文本為簡短摘要 | 文件摘要、會議記錄 |
| 上下文學習 In-Context Learning | 從提示中的範例學習新任務 | Few-shot Prompting |

### 1.4 Token 與 Tokenization

LLM 不是直接處理「文字」，而是處理 **Token**——文本被拆分後的最小單位。

```
文字: "機器學習很有趣"
Tokenization 結果（示意）: ["機器", "學習", "很", "有趣"]

文字: "Machine learning is fun"
Tokenization 結果（示意）: ["Machine", " learning", " is", " fun"]
```

常見的分詞 (Tokenization) 演算法：

| 方法 | 說明 |
|------|------|
| BPE (Byte Pair Encoding) | 從字元開始，反覆合併最常共現的 pair |
| WordPiece | 類似 BPE，但使用似然度 (Likelihood) 決定合併順序 |
| SentencePiece | 統一處理不同語言，不依賴空格分詞 |

**重要概念**：LLM 的輸入和輸出都有 Token 數量限制，稱為**上下文窗口 (Context Window)**。例如 Claude 3.5 Sonnet 支援 200K Token 的上下文窗口。

---

## 2. Transformer 架構回顧 Transformer Architecture Review

### 2.1 Self-Attention 機制回顧 Self-Attention Recap

Transformer 的核心是**自注意力機制 (Self-Attention)**，在第 13-14 週已有介紹，此處做重點回顧。

Self-Attention 允許序列中的每個位置「關注」序列中所有其他位置，計算公式為：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q$ (Query)：查詢矩陣，代表「我在找什麼」
- $K$ (Key)：鍵矩陣，代表「我有什麼」
- $V$ (Value)：值矩陣，代表「我的內容是什麼」
- $d_k$：Key 的維度，用於縮放 (Scaling) 以避免 softmax 梯度消失

**Multi-Head Attention** 則是將注意力分成多個「頭 (Head)」，讓模型同時學習不同類型的注意力模式：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

### 2.2 三種 Transformer 架構變體 Three Transformer Variants

原始 Transformer (Vaswani et al., 2017) 包含 Encoder 和 Decoder。在 LLM 的演進中，衍生出三種主要變體：

#### (1) Encoder-Only（雙向理解型）

```
輸入 Token 序列
  → [Encoder] × N 層
  → 上下文表示 (Contextual Representations)
  → 下游任務 Head（分類、NER 等）
```

- **代表模型**：BERT, RoBERTa, ALBERT, DeBERTa
- **注意力類型**：雙向 (Bidirectional)——每個 Token 可以看到前後所有 Token
- **預訓練任務**：遮蔽語言模型 (Masked Language Model, MLM)——隨機遮蔽 15% Token，讓模型預測
- **優勢**：深度語言理解，適合分類、序列標註 (Sequence Labeling) 等理解任務
- **限制**：不擅長生成長文本

#### (2) Decoder-Only（自迴歸生成型）

```
輸入 Token 序列
  → [Decoder] × N 層（帶因果遮罩 Causal Mask）
  → 下一個 Token 機率分布
  → 自迴歸生成 (Autoregressive Generation)
```

- **代表模型**：GPT 系列、Claude 系列、Llama、Mistral
- **注意力類型**：因果注意力 (Causal Attention)——每個 Token 只能看到自己和之前的 Token
- **預訓練任務**：下一個 Token 預測 (Next Token Prediction)
- **優勢**：強大的文本生成能力，適合對話、寫作、程式生成
- **為什麼主流 LLM 幾乎都是 Decoder-Only？**
  - 與生成任務天然契合
  - 預訓練目標簡單且可擴展 (Scalable)
  - 透過擴大規模 (Scaling) 可以湧現理解能力

```
因果遮罩 (Causal Mask) 的注意力矩陣：

        Token1  Token2  Token3  Token4
Token1  [  1      0      0      0  ]    ← 只看自己
Token2  [  1      1      0      0  ]    ← 看 1,2
Token3  [  1      1      1      0  ]    ← 看 1,2,3
Token4  [  1      1      1      1  ]    ← 看 1,2,3,4
```

#### (3) Encoder-Decoder（序列到序列型）

```
輸入序列
  → [Encoder] × N 層 → 上下文表示
                          ↓ (Cross-Attention)
輸出序列 → [Decoder] × M 層 → 生成結果
```

- **代表模型**：T5, BART, mBART, Flan-T5
- **注意力類型**：Encoder 雙向 + Decoder 因果 + Cross-Attention（Decoder 關注 Encoder 輸出）
- **預訓練任務**：去噪自編碼 (Denoising Autoencoder)——破壞輸入並重建
- **優勢**：適合輸入與輸出結構不同的任務（翻譯、摘要）

### 2.3 三種架構比較 Comparison

| 特性 | Encoder-Only | Decoder-Only | Encoder-Decoder |
|------|:-----------:|:------------:|:---------------:|
| 代表模型 | BERT | GPT, Claude | T5, BART |
| 注意力方向 | 雙向 | 單向（因果） | 雙向 + 因果 |
| 適合任務 | 理解（分類、NER） | 生成（對話、寫作） | 序列轉換（翻譯、摘要） |
| 目前主流 LLM | 否 | **是** | 否 |

---

## 3. 文字嵌入與向量空間 Text Embeddings & Vector Space

### 3.1 什麼是文字嵌入？ What are Text Embeddings?

文字嵌入 (Text Embeddings) 是將文字（詞、句子、段落、文件）映射為固定長度的稠密向量 (Dense Vector) 的技術。嵌入向量捕捉了文字的**語義資訊 (Semantic Information)**。

```
"貓是一種寵物"   → [0.21, -0.53, 0.87, ..., 0.12]   (維度: 768 或 1536)
"狗是人類的好朋友" → [0.19, -0.48, 0.82, ..., 0.15]   ← 語義相近，向量相近
"量子力學的基礎"   → [-0.72, 0.31, -0.15, ..., 0.88]  ← 語義遠離，向量遠離
```

### 3.2 從詞嵌入到句子嵌入 From Word to Sentence Embeddings

嵌入技術經歷了幾個重要階段：

| 時期 | 方法 | 特性 |
|------|------|------|
| 2013 | Word2Vec (Skip-gram, CBOW) | 靜態詞嵌入，每個詞只有一個向量 |
| 2014 | GloVe (Global Vectors) | 基於共現矩陣的靜態詞嵌入 |
| 2018 | ELMo | 上下文相關的詞嵌入（同一個詞在不同句子中有不同向量） |
| 2018 | BERT Embeddings | 使用 Transformer 的上下文嵌入 |
| 2019+ | Sentence-BERT, E5, BGE | 專門為句子/段落設計的嵌入模型 |
| 2023+ | OpenAI Embeddings, Voyage, Cohere Embed | 商用嵌入 API |

### 3.3 嵌入模型的訓練 How Embedding Models are Trained

現代嵌入模型通常使用**對比學習 (Contrastive Learning)** 進行訓練：

```
正樣本對 (Positive Pair): 語義相似的句子 → 拉近向量距離
負樣本對 (Negative Pair): 語義不同的句子 → 推遠向量距離

損失函數 (Loss Function):
  InfoNCE / Contrastive Loss
  目標：最大化正樣本的相似度，最小化負樣本的相似度
```

### 3.4 向量空間的性質 Properties of Vector Space

嵌入向量空間具有以下重要性質：

**1. 語義距離 (Semantic Distance)**

語義相似的文字在向量空間中距離較近。常用的距離/相似度度量：

$$
\text{Cosine Similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}
$$

$$
\text{Euclidean Distance}(\mathbf{a}, \mathbf{b}) = ||\mathbf{a} - \mathbf{b}||_2 = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}
$$

$$
\text{Dot Product}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \cdot b_i
$$

**2. 語義算術 (Semantic Arithmetic)**

經典例子（詞嵌入）：

$$
\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}
$$

**3. 聚類結構 (Cluster Structure)**

相同主題的文字會在向量空間中形成自然的聚類 (Clusters)。這個特性可以透過 t-SNE 或 UMAP 降維後視覺化觀察。

### 3.5 主流嵌入模型比較 Popular Embedding Models

| 模型 | 提供者 | 維度 | 特色 |
|------|--------|:---:|------|
| text-embedding-3-small | OpenAI | 1536 | 高性價比，適合一般用途 |
| text-embedding-3-large | OpenAI | 3072 | 更高精度 |
| voyage-3 | Voyage AI | 1024 | 擅長程式碼和技術文件 |
| embed-v4 | Cohere | 1024 | 多語言支持佳 |
| BGE-M3 | BAAI | 1024 | 開源，多語言，支持稀疏+密集 |
| all-MiniLM-L6-v2 | Sentence-Transformers | 384 | 開源，輕量快速 |

---

## 4. 語義搜尋 Semantic Search

### 4.1 傳統搜尋 vs. 語義搜尋 Traditional vs. Semantic Search

| 特性 | 關鍵字搜尋 (Keyword Search) | 語義搜尋 (Semantic Search) |
|------|:-:|:-:|
| 匹配方式 | 字面匹配 (Lexical Match) | 語義匹配 (Semantic Match) |
| 處理同義詞 | 差——"汽車" 搜不到 "轎車" | 好——理解語義相似性 |
| 常用演算法 | TF-IDF, BM25 | 向量近鄰搜尋 (ANN) |
| 代表系統 | Elasticsearch (BM25) | 向量資料庫 + 嵌入模型 |
| 計算成本 | 低 | 較高（需要嵌入計算） |

### 4.2 語義搜尋的流程 Semantic Search Pipeline

```
離線索引階段 (Offline Indexing):
  文件庫 → 切割 (Chunking) → 嵌入模型 → 向量 → 儲存到向量資料庫

線上查詢階段 (Online Query):
  使用者查詢 → 嵌入模型 → 查詢向量 → 在向量資料庫中做近鄰搜尋
  → 返回 Top-K 最相似的文件段落
```

### 4.3 近似最近鄰搜尋 Approximate Nearest Neighbor (ANN)

在高維向量空間中做精確最近鄰搜尋 (Exact Nearest Neighbor) 的時間複雜度是 $O(n \cdot d)$（n 為向量數，d 為維度），這在大規模資料上不可行。因此實務上使用**近似最近鄰搜尋 (ANN)**。

常見的 ANN 演算法：

| 演算法 | 說明 | 代表實作 |
|--------|------|---------|
| HNSW (Hierarchical Navigable Small World) | 建立多層圖結構，在圖上導航搜尋 | Faiss, Qdrant |
| IVF (Inverted File Index) | 先用聚類分區，搜尋時只查相關分區 | Faiss |
| Product Quantization (PQ) | 壓縮向量以節省記憶體 | Faiss |
| Locality-Sensitive Hashing (LSH) | 使用雜湊函式將相近向量映射到同一桶 | — |

### 4.4 混合搜尋 Hybrid Search

實務上常將關鍵字搜尋與語義搜尋**結合**使用，以兼顧精確匹配和語義理解：

```
查詢 → ┌ 關鍵字搜尋 (BM25)   → 候選集 A ─┐
       └ 語義搜尋 (Embedding) → 候選集 B ─┤
                                           → 融合排序 (Reciprocal Rank Fusion)
                                           → 最終排序結果
```

---

## 5. 檢索增強生成 Retrieval-Augmented Generation (RAG)

### 5.1 為什麼需要 RAG？ Why RAG?

LLM 雖然強大，但存在以下限制：

| 限制 | 說明 |
|------|------|
| 知識截斷 (Knowledge Cutoff) | 模型只知道訓練資料截止日期前的知識 |
| 幻覺 (Hallucination) | 可能自信地生成看似正確但實際錯誤的內容 |
| 缺乏專有知識 (No Domain Data) | 無法存取企業內部文件、最新資訊 |
| 無法引用來源 (No Citations) | 單純的 LLM 無法提供答案的出處 |

RAG 透過**先檢索相關資料，再讓 LLM 根據這些資料生成回答**，有效解決上述問題。

> RAG = Retrieval（檢索相關知識）+ Augmented（將知識注入提示）+ Generation（LLM 生成回答）

### 5.2 RAG 系統的完整架構 Complete RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 系統架構                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1. 資料準備階段 Data Preparation]                           │
│                                                             │
│  原始文件          文件切割           嵌入計算         向量儲存  │
│  (PDF/HTML/MD) → (Chunking) →   (Embedding) →   (Vector DB) │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [2. 查詢階段 Query Phase]                                    │
│                                                             │
│  使用者問題 → 嵌入查詢 → 向量搜尋 → Top-K 相關段落              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [3. 生成階段 Generation Phase]                                │
│                                                             │
│  系統提示 + 檢索到的段落 + 使用者問題 → LLM → 帶引用的回答      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 文件切割 Document Chunking

文件切割是 RAG 流程中的關鍵步驟。切割策略直接影響檢索的品質。

#### 為什麼需要切割？

1. 嵌入模型有**最大 Token 限制**（通常 512 Token）
2. 太長的文件段落會稀釋語義（一段涵蓋太多主題）
3. LLM 的上下文窗口有限，需要精選最相關的段落

#### 常見的切割策略 Common Chunking Strategies

| 策略 | 說明 | 優缺點 |
|------|------|--------|
| 固定大小 (Fixed Size) | 每 N 個字元或 Token 切一段 | 簡單但可能切斷句子 |
| 重疊切割 (Overlapping) | 固定大小 + 相鄰 chunk 有重疊 | 減少上下文丟失，但增加儲存量 |
| 句子分割 (Sentence) | 按句號/換行分割，再組合至目標大小 | 保持句子完整性 |
| 語義分割 (Semantic) | 使用嵌入偵測語義邊界 | 品質最好但計算量大 |
| 遞迴分割 (Recursive) | 依階層分隔符（段落→句子→字詞）遞迴切割 | LangChain 預設方法，平衡品質與效率 |
| 結構化分割 (Structural) | 根據文件結構（標題、章節）切割 | 適合結構化文件（Markdown, HTML） |

#### 切割參數建議 Recommended Parameters

```
chunk_size (切割大小): 200-1000 Token（常見 500）
chunk_overlap (重疊大小): chunk_size 的 10%-20%（常見 50-100 Token）

範例：
  chunk_size = 500 Token
  chunk_overlap = 100 Token
  → 每個 chunk 500 Token，相鄰 chunk 重疊 100 Token
```

### 5.4 向量資料庫 Vector Database

向量資料庫 (Vector Database) 是專門儲存和檢索高維向量的資料庫系統。

#### 主流向量資料庫比較

| 資料庫 | 類型 | 特色 | 適用場景 |
|--------|------|------|---------|
| Chroma | 嵌入式 (Embedded) | 輕量，Python 原生，適合原型 | 學習、小型專案 |
| Faiss (Facebook) | 函式庫 (Library) | 高效 ANN，GPU 加速 | 大規模搜尋 |
| Pinecone | 雲端託管 (Managed) | 全託管服務，無需運維 | 企業生產環境 |
| Qdrant | 開源 + 雲端 | Rust 高效能，豐富的過濾功能 | 需要 metadata 過濾 |
| Weaviate | 開源 + 雲端 | GraphQL API，支援混合搜尋 | 複雜查詢需求 |
| Milvus | 開源 | 分散式架構，十億級向量 | 超大規模部署 |
| pgvector | PostgreSQL 擴充 | 與 SQL 整合 | 已有 PostgreSQL 基礎架構 |

#### 向量資料庫的核心操作

```python
# 概念程式碼（以 Chroma 為例）
import chromadb

# 1. 建立/連接資料庫
client = chromadb.Client()
collection = client.create_collection("my_docs")

# 2. 新增向量（Upsert）
collection.add(
    documents=["文件段落1", "文件段落2", ...],
    ids=["id1", "id2", ...],
    metadatas=[{"source": "file1.pdf"}, ...],  # 元資料
)

# 3. 查詢（語義搜尋）
results = collection.query(
    query_texts=["使用者的問題"],
    n_results=5  # 返回 Top-5
)
```

### 5.5 檢索 + 生成流程 Retrieval + Generation Pipeline

完整的 RAG 查詢流程：

```python
# 概念程式碼
def rag_query(user_question, collection, llm_client):
    # Step 1: 檢索相關段落
    results = collection.query(
        query_texts=[user_question],
        n_results=5
    )
    retrieved_docs = results['documents'][0]

    # Step 2: 組合提示
    context = "\n\n".join(retrieved_docs)
    prompt = f"""根據以下參考資料回答問題。如果資料中沒有相關資訊，請說明無法回答。

參考資料：
{context}

問題：{user_question}

回答："""

    # Step 3: 呼叫 LLM 生成回答
    response = llm_client.generate(prompt)
    return response
```

### 5.6 RAG 的評估 RAG Evaluation

| 評估維度 | 指標 | 說明 |
|---------|------|------|
| 檢索品質 Retrieval Quality | Precision@K, Recall@K, MRR | 檢索到的段落是否相關 |
| 生成品質 Generation Quality | Faithfulness (忠實度) | 回答是否忠實於檢索到的資料 |
| 生成品質 Generation Quality | Answer Relevance (相關性) | 回答是否切合問題 |
| 端到端 End-to-End | Correctness (正確性) | 最終答案是否正確 |

### 5.7 進階 RAG 技術 Advanced RAG Techniques

基本 RAG 可能面臨檢索品質不佳的問題。以下是常見的改進方向：

| 技術 | 說明 |
|------|------|
| 查詢改寫 (Query Rewriting) | 用 LLM 將使用者問題改寫為更適合搜尋的形式 |
| 假設文件嵌入 (HyDE) | 先讓 LLM 生成假設性答案，用答案的嵌入去搜尋 |
| 多步檢索 (Multi-step Retrieval) | 先粗篩再精排 |
| 重新排序 (Re-ranking) | 用 Cross-Encoder 對候選段落重新排序 |
| 查詢分解 (Query Decomposition) | 將複雜問題拆解為多個子問題，分別檢索 |
| 知識圖譜增強 (KG-augmented) | 結合知識圖譜提供結構化推理 |

---

## 6. 提示工程基礎 Prompt Engineering Basics

### 6.1 什麼是提示工程？ What is Prompt Engineering?

提示工程 (Prompt Engineering) 是設計和優化輸入提示 (Prompt) 以引導 LLM 產生期望輸出的技術。好的提示可以顯著提升 LLM 的輸出品質。

> "Prompt Engineering 不是讓 LLM 做它不會的事，而是有效地引導它發揮已有的能力。"

### 6.2 提示的基本結構 Basic Prompt Structure

一個完整的提示通常包含以下元素：

```
┌──────────────────────────┐
│  系統提示 System Prompt    │  ← 定義角色、行為準則、輸出格式
├──────────────────────────┤
│  上下文 Context           │  ← 背景資訊、參考資料（RAG 檢索結果）
├──────────────────────────┤
│  範例 Examples            │  ← Few-shot 範例
├──────────────────────────┤
│  指令 Instruction         │  ← 具體任務描述
├──────────────────────────┤
│  輸入 Input              │  ← 使用者提供的具體內容
├──────────────────────────┤
│  輸出格式 Output Format   │  ← 期望的輸出結構（JSON、Markdown 等）
└──────────────────────────┘
```

### 6.3 零樣本提示 Zero-shot Prompting

**定義**：不提供任何範例，直接要求 LLM 執行任務。

```
提示：
  將以下文字翻譯成英文：
  「深度學習是機器學習的一個子領域。」

LLM 回應：
  "Deep learning is a subfield of machine learning."
```

**適用場景**：
- 任務描述清晰且 LLM 已具備相關能力
- 簡單的分類、翻譯、摘要任務

### 6.4 少樣本提示 Few-shot Prompting

**定義**：在提示中提供幾個輸入-輸出範例，讓 LLM 透過**上下文學習 (In-Context Learning, ICL)** 理解任務模式。

```
提示：
  將技術術語翻譯成日常用語。

  技術術語：TCP/IP 三向交握
  日常用語：兩台電腦互相確認彼此都準備好了，才開始傳資料

  技術術語：遞迴函式
  日常用語：一個函式自己呼叫自己，像是俄羅斯套娃

  技術術語：負載均衡
  日常用語：
```

**要點**：
- 範例數量通常 2-5 個效果最佳
- 範例應涵蓋不同的情況和邊界案例
- 範例的格式和品質直接影響輸出

### 6.5 思維鏈 Chain-of-Thought (CoT) Prompting

**定義**：引導 LLM 在回答前先展示推理過程，從而提升複雜推理任務的準確率。

#### Zero-shot CoT

```
提示：
  一家店有 23 顆蘋果。如果他們用 20 顆做果汁，又買了 6 顆，
  請問店裡現在有幾顆蘋果？

  讓我們一步一步思考。（Let's think step by step.）

LLM 回應：
  1. 一開始有 23 顆蘋果
  2. 用掉 20 顆做果汁：23 - 20 = 3 顆
  3. 又買了 6 顆：3 + 6 = 9 顆
  答案：店裡現在有 9 顆蘋果。
```

#### Few-shot CoT

```
提示：
  問：Roger 有 5 顆網球。他又買了 2 罐網球。每罐有 3 顆。
  他現在有幾顆網球？
  答：Roger 一開始有 5 顆球。2 罐 × 3 顆/罐 = 6 顆。
  5 + 6 = 11。答案是 11 顆。

  問：食堂有 23 顆蘋果。用了 20 顆，又買了 6 顆。
  現在有幾顆？
  答：
```

**為什麼 CoT 有效？**
- 將複雜問題分解為小步驟
- 減少推理跳躍 (Reasoning Shortcuts) 導致的錯誤
- 讓 LLM 的「思考過程」可檢驗

### 6.6 系統提示設計 System Prompt Design

系統提示 (System Prompt) 定義了 LLM 的角色、行為邊界和輸出規範。良好的系統提示是穩定、可靠的 LLM 應用的基礎。

#### 系統提示的核心元素

```
1. 角色定義 (Role Definition)
   "你是一位資深的 Python 軟體工程師，專長是資料科學和機器學習。"

2. 行為準則 (Behavioral Guidelines)
   "回答時保持客觀、準確，不確定時明確說明。"

3. 知識範圍 (Knowledge Scope)
   "只回答與 Python 程式設計和資料科學相關的問題。"

4. 輸出格式 (Output Format)
   "回答以 Markdown 格式呈現，程式碼以 Python code block 包裹。"

5. 限制與安全 (Constraints & Safety)
   "不要生成有害或不當的內容。如果問題超出你的專業範圍，請引導使用者尋求專業協助。"
```

#### 系統提示範例

```
你是一位友善的 ML/DL 課程助教。你的職責是：

1. 用清楚易懂的方式解釋機器學習和深度學習概念
2. 提供 Python 程式碼範例來說明概念
3. 當學生的理解有誤時，溫和地糾正並給予正確觀念
4. 鼓勵學生思考，不直接給出作業答案——改用引導式提問
5. 回答使用繁體中文，專有名詞附上英文

回答格式要求：
- 先給出簡潔的摘要（2-3 句）
- 再提供詳細解釋
- 如適用，附上程式碼範例
- 最後提出一個思考問題，促進學生反思
```

### 6.7 其他進階提示技巧 Other Advanced Techniques

| 技巧 | 說明 | 範例 |
|------|------|------|
| 角色扮演 (Role Play) | 賦予 LLM 特定角色以引導風格 | "你是一位有 20 年經驗的資料科學家" |
| 結構化輸出 (Structured Output) | 要求特定格式（JSON, XML, CSV） | "以 JSON 格式輸出，包含 name, score, reason 欄位" |
| 自我一致性 (Self-Consistency) | 多次生成取多數決 (Majority Voting) | 生成 5 次答案，選最常出現的 |
| 後設提示 (Meta-Prompting) | 讓 LLM 先生成最佳提示 | "請設計一個提示，用於..." |
| 分隔符 (Delimiters) | 用清楚的分隔符區分不同部分 | `###`, `---`, `<context>...</context>` |
| 否定指令 (Negative Instructions) | 明確告訴 LLM 不要做什麼 | "不要編造資料。如果不確定，請說明。" |

---

## 7. LLM API 使用 Using LLM APIs

### 7.1 API 的基本概念 Basic API Concepts

LLM API 允許開發者透過 HTTP 請求與 LLM 互動。主要概念包括：

| 概念 | 說明 |
|------|------|
| API Key | 身份驗證憑證，不可洩露 |
| Endpoint | API 服務的 URL |
| Model | 指定使用的模型版本 |
| Messages | 對話歷史（包含 system, user, assistant 角色） |
| Temperature | 控制輸出的隨機性（0=確定性, 1=多樣性） |
| Max Tokens | 限制輸出的最大 Token 數 |
| Top-p (Nucleus Sampling) | 控制取樣範圍的另一種方式 |

### 7.2 Anthropic Claude API

```python
# 安裝: pip install anthropic
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="你是一位友善的 ML/DL 課程助教。",
    messages=[
        {"role": "user", "content": "請解釋什麼是梯度下降法？"}
    ]
)

print(message.content[0].text)
```

### 7.3 OpenAI GPT API

```python
# 安裝: pip install openai
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一位友善的 ML/DL 課程助教。"},
        {"role": "user", "content": "請解釋什麼是梯度下降法？"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### 7.4 API 安全最佳實踐 API Security Best Practices

```python
# 正確做法：使用環境變數
import os
api_key = os.environ.get("ANTHROPIC_API_KEY")

# 錯誤做法：硬編碼 API Key（絕對不要這樣做！）
# api_key = "sk-ant-xxx..."  # 危險！

# 使用 .env 檔案管理金鑰
# 1. 建立 .env 檔案（加入 .gitignore）
# 2. pip install python-dotenv
# 3. 在程式中載入
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("ANTHROPIC_API_KEY")
```

### 7.5 Temperature 與生成參數 Temperature & Generation Parameters

```
Temperature (溫度) 的影響：

temperature = 0.0  → 最確定、最一致的輸出（適合事實性問答）
temperature = 0.3  → 低度創意，較穩定（適合程式碼生成）
temperature = 0.7  → 適度創意（適合一般對話）
temperature = 1.0  → 高度創意、多樣化（適合創意寫作）

注意：temperature 越高，輸出越不可預測，也越容易產生幻覺。
```

---

## 8. LLM 的限制 Limitations of LLMs

### 8.1 幻覺 Hallucination

**定義**：LLM 生成看似合理但實際上不正確或無依據的內容。

**幻覺的類型**：

| 類型 | 說明 | 範例 |
|------|------|------|
| 事實性幻覺 Factual | 生成錯誤的事實 | 虛構的論文引用、錯誤的歷史日期 |
| 忠實性幻覺 Faithfulness | 回答偏離提供的參考資料 | 在 RAG 中忽略檢索到的資料 |
| 一致性幻覺 Consistency | 同一對話中自相矛盾 | 前後邏輯不一致 |

**為什麼會產生幻覺？**

1. **訓練資料的雜訊**：訓練資料中本身可能包含錯誤資訊
2. **統計預測的本質**：LLM 本質上是在預測「最可能的下一個 Token」，而非「最正確的答案」
3. **過度自信 (Overconfidence)**：LLM 通常不會表達不確定性
4. **上下文不足**：當問題超出模型的知識範圍時，模型傾向「補腦」

**緩解策略 Mitigation Strategies**：

| 策略 | 說明 |
|------|------|
| RAG（檢索增強） | 提供外部知識作為依據 |
| 降低 Temperature | 減少隨機性 |
| 要求引用來源 | 在提示中要求模型引用參考資料 |
| 事實查核 (Fact-checking) | 使用額外工具或人工驗證 |
| 多輪生成投票 | 生成多次答案，檢查一致性 |

### 8.2 知識截斷 Knowledge Cutoff

LLM 的知識受限於訓練資料的時間範圍。超過截止日期的事件，模型無法得知。

```
解決方案：
1. RAG — 從即時更新的知識庫檢索最新資訊
2. 工具使用 (Tool Use) — 讓 LLM 呼叫搜尋引擎、API 等外部工具
3. 微調 (Fine-tuning) — 用最新資料重新訓練（成本高）
```

### 8.3 推理能力的限制 Reasoning Limitations

- **數學計算**：LLM 對複雜算術可能出錯（尤其是大數運算）
- **邏輯推理**：在多步邏輯鏈中可能出錯
- **因果推理**：傾向於相關性而非因果性 (Correlation vs. Causation)
- **空間推理**：對三維空間、地理方位等推理能力較弱

### 8.4 偏見與公平性 Bias & Fairness

LLM 的訓練資料可能包含社會偏見 (Social Bias)，導致模型輸出反映或放大這些偏見。

---

## 9. 負責任使用 LLM Responsible Use of LLMs

### 9.1 倫理原則 Ethical Principles

| 原則 | 說明 |
|------|------|
| 透明性 (Transparency) | 向使用者告知內容是 AI 生成的 |
| 可驗證性 (Verifiability) | 提供資訊來源，允許使用者驗證 |
| 隱私保護 (Privacy) | 不將敏感/個人資料傳送給 LLM API |
| 公平性 (Fairness) | 注意並減輕輸出中的偏見 |
| 問責性 (Accountability) | 使用者/開發者對 LLM 的應用結果負責 |

### 9.2 學術誠信 Academic Integrity

在學術場景中使用 LLM 需要注意：

- **標註 AI 輔助**：明確說明哪些部分使用了 AI 工具
- **理解而非複製**：使用 LLM 幫助理解概念，而非直接抄襲輸出
- **批判性思維**：不盲目信任 LLM 的輸出，保持獨立思考
- **遵守規範**：遵循學校/課程的 AI 使用政策

### 9.3 安全考量 Security Considerations

| 風險 | 說明 | 對策 |
|------|------|------|
| Prompt Injection | 惡意輸入操縱 LLM 行為 | 輸入清洗、角色固定 |
| 資料外洩 | 敏感資料被傳送到外部 API | 資料脫敏、本地部署 |
| API Key 洩露 | 金鑰被寫入程式碼或上傳 | 環境變數、.gitignore |
| 有害內容生成 | LLM 被誘導生成不當內容 | 安全護欄 (Guardrails)、輸出過濾 |

---

## 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 大型語言模型 | Large Language Model (LLM) | 基於 Transformer 的大規模語言生成模型 |
| 自迴歸 | Autoregressive | 逐步生成，每步以前面的輸出為輸入 |
| 因果注意力 | Causal Attention | 每個位置只能關注自身及之前的位置 |
| Token | Token | 文本的最小處理單位（子詞或字元組合） |
| 分詞 | Tokenization | 將文本切分為 Token 序列 |
| 上下文窗口 | Context Window | LLM 一次能處理的最大 Token 數 |
| 文字嵌入 | Text Embedding | 將文字映射為固定長度的稠密向量 |
| 餘弦相似度 | Cosine Similarity | 衡量兩個向量方向相似程度的指標 |
| 語義搜尋 | Semantic Search | 基於語義相似度（而非關鍵字匹配）的搜尋 |
| 近似最近鄰搜尋 | Approximate Nearest Neighbor (ANN) | 高效的向量相似度搜尋演算法 |
| 向量資料庫 | Vector Database | 專門儲存和檢索高維向量的資料庫 |
| 檢索增強生成 | Retrieval-Augmented Generation (RAG) | 結合檢索與 LLM 生成的架構 |
| 文件切割 | Chunking | 將長文件切分為適合嵌入的小段落 |
| 提示工程 | Prompt Engineering | 設計有效提示以引導 LLM 輸出的技術 |
| 零樣本 | Zero-shot | 不提供範例直接執行任務 |
| 少樣本 | Few-shot | 提供少量範例讓模型學習任務模式 |
| 思維鏈 | Chain-of-Thought (CoT) | 引導模型展示逐步推理過程 |
| 系統提示 | System Prompt | 定義 LLM 的角色和行為準則 |
| 幻覺 | Hallucination | LLM 生成看似合理但實際錯誤的內容 |
| 知識截斷 | Knowledge Cutoff | 模型知識受限於訓練資料的時間範圍 |
| 溫度 | Temperature | 控制 LLM 輸出隨機性的超參數 |
| 上下文學習 | In-Context Learning (ICL) | LLM 從提示中的範例學習新任務的能力 |
| 對比學習 | Contrastive Learning | 透過正負樣本對訓練嵌入模型 |
| 微調 | Fine-tuning | 在特定資料上進一步訓練預訓練模型 |
| 提示注入 | Prompt Injection | 透過惡意輸入操縱 LLM 行為的攻擊 |

---

## 延伸閱讀 Further Reading

- Vaswani et al. (2017). "Attention Is All You Need" — Transformer 原始論文
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — RAG 原始論文
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" — CoT 論文
- Brown et al. (2020). "Language Models are Few-Shot Learners" — GPT-3 / In-Context Learning
- Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- Anthropic Claude 官方文件：https://docs.anthropic.com/
- OpenAI API 官方文件：https://platform.openai.com/docs/
- LangChain 官方文件：https://python.langchain.com/ — RAG 應用框架
- Chroma 官方文件：https://docs.trychroma.com/ — 輕量向量資料庫
- Hugging Face Sentence Transformers：https://www.sbert.net/ — 開源嵌入模型
