# NLP Pipeline 升級設計：Regex → 30+ 真實 NLP

> 日期：2026-03-08
> 目標：將 42 層 pipeline 中的真實 NLP 層數從 13 → 30+
> 約束：本地 Windows 11, Python 3.14, 不需 GPU

## 現況分析

| 類別 | 層數 | 比例 |
|------|------|------|
| 真實 NLP | 13 | 31% |
| Regex/規則 | 23 | 55% |
| 模板 | 6 | 14% |

## 升級策略

### 原則
1. **用已安裝的套件**：scikit-learn, jieba, snownlp, faiss 都已就緒
2. **訓練資料生成**：在 trainer.py 加入新分類器的訓練資料
3. **向後相容**：每層保留 regex fallback，ML 模型優先
4. **不裝大型套件**：跳過 spaCy (Python 3.14 不相容), 跳過 sentence-transformers (太大)
5. **新增 wordfreq**：唯一需要新裝的套件 (~5MB)

## 升級清單

### Phase 1：scikit-learn ML 分類器升級 (7 層)

將 regex 層升級為 TF-IDF + ML 分類器，與 intent/emotion 同架構。

| 層 | 函數 | 升級內容 | 模型 |
|----|------|---------|------|
| L8 | sub_intent_detector | 子意圖 ML 分類 | TF-IDF + LinearSVC |
| L12 | confidence_estimator | 信心度 ML 評估 | TF-IDF + LogisticRegression |
| L13 | urgency_detector | 緊急度 ML 分類 | TF-IDF + LinearSVC |
| L14 | politeness_detector | 禮貌度 ML 評分 | TF-IDF + LogisticRegression |
| L18 | learning_style_detector | 學習風格 ML 分類 | TF-IDF + LinearSVC |
| L20 | knowledge_gap_detector | 知識缺口 ML 偵測 | TF-IDF + LogisticRegression |
| L26 | question_quality_scorer | 問題品質 ML 評分 | TF-IDF + LogisticRegression |

### Phase 2：jieba 深度分析升級 (4 層)

利用 jieba 的進階功能：analyse, posseg, userdict。

| 層 | 函數 | 升級內容 |
|----|------|---------|
| L16 | vocabulary_level_scorer | jieba.analyse.tfidf 詞彙權重 + wordfreq 詞頻 |
| L17 | technical_fluency_scorer | jieba.posseg 詞性分布統計 → ML 特徵 |
| L23 | named_entity_recognizer | jieba 自訂詞典 (ML/DL 術語 200+) + POS-based NER |
| L33 | query_expander | jieba.analyse 關鍵詞 + 同義詞 embedding 擴展 |

### Phase 3：FAISS 向量檢索升級 (3 層)

faiss 已安裝，可以做本地向量搜尋。

| 層 | 函數 | 升級內容 |
|----|------|---------|
| L34 | rag_retriever | FTS5 + FAISS 雙通道檢索 (TF-IDF 向量) |
| L35 | semantic_reranker | FAISS cosine similarity 重排 |
| L36 | cross_week_linker | FAISS 跨週語意相似度搜尋 |

### Phase 4：snownlp + 多特徵升級 (3 層)

| 層 | 函數 | 升級內容 |
|----|------|---------|
| L11 | frustration_escalator | snownlp 多輪情感趨勢分析 |
| L15 | difficulty_assessor | 多特徵 ML (詞彙+長度+術語+POS) |
| L32 | knowledge_state_tracker | snownlp + jieba 概念理解度追蹤 |

### Phase 5：新增協調層 (3 層)

| 新層 | 函數 | 功能 | 套件 |
|------|------|------|------|
| C1 | aggregate_confidence | 聚合多層 confidence 加權投票 | numpy |
| C2 | resolve_conflicts | 解決 intent/emotion 矛盾 | scikit-learn |
| C3 | route_pipeline | 根據訊息特徵自適應跳層 | numpy |

## 升級後數字

| 類別 | 升級前 | 升級後 |
|------|--------|--------|
| 真實 NLP | 13 | **33** |
| Regex/規則 | 23 | 6 |
| 模板 | 6 | 6 |
| 總層數 | 42 | **45** (+ 3 協調層) |
| 真實 NLP 比例 | 31% | **73%** |

## 新增依賴

| 套件 | 大小 | 用途 |
|------|------|------|
| wordfreq | ~5MB | 詞頻分析 (L16) |

其餘全部使用已安裝套件。

## 訓練資料擴展

trainer.py 需新增 5 個分類器的訓練資料生成器：
- `_generate_sub_intent_data()` — 每個主意圖下 3-5 個子意圖
- `_generate_confidence_data()` — 高/中/低信心度語句
- `_generate_urgency_data()` — 高/一般/低緊急度語句
- `_generate_politeness_data()` — 禮貌/中性/直接語句
- `_generate_learning_style_data()` — visual/practical/textual/balanced

## 測試計畫

1. 每個升級層寫 pytest 單元測試
2. 全 pipeline 整合測試 (4 場景)
3. 所有 38 現有後端測試必須通過
4. 前端 37 測試必須通過
5. 本地 commit 後再推雲端
