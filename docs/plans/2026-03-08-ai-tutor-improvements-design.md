# AI 助教系統改善設計

> 日期：2026-03-08
> 方案：B（全面平行）

## 改善範圍

三個平行工作流：

1. ChatPanel Markdown + 程式碼高亮 + LaTeX 渲染
2. 前端測試覆蓋率提升
3. 30+ Tiny NLP 微層調度架構 + 語料擴充

---

## Section 1：ChatPanel Markdown + 程式碼高亮 + LaTeX

**修改檔案：** `platform/frontend/src/components/llm/ChatPanel.tsx`

**新增依賴：** `react-markdown`, `remark-math`, `rehype-katex`, `rehype-highlight`, `katex`

**改動內容：**

- AI 回覆使用 `<ReactMarkdown>` 渲染（使用者訊息維持純文字）
- remark-math 解析 `$...$` 和 `$$...$$`
- rehype-katex 渲染數學公式
- rehype-highlight 程式碼語法高亮
- 引入 katex CSS 和 highlight.js 主題 CSS

---

## Section 2：前端測試覆蓋率提升

**新增測試檔案：**

| 檔案 | 測試重點 |
|------|---------|
| `ChatPanel.test.tsx` | 模式切換、訊息送出、清除、loading 狀態、Markdown 渲染 |
| `QuizPanel.test.tsx` | 載入題目、選擇答案、提交批改、重新作答 |
| `WeekPage.test.tsx` | 正確週次渲染、無效週次、lazy load |
| `Dashboard.test.tsx` | 班級總覽載入、學生查詢、圖表渲染 |
| `Home.test.tsx` | 18 週課程卡片列表渲染 |
| `Sidebar.test.tsx` | 導航連結、當前週次高亮 |

**策略：** vitest + @testing-library/react，mock fetchAPI 和 useAuth，~25-30 個測試案例。

---

## Section 3：30+ Tiny NLP 微層調度架構

### 新增依賴

| 套件 | 用途 | 大小 |
|------|------|------|
| jieba | 中文分詞、關鍵詞提取、POS 標註 | ~58MB |
| snownlp | 中文情感分析、摘要、關鍵詞 | ~40MB |
| langdetect | 語言偵測 | ~2MB |
| nltk (punkt, averaged_perceptron_tagger, stopwords) | 英文分詞、POS、停用詞 | ~15MB |
| textstat | 文本可讀性/複雜度評分 | <1MB |
| sentence-transformers (paraphrase-multilingual-MiniLM-L6-v2) | 語意相似度、嵌入 | ~90MB |
| rapidfuzz | 模糊字串比對 | <1MB |

### 42 層微 NLP 架構

#### A. 文本前處理層 (Layer 1-6)

| # | 微層 | NLP 套件 |
|---|------|---------|
| 1 | ChineseSegmenter | jieba |
| 2 | POSTagger | jieba.posseg |
| 3 | SentenceSplitter | nltk.punkt |
| 4 | LanguageDetector | langdetect |
| 5 | TextNormalizer | 自建 |
| 6 | StopwordFilter | nltk + 自建中文 |

#### B. 學生理解層 (Layer 7-14)

| # | 微層 | NLP 套件 |
|---|------|---------|
| 7 | IntentClassifier | sklearn (ML) + regex |
| 8 | SubIntentDetector | sklearn |
| 9 | EmotionClassifier | sklearn (ML) + regex |
| 10 | SentimentScorer | snownlp |
| 11 | FrustrationEscalator | 自建規則 |
| 12 | ConfidenceEstimator | snownlp + regex |
| 13 | UrgencyDetector | regex + NER |
| 14 | PolitenessDetector | snownlp |

#### C. 學生程度評估層 (Layer 15-20)

| # | 微層 | NLP 套件 |
|---|------|---------|
| 15 | DifficultyAssessor | textstat + jieba |
| 16 | VocabularyLevelScorer | jieba + 術語庫 |
| 17 | TechnicalFluencyScorer | jieba.posseg |
| 18 | LearningStyleDetector | 規則引擎 |
| 19 | MisconceptionDetector | rapidfuzz + 概念庫 |
| 20 | KnowledgeGapDetector | sentence-transformers |

#### D. 內容分析層 (Layer 21-27)

| # | 微層 | NLP 套件 |
|---|------|---------|
| 21 | KeywordExtractor | jieba.analyse (TF-IDF/TextRank) |
| 22 | DomainConceptMatcher | rapidfuzz + 概念庫 |
| 23 | NamedEntityRecognizer | jieba + 自建詞典 |
| 24 | CodeBlockDetector | regex + AST |
| 25 | MathExpressionDetector | regex |
| 26 | QuestionQualityScorer | 多特徵 |
| 27 | ReadabilityScorer | textstat |

#### E. 上下文與記憶層 (Layer 28-32)

| # | 微層 | NLP 套件 |
|---|------|---------|
| 28 | ConversationTracker | 自建 |
| 29 | TopicContinuityDetector | sentence-transformers |
| 30 | HintLadderManager | 狀態機 |
| 31 | SessionSummarizer | snownlp + jieba |
| 32 | KnowledgeStateTracker | 自建 |

#### F. 檢索增強層 (Layer 33-36)

| # | 微層 | NLP 套件 |
|---|------|---------|
| 33 | QueryExpander | jieba + sentence-transformers |
| 34 | RAGRetriever | SQLite FTS5 |
| 35 | SemanticReranker | sentence-transformers |
| 36 | CrossWeekLinker | sentence-transformers |

#### G. 回應生成層 (Layer 37-42)

| # | 微層 | NLP 套件 |
|---|------|---------|
| 37 | ResponseAssembler | 模板引擎 |
| 38 | ComplexityAdjuster | textstat |
| 39 | CitationInjector | 自建 |
| 40 | FollowUpGenerator | 規則 + 模板 |
| 41 | EncouragementGenerator | 情緒感知模板 |
| 42 | ResponseCompletenessChecker | 規則 |

### 調度架構

```
UserMessage → [前處理 1-6] → [理解 7-14] → [程度 15-20] → [內容 21-27]
            → [上下文 28-32] → [檢索 33-36] → [回應 37-42] → FinalResponse
```

每層接收 NLPContext 物件，寫入欄位，傳給下一層。

### 語料擴充

- 18 種 intent 各新增 15-20 條中英文訓練語句
- 5 種 emotion 各新增 15-20 條
- 模擬真實學生提問風格
- 訓練後覆寫 data/nlp_models/*.pkl
