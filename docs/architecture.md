# 系統架構圖 / System Architecture

## 整體架構

```
┌─────────────────────────────────────────────────────────┐
│                    使用者 (瀏覽器)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ 首頁     │  │ 週次頁面  │  │ 儀表板   │  │ 管理面板 │ │
│  │ 18週導覽 │  │ 視覺化+  │  │ 學習分析 │  │ 使用者   │ │
│  │          │  │ Quiz+Chat │  │          │  │ LLM設定  │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                 Nginx (Production)                       │
│            靜態檔案 + API 反向代理                         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend                         │
│                                                         │
│  ┌─── API Layer ────────────────────────────────────┐   │
│  │ auth   admin   llm   models   analytics   quiz   │   │
│  │ rag                                              │   │
│  └──┬────────┬──────────┬───────────────────────────┘   │
│     │        │          │                               │
│  ┌──▼──┐  ┌──▼───┐  ┌──▼──────────────────────────┐   │
│  │ Auth │  │ LLM  │  │ NLP Pipeline (7 layers)     │   │
│  │ JWT  │  │      │  │ Intent → Emotion → Difficulty│   │
│  │ RBAC │  │ Tutor│  │ → Topic → Context → Rerank  │   │
│  └──────┘  │      │  │ → Response                   │   │
│            └──┬───┘  └──────────────────────────────┘   │
│               │                                         │
│  ┌────────────▼─────────────────────────────────────┐   │
│  │           LLM Abstraction Layer                  │   │
│  │  ┌─────────┐ ┌────────┐ ┌──────┐ ┌───────────┐  │   │
│  │  │ Claude  │ │  GPT   │ │Ollama│ │ Local NLP │  │   │
│  │  │ API     │ │  API   │ │      │ │ (Offline) │  │   │
│  │  └─────────┘ └────────┘ └──────┘ └───────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Data Layer                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │   │
│  │  │ SQLite   │  │ RAG/FTS5 │  │ NLP Models    │  │   │
│  │  │ users    │  │ rag_fts  │  │ intent.pkl    │  │   │
│  │  │ events   │  │ rag_chunk│  │ emotion.pkl   │  │   │
│  │  │ settings │  │          │  │ corpus_tfidf  │  │   │
│  │  └──────────┘  └──────────┘  └───────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

## 資料庫 Schema

┌──────────────────┐     ┌─────────────────────┐
│     users         │     │  teacher_students    │
├──────────────────┤     ├─────────────────────┤
│ id (PK)          │◄────│ teacher_id (FK)      │
│ username (UNIQUE)│     │ student_id (FK)      │
│ password_hash    │◄────│                      │
│ display_name     │     └─────────────────────┘
│ email            │
│ role             │     ┌─────────────────────┐
│ is_active        │     │  learning_events     │
│ created_at       │     ├─────────────────────┤
│ updated_at       │     │ id (PK)             │
└──────────────────┘     │ student_id          │
                         │ week                │
┌──────────────────┐     │ event_type          │
│ system_settings   │     │ topic               │
├──────────────────┤     │ score               │
│ key (PK)         │     │ duration_seconds    │
│ value            │     │ metadata (JSON)     │
│ updated_at       │     │ timestamp           │
└──────────────────┘     └─────────────────────┘

┌──────────────────┐     ┌─────────────────────┐
│   rag_chunks      │     │   rag_fts (FTS5)    │
├──────────────────┤     ├─────────────────────┤
│ id (PK)          │────▶│ id                  │
│ content          │     │ content             │
│ week             │     │ title               │
│ file_type        │     └─────────────────────┘
│ title            │
│ source           │
└──────────────────┘

## API 端點總覽

Auth (3)        POST /api/auth/login, GET /me, POST /register
Admin (10)      CRUD users, teacher-student, settings, train-nlp
LLM (3)         POST /chat, GET /model-info, WS /ws/chat
Models (6)      gradient-descent, loss-landscape, decision-boundary,
                roc-pr, tree, activations
Analytics (3)   POST /events, GET /students/{id}, GET /summary
Quiz (2)        GET /week/{w}, POST /submit
RAG (2)         POST /search, POST /ingest
Health (1)      GET /health
                                              Total: 30 endpoints

## 前端元件對應

W1  EnvironmentSetupViz     W10 LearningCurveViz
W2  EDAViz                  W11 ActivationFunctionViz
W3  DataSplitViz            W12 CNNLayerViz
W4  GradientDescentViz      W13 AttentionViz
W5  DecisionBoundaryViz     W14 TrainingComparisonViz
W6  DecisionBoundaryViz     W15 FairnessViz
W7  TreeGrowthViz           W16 MLOpsFlowViz
W8  FeatureImportanceViz    W17 EmbeddingSpaceViz
W9  PipelineFlowViz         W18 ProjectShowcaseViz
```

## 部署架構 (Docker)

```
docker-compose.yml
  ├── frontend (Nginx:80)
  │     ├── 靜態 React build
  │     └── 反向代理 → backend:8000
  ├── backend (Uvicorn:8000)
  │     ├── FastAPI app
  │     └── volume: backend-data (SQLite + NLP models)
  └── volume: backend-data
```
