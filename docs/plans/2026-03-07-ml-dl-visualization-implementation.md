# ML/DL 視覺化工具教材系統 - 實作計劃

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 建置完整的 18 週 ML/DL 視覺化互動教學系統，含課程教材、互動平台、LLM 助教與學習分析。

**Architecture:** 模組化分層架構 — 前端 React+TypeScript 提供視覺化互動元件，後端 FastAPI 提供 ML 模型訓練/推論與 LLM 抽象層，教材以 Markdown + Jupyter Notebook 獨立管理。

**Tech Stack:** React 18, TypeScript, Vite, Recharts/D3.js, FastAPI, Python 3.11+, scikit-learn, PyTorch, Anthropic SDK, OpenAI SDK, SQLite, WebSocket

---

## Phase 1: 專案基礎建設

### Task 1: 初始化專案目錄結構

**Files:**
- Create: `ml-dl-visualization/README.md`
- Create: `ml-dl-visualization/curriculum/syllabus.md`
- Create: `ml-dl-visualization/datasets/README.md`
- Create: 18 週的目錄結構 `curriculum/week-01/` ~ `curriculum/week-18/`

**Step 1: 建立完整目錄樹**

```bash
cd "D:/course/教材教具/ml-dl-visualization"
for i in $(seq -w 1 18); do
  mkdir -p "curriculum/week-$i"
done
mkdir -p datasets platform/frontend platform/backend docs/plans
```

**Step 2: 建立 README.md**

```markdown
# ML/DL 視覺化工具教學系統

18 週機器學習與深度學習視覺化互動教材，含互動平台、LLM 助教與學習分析。

## 架構
- `curriculum/` - 18 週課程教材（講義、Notebook、作業、評量）
- `platform/frontend/` - React 互動平台
- `platform/backend/` - FastAPI 後端
- `datasets/` - 範例資料集

## 快速開始
見各子目錄 README。
```

**Step 3: 建立課程大綱 syllabus.md**

完整 18 週大綱，含每週主題、學習目標、核心/進階標示。

---

### Task 2: 初始化 FastAPI 後端

**Files:**
- Create: `platform/backend/pyproject.toml`
- Create: `platform/backend/app/__init__.py`
- Create: `platform/backend/app/main.py`
- Create: `platform/backend/app/config.py`
- Create: `platform/backend/tests/__init__.py`

**Step 1: 建立 pyproject.toml**

```toml
[project]
name = "ml-dl-viz-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "scikit-learn>=1.6",
    "torch>=2.5",
    "numpy>=2.0",
    "pandas>=2.2",
    "matplotlib>=3.9",
    "anthropic>=0.42",
    "openai>=1.60",
    "websockets>=14.0",
    "pydantic>=2.10",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.28", "pytest-asyncio>=0.25"]
```

**Step 2: 建立 app/main.py — FastAPI 應用入口**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ML/DL 視覺化教學平台", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Step 3: 建立 app/config.py**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"  # anthropic | openai | ollama
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "claude-sonnet-4-20250514"

    class Config:
        env_file = ".env"

settings = Settings()
```

**Step 4: 寫測試驗證伺服器啟動**

```python
# tests/test_health.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

**Step 5: 執行測試**

```bash
cd platform/backend
pip install -e ".[dev]"
pytest tests/test_health.py -v
```

---

### Task 3: 初始化 React 前端

**Files:**
- Create: `platform/frontend/` (via Vite scaffold)
- Modify: `platform/frontend/package.json`
- Create: `platform/frontend/src/App.tsx`

**Step 1: 使用 Vite 建立 React+TS 專案**

```bash
cd "D:/course/教材教具/ml-dl-visualization/platform"
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

**Step 2: 安裝核心依賴**

```bash
npm install react-router-dom recharts d3 @types/d3 lucide-react
npm install -D tailwindcss @tailwindcss/vite
```

**Step 3: 設定 Tailwind CSS**

vite.config.ts 加入 Tailwind plugin，建立 `src/index.css` 引入 `@import "tailwindcss";`

**Step 4: 建立基礎路由結構**

```tsx
// src/App.tsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```

**Step 5: 驗證前端啟動**

```bash
npm run dev
# 預期：http://localhost:5173 正常顯示
```

---

## Phase 2: LLM 助教抽象層

### Task 4: 建立多模型 LLM 抽象層

**Files:**
- Create: `platform/backend/app/llm/__init__.py`
- Create: `platform/backend/app/llm/base.py`
- Create: `platform/backend/app/llm/anthropic_provider.py`
- Create: `platform/backend/app/llm/openai_provider.py`
- Create: `platform/backend/app/llm/ollama_provider.py`
- Create: `platform/backend/app/llm/factory.py`
- Create: `platform/backend/tests/test_llm.py`

**Step 1: 定義 LLM 基礎介面**

```python
# app/llm/base.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

class LLMMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict | None = None

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        ...

    @abstractmethod
    async def stream(self, messages: list[LLMMessage], system: str = ""):
        ...
```

**Step 2: 實作 Anthropic Provider**

```python
# app/llm/anthropic_provider.py
import anthropic
from .base import LLMProvider, LLMMessage, LLMResponse

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return LLMResponse(
            content=resp.content[0].text,
            model=resp.model,
            usage={"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
        )

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as stream:
            async for text in stream.text_stream:
                yield text
```

**Step 3: 實作 OpenAI Provider**

```python
# app/llm/openai_provider.py
from openai import AsyncOpenAI
from .base import LLMProvider, LLMMessage, LLMResponse

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        resp = await self.client.chat.completions.create(model=self.model, messages=msgs)
        choice = resp.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=resp.model,
            usage={"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens},
        )

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        stream = await self.client.chat.completions.create(
            model=self.model, messages=msgs, stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

**Step 4: 實作 Ollama Provider**

```python
# app/llm/ollama_provider.py
import httpx
from .base import LLMProvider, LLMMessage, LLMResponse

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": False},
                timeout=120,
            )
            data = resp.json()
        return LLMResponse(content=data["message"]["content"], model=self.model)

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": True},
                timeout=120,
            ) as resp:
                import json
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and data["message"].get("content"):
                            yield data["message"]["content"]
```

**Step 5: 建立 Factory**

```python
# app/llm/factory.py
from app.config import settings
from .base import LLMProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

def create_llm_provider() -> LLMProvider:
    match settings.llm_provider:
        case "anthropic":
            return AnthropicProvider(api_key=settings.anthropic_api_key, model=settings.model_name)
        case "openai":
            return OpenAIProvider(api_key=settings.openai_api_key, model=settings.model_name)
        case "ollama":
            return OllamaProvider(base_url=settings.ollama_base_url, model=settings.model_name)
        case _:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
```

**Step 6: 寫單元測試**

```python
# tests/test_llm.py
from app.llm.base import LLMMessage, LLMResponse
from app.llm.factory import create_llm_provider

def test_llm_message_model():
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

def test_llm_response_model():
    resp = LLMResponse(content="Hi", model="test")
    assert resp.content == "Hi"
    assert resp.usage is None
```

---

### Task 5: 建立 LLM 助教分層提示系統 (Hint Ladder)

**Files:**
- Create: `platform/backend/app/llm/tutor.py`
- Create: `platform/backend/app/llm/prompts.py`
- Create: `platform/backend/tests/test_tutor.py`

**Step 1: 定義提示模板**

```python
# app/llm/prompts.py
SYSTEM_TUTOR = """你是一位 ML/DL 視覺化課程的 AI 助教。你的任務是引導學生學習，而非直接給答案。

## 分層提示策略 (Hint Ladder)
依照以下層級逐步引導，每次回覆只進一層：

Level 1 - 釐清問題：詢問學生想解決什麼、已嘗試什麼
Level 2 - 概念提示：提供相關概念或原理的提示
Level 3 - 步驟引導：給出解題方向與步驟
Level 4 - 局部範例：僅在學生多次嘗試後，給出部分程式碼範例

## 規則
- 永遠不要直接給出完整作業答案
- 要求學生先描述錯誤訊息、輸入/輸出、已嘗試的方法
- 依程度調整：初學者用類比與概念檢核；進階者用推導檢查與邊界條件
- 回覆時使用中文，專有名詞附英文
- 鼓勵學生自主思考，建立學習信心

## 當前週次：{week}
## 當前主題：{topic}
"""

SYSTEM_HOMEWORK = """你是一位 ML/DL 課程的 AI 助教，目前學生在進行作業。

## 作業模式規則
- 強制要求學生先提交思路或中間結果
- 不給完整答案，只提供引導
- 如果學生直接要答案，引導他們回到自主思考
- 每次回覆結尾附一個引導性問題

## 當前作業：{assignment}
"""
```

**Step 2: 建立 Tutor 類別**

```python
# app/llm/tutor.py
from .base import LLMProvider, LLMMessage
from .prompts import SYSTEM_TUTOR, SYSTEM_HOMEWORK

class AITutor:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def ask(
        self, messages: list[LLMMessage], week: int, topic: str, mode: str = "tutor"
    ):
        if mode == "homework":
            system = SYSTEM_HOMEWORK.format(assignment=topic)
        else:
            system = SYSTEM_TUTOR.format(week=week, topic=topic)
        return await self.provider.chat(messages, system=system)

    async def ask_stream(
        self, messages: list[LLMMessage], week: int, topic: str, mode: str = "tutor"
    ):
        if mode == "homework":
            system = SYSTEM_HOMEWORK.format(assignment=topic)
        else:
            system = SYSTEM_TUTOR.format(week=week, topic=topic)
        async for chunk in self.provider.stream(messages, system=system):
            yield chunk
```

---

### Task 6: 建立 LLM API 端點

**Files:**
- Create: `platform/backend/app/api/__init__.py`
- Create: `platform/backend/app/api/llm_routes.py`
- Modify: `platform/backend/app/main.py`

**Step 1: 建立聊天 API**

```python
# app/api/llm_routes.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from app.llm.factory import create_llm_provider
from app.llm.tutor import AITutor
from app.llm.base import LLMMessage

router = APIRouter(prefix="/api/llm", tags=["LLM"])

class ChatRequest(BaseModel):
    messages: list[LLMMessage]
    week: int = 1
    topic: str = ""
    mode: str = "tutor"  # tutor | homework

@router.post("/chat")
async def chat(req: ChatRequest):
    provider = create_llm_provider()
    tutor = AITutor(provider)
    response = await tutor.ask(req.messages, week=req.week, topic=req.topic, mode=req.mode)
    return {"response": response.content, "model": response.model}

@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    provider = create_llm_provider()
    tutor = AITutor(provider)
    try:
        while True:
            data = await websocket.receive_json()
            messages = [LLMMessage(**m) for m in data.get("messages", [])]
            week = data.get("week", 1)
            topic = data.get("topic", "")
            mode = data.get("mode", "tutor")
            async for chunk in tutor.ask_stream(messages, week=week, topic=topic, mode=mode):
                await websocket.send_json({"type": "chunk", "content": chunk})
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
```

**Step 2: 註冊路由至 main.py**

在 `app/main.py` 加入：
```python
from app.api.llm_routes import router as llm_router
app.include_router(llm_router)
```

---

## Phase 3: ML 模型服務 API

### Task 7: 建立 ML 模型訓練與視覺化 API

**Files:**
- Create: `platform/backend/app/models/__init__.py`
- Create: `platform/backend/app/models/linear.py`
- Create: `platform/backend/app/models/classification.py`
- Create: `platform/backend/app/models/tree.py`
- Create: `platform/backend/app/models/neural.py`
- Create: `platform/backend/app/api/model_routes.py`

**Step 1: 線性回歸模型服務（Week 4）**

```python
# app/models/linear.py
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from pydantic import BaseModel

class GradientDescentResult(BaseModel):
    weights_history: list[list[float]]
    loss_history: list[float]
    final_weights: list[float]
    final_loss: float

def run_gradient_descent(
    X: list[list[float]], y: list[float],
    learning_rate: float = 0.01, epochs: int = 100
) -> GradientDescentResult:
    X_arr = np.array(X)
    y_arr = np.array(y)
    n, d = X_arr.shape
    w = np.zeros(d)
    b = 0.0
    weights_history = []
    loss_history = []

    for _ in range(epochs):
        pred = X_arr @ w + b
        error = pred - y_arr
        loss = float(np.mean(error ** 2))
        loss_history.append(loss)
        weights_history.append(w.tolist() + [b])
        grad_w = (2 / n) * (X_arr.T @ error)
        grad_b = (2 / n) * np.sum(error)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return GradientDescentResult(
        weights_history=weights_history,
        loss_history=loss_history,
        final_weights=w.tolist() + [b],
        final_loss=loss_history[-1],
    )

def compute_loss_landscape(
    X: list[list[float]], y: list[float],
    w0_range: tuple = (-5, 5), w1_range: tuple = (-5, 5), resolution: int = 50
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)
    w0s = np.linspace(*w0_range, resolution)
    w1s = np.linspace(*w1_range, resolution)
    Z = np.zeros((resolution, resolution))
    for i, w0 in enumerate(w0s):
        for j, w1 in enumerate(w1s):
            pred = X_arr[:, 0] * w0 + w1
            Z[i, j] = float(np.mean((pred - y_arr) ** 2))
    return {"w0": w0s.tolist(), "w1": w1s.tolist(), "loss": Z.tolist()}
```

**Step 2: 分類模型服務（Week 5-6）**

```python
# app/models/classification.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler

def train_and_get_decision_boundary(
    X: list[list[float]], y: list[int],
    model_type: str = "logistic", C: float = 1.0, kernel: str = "rbf",
    resolution: int = 100
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    if model_type == "logistic":
        model = LogisticRegression(C=C)
    else:
        model = SVC(C=C, kernel=kernel, probability=True)
    model.fit(X_scaled, y_arr)

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    return {
        "xx": xx.tolist(), "yy": yy.tolist(), "Z": Z.tolist(),
        "X": X_scaled.tolist(), "y": y_arr.tolist(),
        "accuracy": float(model.score(X_scaled, y_arr)),
    }

def get_roc_pr_curves(
    X: list[list[float]], y: list[int], model_type: str = "logistic", C: float = 1.0
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)
    model = LogisticRegression(C=C) if model_type == "logistic" else SVC(C=C, probability=True)
    model.fit(X_arr, y_arr)
    proba = model.predict_proba(X_arr)[:, 1]
    fpr, tpr, _ = roc_curve(y_arr, proba)
    precision, recall, _ = precision_recall_curve(y_arr, proba)
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist()},
    }
```

**Step 3: 樹模型服務（Week 7-8）**

```python
# app/models/tree.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import json

def train_tree_model(
    X: list[list[float]], y: list[int],
    model_type: str = "decision_tree", max_depth: int = 5,
    n_estimators: int = 100, feature_names: list[str] | None = None
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)

    if model_type == "decision_tree":
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == "random_forest":
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    else:
        model = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)

    model.fit(X_arr, y_arr)
    result = {
        "accuracy": float(model.score(X_arr, y_arr)),
        "feature_importances": model.feature_importances_.tolist(),
    }
    if model_type == "decision_tree":
        result["tree_text"] = export_text(model, feature_names=feature_names)
    return result
```

**Step 4: 建立模型 API 路由**

```python
# app/api/model_routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.models.linear import run_gradient_descent, compute_loss_landscape
from app.models.classification import train_and_get_decision_boundary, get_roc_pr_curves
from app.models.tree import train_tree_model

router = APIRouter(prefix="/api/models", tags=["Models"])

class GradientDescentRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    learning_rate: float = 0.01
    epochs: int = 100

@router.post("/gradient-descent")
async def gradient_descent(req: GradientDescentRequest):
    return run_gradient_descent(req.X, req.y, req.learning_rate, req.epochs)

class LossLandscapeRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    w0_range: list[float] = [-5, 5]
    w1_range: list[float] = [-5, 5]
    resolution: int = 50

@router.post("/loss-landscape")
async def loss_landscape(req: LossLandscapeRequest):
    return compute_loss_landscape(
        req.X, req.y, tuple(req.w0_range), tuple(req.w1_range), req.resolution
    )

class DecisionBoundaryRequest(BaseModel):
    X: list[list[float]]
    y: list[int]
    model_type: str = "logistic"
    C: float = 1.0
    kernel: str = "rbf"

@router.post("/decision-boundary")
async def decision_boundary(req: DecisionBoundaryRequest):
    return train_and_get_decision_boundary(
        req.X, req.y, req.model_type, req.C, req.kernel
    )

class TreeModelRequest(BaseModel):
    X: list[list[float]]
    y: list[int]
    model_type: str = "decision_tree"
    max_depth: int = 5
    n_estimators: int = 100
    feature_names: list[str] | None = None

@router.post("/tree")
async def tree_model(req: TreeModelRequest):
    return train_tree_model(
        req.X, req.y, req.model_type, req.max_depth, req.n_estimators, req.feature_names
    )
```

**Step 5: 註冊路由**

在 `app/main.py` 加入：
```python
from app.api.model_routes import router as model_router
app.include_router(model_router)
```

---

## Phase 4: React 前端視覺化元件

### Task 8: 建立前端基礎架構與佈局

**Files:**
- Create: `platform/frontend/src/components/Layout.tsx`
- Create: `platform/frontend/src/components/Sidebar.tsx`
- Create: `platform/frontend/src/pages/Home.tsx`
- Create: `platform/frontend/src/lib/api.ts`
- Create: `platform/frontend/src/types/index.ts`

**Step 1: 建立 API 客戶端**

```typescript
// src/lib/api.ts
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function fetchAPI<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export function createWebSocket(path: string): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return new WebSocket(`${wsBase}${path}`);
}
```

**Step 2: 建立側邊欄（18 週導覽）**

```tsx
// src/components/Sidebar.tsx
import { NavLink } from "react-router-dom";

const weeks = [
  { id: 1, title: "課程導論、Python 環境" },
  { id: 2, title: "資料視覺化與 EDA" },
  { id: 3, title: "監督式學習概念" },
  { id: 4, title: "線性回歸與梯度下降" },
  { id: 5, title: "分類與決策邊界" },
  { id: 6, title: "SVM 與核方法" },
  { id: 7, title: "樹模型與集成" },
  { id: 8, title: "特徵重要度與 SHAP" },
  { id: 9, title: "特徵工程與前處理" },
  { id: 10, title: "超參數調校" },
  { id: 11, title: "神經網路基礎" },
  { id: 12, title: "CNN 視覺化" },
  { id: 13, title: "RNN/Transformers" },
  { id: 14, title: "DL 訓練技巧" },
  { id: 15, title: "模型評估與公平性" },
  { id: 16, title: "MLOps 入門" },
  { id: 17, title: "LLM 與嵌入應用" },
  { id: 18, title: "綜合專題展示" },
];

export default function Sidebar() {
  return (
    <nav className="w-64 bg-gray-50 border-r h-screen overflow-y-auto p-4">
      <h2 className="text-lg font-bold mb-4">ML/DL 視覺化</h2>
      <ul className="space-y-1">
        {weeks.map((w) => (
          <li key={w.id}>
            <NavLink
              to={`/week/${w.id}`}
              className={({ isActive }) =>
                `block px-3 py-2 rounded text-sm ${isActive ? "bg-blue-100 text-blue-800 font-medium" : "text-gray-700 hover:bg-gray-100"}`
              }
            >
              W{w.id}: {w.title}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
```

**Step 3: 建立 Layout**

```tsx
// src/components/Layout.tsx
import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";

export default function Layout() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}
```

---

### Task 9: 建立核心視覺化元件

**Files:**
- Create: `platform/frontend/src/components/viz/GradientDescentViz.tsx`
- Create: `platform/frontend/src/components/viz/LossLandscapeViz.tsx`
- Create: `platform/frontend/src/components/viz/DecisionBoundaryViz.tsx`
- Create: `platform/frontend/src/components/viz/ActivationFunctionViz.tsx`
- Create: `platform/frontend/src/components/viz/FeatureImportanceViz.tsx`
- Create: `platform/frontend/src/components/viz/LearningCurveViz.tsx`

每個視覺化元件包含：
- 互動參數控制面板（滑桿、下拉選單）
- D3.js/Recharts 圖表渲染
- API 呼叫與即時更新
- 載入/錯誤狀態處理

（詳細實作見各週 Task）

---

### Task 10: 建立 LLM 助教聊天介面

**Files:**
- Create: `platform/frontend/src/components/llm/ChatPanel.tsx`
- Create: `platform/frontend/src/components/llm/ChatMessage.tsx`
- Create: `platform/frontend/src/components/llm/ChatInput.tsx`
- Create: `platform/frontend/src/hooks/useChat.ts`

**Step 1: 建立 useChat Hook（WebSocket 串流）**

```typescript
// src/hooks/useChat.ts
import { useState, useCallback, useRef } from "react";
import { createWebSocket } from "../lib/api";

interface Message { role: "user" | "assistant"; content: string }

export function useChat(week: number, topic: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const send = useCallback((content: string, mode: string = "tutor") => {
    const userMsg: Message = { role: "user", content };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    const ws = createWebSocket("/api/llm/ws/chat");
    wsRef.current = ws;
    let assistantContent = "";

    ws.onopen = () => {
      ws.send(JSON.stringify({
        messages: [...messages, userMsg].map((m) => ({ role: m.role, content: m.content })),
        week, topic, mode,
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "chunk") {
        assistantContent += data.content;
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.role === "assistant") {
            updated[updated.length - 1] = { ...last, content: assistantContent };
          } else {
            updated.push({ role: "assistant", content: assistantContent });
          }
          return updated;
        });
      } else if (data.type === "done") {
        setIsLoading(false);
        ws.close();
      }
    };

    ws.onerror = () => { setIsLoading(false); };
  }, [messages, week, topic]);

  const clear = useCallback(() => { setMessages([]); }, []);

  return { messages, isLoading, send, clear };
}
```

**Step 2: 建立 ChatPanel 元件**

```tsx
// src/components/llm/ChatPanel.tsx
import { useState } from "react";
import { useChat } from "../../hooks/useChat";

interface Props { week: number; topic: string }

export default function ChatPanel({ week, topic }: Props) {
  const { messages, isLoading, send, clear } = useChat(week, topic);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"tutor" | "homework">("tutor");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    send(input.trim(), mode);
    setInput("");
  };

  return (
    <div className="flex flex-col h-96 border rounded-lg">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b">
        <span className="font-medium">AI 助教</span>
        <div className="flex gap-2">
          <button
            onClick={() => setMode("tutor")}
            className={`px-2 py-1 text-xs rounded ${mode === "tutor" ? "bg-blue-500 text-white" : "bg-gray-200"}`}
          >學習模式</button>
          <button
            onClick={() => setMode("homework")}
            className={`px-2 py-1 text-xs rounded ${mode === "homework" ? "bg-orange-500 text-white" : "bg-gray-200"}`}
          >作業模式</button>
          <button onClick={clear} className="px-2 py-1 text-xs bg-gray-200 rounded">清除</button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((m, i) => (
          <div key={i} className={`${m.role === "user" ? "text-right" : "text-left"}`}>
            <div className={`inline-block px-3 py-2 rounded-lg max-w-[80%] ${
              m.role === "user" ? "bg-blue-500 text-white" : "bg-gray-100"
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {isLoading && <div className="text-gray-400 text-sm">AI 助教正在思考...</div>}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2 p-3 border-t">
        <input
          value={input} onChange={(e) => setInput(e.target.value)}
          className="flex-1 border rounded px-3 py-2 text-sm"
          placeholder="輸入問題..."
        />
        <button type="submit" className="px-4 py-2 bg-blue-500 text-white rounded text-sm">
          送出
        </button>
      </form>
    </div>
  );
}
```

---

### Task 11: 建立各週頁面路由

**Files:**
- Create: `platform/frontend/src/pages/WeekPage.tsx`
- Modify: `platform/frontend/src/App.tsx`

每週頁面包含：講義區、視覺化互動區、AI 助教面板。

```tsx
// src/pages/WeekPage.tsx
import { useParams } from "react-router-dom";
import ChatPanel from "../components/llm/ChatPanel";
// 各週視覺化元件動態載入
import { lazy, Suspense } from "react";

const weekComponents: Record<number, React.LazyExoticComponent<React.ComponentType>> = {
  4: lazy(() => import("../components/viz/GradientDescentViz")),
  5: lazy(() => import("../components/viz/DecisionBoundaryViz")),
  // ...其他週次
};

const weekTopics: Record<number, string> = {
  1: "課程導論、Python 與資料科學環境",
  2: "資料視覺化與 EDA",
  3: "監督式學習概念、資料分割與交叉驗證",
  4: "線性回歸：損失函數、梯度下降視覺化",
  // ...完整 18 週
};

export default function WeekPage() {
  const { weekId } = useParams();
  const week = Number(weekId);
  const VizComponent = weekComponents[week];
  const topic = weekTopics[week] || "";

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">第 {week} 週：{topic}</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          {VizComponent ? (
            <Suspense fallback={<div>載入中...</div>}>
              <VizComponent />
            </Suspense>
          ) : (
            <div className="bg-gray-50 p-8 rounded text-center text-gray-500">
              本週視覺化模組開發中
            </div>
          )}
        </div>
        <ChatPanel week={week} topic={topic} />
      </div>
    </div>
  );
}
```

---

## Phase 5: 18 週課程教材

### Task 12-29: 各週教材（每週一個 Task）

每週產出 6 個檔案，以統一格式建立。以下以 **Task 12（Week 1）** 為範例：

### Task 12: Week 1 — 課程導論、Python 與資料科學環境

**Files:**
- Create: `curriculum/week-01/lecture.md`
- Create: `curriculum/week-01/slides.md`
- Create: `curriculum/week-01/notebook.ipynb`
- Create: `curriculum/week-01/assignment.md`
- Create: `curriculum/week-01/rubric.md`
- Create: `curriculum/week-01/teacher-guide.md`

**lecture.md 結構：**
```markdown
# 第 1 週：課程導論、Python 與資料科學環境

## 學習目標
- 了解 ML/DL 的基本概念與應用場景
- 建立 Python 資料科學開發環境
- 熟悉 Jupyter Notebook / Google Colab 操作
- 認識課程平台與 AI 助教使用方式

## 1. 機器學習概覽
### 1.1 什麼是機器學習 (Machine Learning)？
...
### 1.2 ML 與 DL 的關係
...
### 1.3 應用場景
...

## 2. Python 環境建置
### 2.1 Anaconda / Miniconda 安裝
...
### 2.2 虛擬環境管理
...
### 2.3 核心套件安裝
...

## 3. Jupyter Notebook 入門
...

## 4. 課程平台導覽
### 4.1 視覺化互動區操作
### 4.2 AI 助教使用規範
### 4.3 作業提交流程

## 關鍵詞彙
| 中文 | 英文 | 說明 |
|------|------|------|
| 機器學習 | Machine Learning | ... |
| 深度學習 | Deep Learning | ... |
```

**teacher-guide.md 結構：**
```markdown
# 第 1 週教師手冊

## 時間分配（90 分鐘）
| 時段 | 分鐘 | 活動 | 說明 |
|------|------|------|------|
| 前段 | 30 | 課程介紹 + ML 概覽 | 投影片講解 |
| 中段 | 40 | 環境建置實作 | 學生操作、助教巡場 |
| 後段 | 20 | 平台導覽 + 討論 | 示範 AI 助教、Q&A |

## 檢核點
- [ ] 學生成功啟動 Jupyter Notebook
- [ ] 學生能執行 import numpy, pandas, matplotlib
- [ ] 學生成功登入課程平台

## AI 助教引導策略
本週以介紹為主，助教設定為「歡迎模式」...

## 常見問題
1. Anaconda 安裝失敗 → ...
2. pip 與 conda 衝突 → ...
```

**notebook.ipynb 結構：**
- Cell 1: Markdown — 標題與目標
- Cell 2: Code — 環境檢查（Python 版本、套件安裝）
- Cell 3: Code — NumPy 基礎操作
- Cell 4: Code — Pandas 讀取範例資料
- Cell 5: Code — Matplotlib 基礎圖表
- Cell 6: Markdown — 練習題

**assignment.md + rubric.md：**
每週作業說明與評分標準。

---

### Task 13-29: Week 2 ~ Week 18

（與 Task 12 相同結構，每週依主題填入對應內容。以下列出各週重點：）

| Task | 週次 | 重點內容 |
|------|------|---------|
| 13 | W02 | Matplotlib/Seaborn/Plotly 互動圖表、EDA 流程 |
| 14 | W03 | train/test split、k-fold CV、stratified sampling |
| 15 | W04 | 損失函數推導、梯度下降動畫、學習率實驗 |
| 16 | W05 | Logistic Regression、決策邊界、ROC/PR 曲線 |
| 17 | W06 | SVM 線性/核、支持向量視覺化、核技巧 |
| 18 | W07 | DecisionTree、RandomForest、GBDT、集成比較 |
| 19 | W08 | Feature Importance、Permutation、SHAP 值 |
| 20 | W09 | Pipeline、Encoding、Scaling、Feature Selection |
| 21 | W10 | GridSearch、RandomSearch、學習曲線、驗證曲線 |
| 22 | W11 | Perceptron、激活函數、BatchNorm、正則化 |
| 23 | W12 | Conv2D 視覺化、特徵圖、CAM/Grad-CAM |
| 24 | W13 | RNN/LSTM/GRU 結構、Attention、Transformer 概念 |
| 25 | W14 | LR Scheduler、Early Stopping、Data Augmentation |
| 26 | W15 | Confusion Matrix 深入、Fairness Metrics、Robustness |
| 27 | W16 | MLflow、Model Registry、Docker 部署、監測 |
| 28 | W17 | Embeddings、RAG 概念、Prompt Engineering 入門 |
| 29 | W18 | 專題整合、展示模板、課程回饋問卷 |

---

## Phase 6: 學習分析儀表板

### Task 30: 建立後端學習分析模組

**Files:**
- Create: `platform/backend/app/analytics/__init__.py`
- Create: `platform/backend/app/analytics/models.py`
- Create: `platform/backend/app/analytics/tracker.py`
- Create: `platform/backend/app/api/analytics_routes.py`

追蹤：答題時間、錯誤型態、LLM 查詢主題、學習進度。
以 SQLite 儲存，提供 REST API 給前端儀表板。

### Task 31: 建立前端學習分析儀表板

**Files:**
- Create: `platform/frontend/src/pages/Dashboard.tsx`
- Create: `platform/frontend/src/components/analytics/ProgressChart.tsx`
- Create: `platform/frontend/src/components/analytics/ErrorHeatmap.tsx`
- Create: `platform/frontend/src/components/analytics/TopicCloud.tsx`

使用 Recharts 呈現：學習進度折線圖、錯誤熱力圖、LLM 查詢主題雲。

---

## Phase 7: 整合與完善

### Task 32: 建立操作手冊

**Files:**
- Create: `curriculum/platform-manual.md`

內容：平台登入、視覺化操作、任務提交、AI 助教使用規範、常見問題。

### Task 33: 建立評量工具包

**Files:**
- Create: `curriculum/assessment/concept-quiz-template.md`
- Create: `curriculum/assessment/satisfaction-survey.md`
- Create: `curriculum/assessment/project-rubric.md`

概念理解測驗模板、平台滿意度問卷、期末專題評分表。

### Task 34: 範例資料集準備

**Files:**
- Create: `datasets/README.md`（含 Kaggle 下載指引）
- Create: `datasets/download.py`（自動下載腳本）

每週對應的 Kaggle 資料集清單與下載工具。

### Task 35: 端對端測試與文件

**Files:**
- Create: `platform/backend/tests/test_integration.py`
- Create: `platform/frontend/src/__tests__/App.test.tsx`

驗證前後端串接、LLM 助教回應、視覺化元件渲染。

---

## 執行順序依賴圖

```
Task 1 (目錄結構)
  ├── Task 2 (FastAPI) ──┐
  │    ├── Task 4 (LLM 抽象層) → Task 5 (Hint Ladder) → Task 6 (LLM API)
  │    ├── Task 7 (ML 模型 API)
  │    └── Task 30 (學習分析後端)
  └── Task 3 (React) ──┐
       ├── Task 8 (前端基礎) → Task 9 (視覺化元件) → Task 11 (週頁面)
       ├── Task 10 (聊天介面)
       └── Task 31 (分析儀表板)

Task 12-29 (18 週教材) ── 可平行，不依賴平台
Task 32-34 (手冊/評量/資料) ── 可平行
Task 35 (整合測試) ── 依賴全部完成
```
