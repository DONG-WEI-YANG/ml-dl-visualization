# AI 助教系統改善 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance the AI teaching assistant with Markdown/LaTeX rendering, comprehensive frontend tests, and a 42-layer tiny NLP pipeline using real NLP libraries.

**Architecture:** Three parallel workstreams: (1) Frontend ChatPanel rendering upgrade with react-markdown + KaTeX + highlight.js, (2) Frontend test suite expansion from 5 to 35+ tests using vitest + testing-library, (3) Backend NLP pipeline expansion from 7 to 42 layers using jieba, snownlp, langdetect, nltk, textstat, sentence-transformers, rapidfuzz.

**Tech Stack:** React 19, TypeScript, Vite, Tailwind CSS, vitest, FastAPI, Python 3.11+, scikit-learn, jieba, snownlp, langdetect, nltk, textstat, sentence-transformers, rapidfuzz

---

## Workstream A: ChatPanel Markdown + LaTeX Rendering

### Task A1: Install frontend rendering dependencies

**Files:**
- Modify: `platform/frontend/package.json`

**Step 1: Install packages**

Run:
```bash
cd platform/frontend
npm install react-markdown remark-math remark-gfm rehype-katex rehype-highlight katex
npm install -D @types/katex
```

Expected: package.json updated with new dependencies.

**Step 2: Verify installation**

Run: `cd platform/frontend && npm ls react-markdown remark-math rehype-katex rehype-highlight`
Expected: All packages listed without errors.

---

### Task A2: Add CSS imports for KaTeX and highlight.js

**Files:**
- Modify: `platform/frontend/src/main.tsx`

**Step 1: Add CSS imports at top of main.tsx**

After the existing `import "./index.css"` line, add:

```tsx
import "katex/dist/katex.min.css";
import "highlight.js/styles/github.css";
```

**Step 2: Verify build**

Run: `cd platform/frontend && npx tsc --noEmit`
Expected: No type errors.

---

### Task A3: Create MarkdownRenderer component

**Files:**
- Create: `platform/frontend/src/components/llm/MarkdownRenderer.tsx`

**Step 1: Create the component**

```tsx
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";

interface Props {
  content: string;
}

export default function MarkdownRenderer({ content }: Props) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath, remarkGfm]}
      rehypePlugins={[rehypeKatex, rehypeHighlight]}
      components={{
        pre({ children }) {
          return <pre className="rounded-lg overflow-x-auto text-sm my-2">{children}</pre>;
        },
        code({ className, children, ...props }) {
          const isInline = !className;
          if (isInline) {
            return (
              <code className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded text-xs font-mono" {...props}>
                {children}
              </code>
            );
          }
          return <code className={className} {...props}>{children}</code>;
        },
        table({ children }) {
          return (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full text-sm border-collapse border border-gray-300">{children}</table>
            </div>
          );
        },
        th({ children }) {
          return <th className="border border-gray-300 px-3 py-1.5 bg-gray-100 text-left font-medium">{children}</th>;
        },
        td({ children }) {
          return <td className="border border-gray-300 px-3 py-1.5">{children}</td>;
        },
        p({ children }) {
          return <p className="my-1.5 leading-relaxed">{children}</p>;
        },
        ul({ children }) {
          return <ul className="list-disc pl-5 my-1.5 space-y-0.5">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="list-decimal pl-5 my-1.5 space-y-0.5">{children}</ol>;
        },
        blockquote({ children }) {
          return <blockquote className="border-l-3 border-blue-400 pl-3 my-2 text-gray-600 italic">{children}</blockquote>;
        },
        h3({ children }) {
          return <h3 className="font-semibold text-base mt-3 mb-1">{children}</h3>;
        },
        h4({ children }) {
          return <h4 className="font-medium text-sm mt-2 mb-1">{children}</h4>;
        },
        strong({ children }) {
          return <strong className="font-semibold text-gray-900">{children}</strong>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
```

**Step 2: Verify build**

Run: `cd platform/frontend && npx tsc --noEmit`
Expected: No type errors.

---

### Task A4: Integrate MarkdownRenderer into ChatPanel

**Files:**
- Modify: `platform/frontend/src/components/llm/ChatPanel.tsx`

**Step 1: Replace the plain text rendering for assistant messages**

Replace the existing message rendering (the `<pre>` tag inside the message bubble) with:

```tsx
import MarkdownRenderer from "./MarkdownRenderer";
```

Then change the message content rendering from:

```tsx
<pre className="whitespace-pre-wrap font-sans">{m.content}</pre>
```

To:

```tsx
{m.role === "assistant" ? (
  <MarkdownRenderer content={m.content} />
) : (
  <pre className="whitespace-pre-wrap font-sans">{m.content}</pre>
)}
```

**Step 2: Verify build and dev server**

Run: `cd platform/frontend && npx tsc --noEmit && npm run build`
Expected: Build succeeds without errors.

---

## Workstream B: Frontend Test Coverage

### Task B1: Create ChatPanel test

**Files:**
- Create: `platform/frontend/src/test/ChatPanel.test.tsx`

**Step 1: Write the test file**

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ChatPanel from "../components/llm/ChatPanel";

// Mock useChat hook
const mockSend = vi.fn();
const mockClear = vi.fn();
let mockMessages: { role: string; content: string }[] = [];
let mockIsLoading = false;

vi.mock("../hooks/useChat", () => ({
  useChat: () => ({
    messages: mockMessages,
    isLoading: mockIsLoading,
    send: mockSend,
    clear: mockClear,
  }),
}));

describe("ChatPanel", () => {
  beforeEach(() => {
    mockMessages = [];
    mockIsLoading = false;
    mockSend.mockClear();
    mockClear.mockClear();
  });

  it("renders welcome message when no messages", () => {
    render(<ChatPanel week={1} topic="Python 環境" />);
    expect(screen.getByText("歡迎使用 AI 助教！")).toBeInTheDocument();
  });

  it("switches between tutor and homework modes", () => {
    render(<ChatPanel week={1} topic="Python 環境" />);
    const homeworkBtn = screen.getByText("作業模式");
    fireEvent.click(homeworkBtn);
    expect(screen.getByText(/作業模式/)).toBeInTheDocument();
  });

  it("sends a message on form submit", () => {
    render(<ChatPanel week={4} topic="梯度下降" />);
    const input = screen.getByPlaceholderText("輸入你的問題...");
    fireEvent.change(input, { target: { value: "什麼是梯度下降？" } });
    fireEvent.submit(input.closest("form")!);
    expect(mockSend).toHaveBeenCalledWith("什麼是梯度下降？", "tutor");
  });

  it("does not send empty message", () => {
    render(<ChatPanel week={1} topic="test" />);
    const input = screen.getByPlaceholderText("輸入你的問題...");
    fireEvent.submit(input.closest("form")!);
    expect(mockSend).not.toHaveBeenCalled();
  });

  it("clears messages on clear button", () => {
    render(<ChatPanel week={1} topic="test" />);
    fireEvent.click(screen.getByText("清除"));
    expect(mockClear).toHaveBeenCalled();
  });

  it("disables input when loading", () => {
    mockIsLoading = true;
    render(<ChatPanel week={1} topic="test" />);
    expect(screen.getByPlaceholderText("輸入你的問題...")).toBeDisabled();
  });

  it("displays user and assistant messages", () => {
    mockMessages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ];
    render(<ChatPanel week={1} topic="test" />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("shows loading indicator", () => {
    mockIsLoading = true;
    render(<ChatPanel week={1} topic="test" />);
    expect(screen.getByText("AI 助教思考中...")).toBeInTheDocument();
  });
});
```

**Step 2: Run the test**

Run: `cd platform/frontend && npx vitest run src/test/ChatPanel.test.tsx`
Expected: All 8 tests pass.

---

### Task B2: Create QuizPanel test

**Files:**
- Create: `platform/frontend/src/test/QuizPanel.test.tsx`

**Step 1: Write the test file**

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import QuizPanel from "../components/quiz/QuizPanel";

// Mock fetchAPI
const mockFetchAPI = vi.fn();
vi.mock("../lib/api", () => ({
  fetchAPI: (...args: unknown[]) => mockFetchAPI(...args),
}));

const MOCK_QUESTIONS = {
  questions: [
    { id: "w01q1", question: "Python 中用來處理表格資料的主要套件是？", options: ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"] },
    { id: "w01q2", question: "Jupyter Notebook 的副檔名是？", options: [".py", ".ipynb", ".jnb", ".nb"] },
  ],
};

const MOCK_GRADE = {
  score: 1,
  total: 2,
  percentage: 50,
  results: [
    { id: "w01q1", correct: true, correct_answer: 1, user_answer: 1 },
    { id: "w01q2", correct: false, correct_answer: 1, user_answer: 0 },
  ],
};

describe("QuizPanel", () => {
  beforeEach(() => {
    mockFetchAPI.mockReset();
  });

  it("loads and displays questions", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => {
      expect(screen.getByText(/Python 中用來處理表格資料/)).toBeInTheDocument();
    });
  });

  it("allows selecting answers", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => screen.getByText(/Python 中用來處理表格資料/));
    const pandasBtn = screen.getByText("Pandas");
    fireEvent.click(pandasBtn);
    expect(screen.getByText(/已作答 1\/2/)).toBeInTheDocument();
  });

  it("submits answers and shows results", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => screen.getByText(/Python 中用來處理表格資料/));

    // Select answers
    fireEvent.click(screen.getByText("Pandas"));
    fireEvent.click(screen.getByText(".py"));

    // Submit
    mockFetchAPI.mockResolvedValueOnce(MOCK_GRADE);
    fireEvent.click(screen.getByText("提交答案"));
    await waitFor(() => {
      expect(screen.getByText(/1\/2/)).toBeInTheDocument();
    });
  });

  it("shows reset button after grading", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => screen.getByText(/Python 中用來處理表格資料/));

    fireEvent.click(screen.getByText("Pandas"));
    fireEvent.click(screen.getByText(".py"));

    mockFetchAPI.mockResolvedValueOnce(MOCK_GRADE);
    fireEvent.click(screen.getByText("提交答案"));
    await waitFor(() => {
      expect(screen.getByText("重新作答")).toBeInTheDocument();
    });
  });

  it("renders nothing when no questions", async () => {
    mockFetchAPI.mockResolvedValueOnce({ questions: [] });
    const { container } = render(<QuizPanel week={99} />);
    await waitFor(() => {
      expect(container.innerHTML).toBe("");
    });
  });
});
```

**Step 2: Run the test**

Run: `cd platform/frontend && npx vitest run src/test/QuizPanel.test.tsx`
Expected: All 5 tests pass.

---

### Task B3: Create Home page test

**Files:**
- Create: `platform/frontend/src/test/Home.test.tsx`

**Step 1: Write the test file**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Home from "../pages/Home";

describe("Home", () => {
  it("renders page title", () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(screen.getByText("ML/DL 視覺化互動教學平台")).toBeInTheDocument();
  });

  it("renders all 18 week cards", () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    const list = screen.getByRole("list", { name: /18-week curriculum/ });
    expect(list.children.length).toBe(18);
  });

  it("renders core and advanced labels", () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    const advancedLabels = screen.getAllByText("進階");
    const coreLabels = screen.getAllByText("核心");
    expect(advancedLabels.length).toBeGreaterThan(0);
    expect(coreLabels.length).toBeGreaterThan(0);
  });

  it("has links to week pages", () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    const links = screen.getAllByRole("link");
    const weekLinks = links.filter((l) => l.getAttribute("href")?.startsWith("/week/"));
    expect(weekLinks.length).toBe(18);
  });
});
```

**Step 2: Run**

Run: `cd platform/frontend && npx vitest run src/test/Home.test.tsx`
Expected: 4 tests pass.

---

### Task B4: Create Sidebar test

**Files:**
- Create: `platform/frontend/src/test/Sidebar.test.tsx`

**Step 1: Write the test file**

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Sidebar from "../components/Sidebar";

vi.mock("../hooks/useAuth", () => ({
  useAuth: () => ({
    user: { username: "admin", display_name: "管理員", role: "admin" },
    logout: vi.fn(),
  }),
}));

describe("Sidebar", () => {
  it("renders platform title", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.getByText("ML/DL 視覺化")).toBeInTheDocument();
  });

  it("renders 18 week navigation links", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    const links = screen.getAllByRole("link");
    const weekLinks = links.filter((l) => l.getAttribute("href")?.match(/\/week\/\d+/));
    expect(weekLinks.length).toBe(18);
  });

  it("shows dashboard link", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.getByText("學習分析")).toBeInTheDocument();
  });

  it("shows admin link for admin user", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.getByText("系統管理")).toBeInTheDocument();
  });

  it("shows user display name", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.getByText("管理員")).toBeInTheDocument();
  });

  it("highlights current week", () => {
    render(<MemoryRouter initialEntries={["/week/4"]}><Sidebar /></MemoryRouter>);
    const links = screen.getAllByRole("link");
    const week4 = links.find((l) => l.getAttribute("href") === "/week/4");
    expect(week4?.className).toContain("bg-blue-50");
  });
});
```

**Step 2: Run**

Run: `cd platform/frontend && npx vitest run src/test/Sidebar.test.tsx`
Expected: 6 tests pass.

---

### Task B5: Create WeekPage test

**Files:**
- Create: `platform/frontend/src/test/WeekPage.test.tsx`

**Step 1: Write the test file**

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import WeekPage from "../pages/WeekPage";

vi.mock("../hooks/useChat", () => ({
  useChat: () => ({ messages: [], isLoading: false, send: vi.fn(), clear: vi.fn() }),
}));

vi.mock("../lib/api", () => ({
  fetchAPI: vi.fn().mockResolvedValue({ questions: [] }),
}));

function renderWeekPage(weekId: string) {
  return render(
    <MemoryRouter initialEntries={[`/week/${weekId}`]}>
      <Routes>
        <Route path="/week/:weekId" element={<WeekPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("WeekPage", () => {
  it("renders week title for valid week", () => {
    renderWeekPage("4");
    expect(screen.getByText(/第 4 週/)).toBeInTheDocument();
    expect(screen.getByText(/線性回歸與梯度下降/)).toBeInTheDocument();
  });

  it("shows error for invalid week", () => {
    renderWeekPage("99");
    expect(screen.getByText(/找不到第 99 週的內容/)).toBeInTheDocument();
  });

  it("renders visualization section", () => {
    renderWeekPage("1");
    expect(screen.getByText("視覺化互動區")).toBeInTheDocument();
  });

  it("renders curriculum download links", () => {
    renderWeekPage("1");
    expect(screen.getByText("講義")).toBeInTheDocument();
    expect(screen.getByText("投影片")).toBeInTheDocument();
    expect(screen.getByText("Notebook")).toBeInTheDocument();
    expect(screen.getByText("作業")).toBeInTheDocument();
  });

  it("renders chat panel", () => {
    renderWeekPage("1");
    expect(screen.getByText("AI 助教")).toBeInTheDocument();
  });
});
```

**Step 2: Run**

Run: `cd platform/frontend && npx vitest run src/test/WeekPage.test.tsx`
Expected: 5 tests pass.

---

### Task B6: Create Dashboard test

**Files:**
- Create: `platform/frontend/src/test/Dashboard.test.tsx`

**Step 1: Write the test file**

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import Dashboard from "../pages/Dashboard";

const mockFetchAPI = vi.fn();
vi.mock("../lib/api", () => ({
  fetchAPI: (...args: unknown[]) => mockFetchAPI(...args),
}));

// Mock recharts to avoid SVG rendering issues in test
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

const MOCK_SUMMARY = {
  total_students: 30,
  total_events: 1500,
  average_score: 78.5,
  popular_llm_topics: [
    { topic: "梯度下降", count: 45 },
    { topic: "CNN", count: 30 },
  ],
};

const MOCK_STUDENT = {
  student_id: "S001",
  total_weeks_completed: 10,
  total_time_minutes: 450,
  average_score: 82.3,
  weekly_progress: [
    { week: 1, completed: true, quiz_score: 90, assignment_score: 85, llm_interactions: 5, time_spent_minutes: 30 },
  ],
  llm_topics: [{ topic: "過擬合", count: 3 }],
  error_patterns: [{ type: "TypeError", count: 2 }],
};

describe("Dashboard", () => {
  beforeEach(() => {
    mockFetchAPI.mockReset();
  });

  it("shows loading state", () => {
    mockFetchAPI.mockReturnValue(new Promise(() => {}));
    render(<Dashboard />);
    expect(screen.getByText("載入中...")).toBeInTheDocument();
  });

  it("displays class summary", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_SUMMARY);
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText("30")).toBeInTheDocument();
      expect(screen.getByText("1500")).toBeInTheDocument();
      expect(screen.getByText("78.5")).toBeInTheDocument();
    });
  });

  it("shows error on fetch failure", async () => {
    mockFetchAPI.mockRejectedValueOnce(new Error("fail"));
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText("無法載入班級總覽資料")).toBeInTheDocument();
    });
  });

  it("looks up student analytics", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_SUMMARY);
    render(<Dashboard />);
    await waitFor(() => screen.getByText("30"));

    const input = screen.getByPlaceholderText("輸入學生 ID");
    fireEvent.change(input, { target: { value: "S001" } });

    mockFetchAPI.mockResolvedValueOnce(MOCK_STUDENT);
    fireEvent.click(screen.getByText("查詢"));
    await waitFor(() => {
      expect(screen.getByText("10 / 18")).toBeInTheDocument();
      expect(screen.getByText("82.3")).toBeInTheDocument();
    });
  });

  it("shows student not found error", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_SUMMARY);
    render(<Dashboard />);
    await waitFor(() => screen.getByText("30"));

    const input = screen.getByPlaceholderText("輸入學生 ID");
    fireEvent.change(input, { target: { value: "INVALID" } });

    mockFetchAPI.mockRejectedValueOnce(new Error("404"));
    fireEvent.click(screen.getByText("查詢"));
    await waitFor(() => {
      expect(screen.getByText("找不到該學生資料")).toBeInTheDocument();
    });
  });
});
```

**Step 2: Run**

Run: `cd platform/frontend && npx vitest run src/test/Dashboard.test.tsx`
Expected: 5 tests pass.

---

### Task B7: Run full frontend test suite

**Step 1: Run all tests**

Run: `cd platform/frontend && npx vitest run`
Expected: 35+ tests pass (5 existing + 8 + 5 + 4 + 6 + 5 + 5 new).

---

## Workstream C: 42-Layer Tiny NLP Pipeline

### Task C1: Install NLP dependencies

**Files:**
- Modify: `platform/backend/requirements.txt`

**Step 1: Add new dependencies to requirements.txt**

Append:
```
jieba>=0.42.1
snownlp>=0.12.3
langdetect>=1.0.9
nltk>=3.9.0
textstat>=0.7.4
sentence-transformers>=3.0.0
rapidfuzz>=3.10.0
```

**Step 2: Install**

Run: `cd platform/backend && pip install -r requirements.txt`
Expected: All packages install successfully.

**Step 3: Download NLTK data**

Run: `python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords')"`
Expected: Downloads complete.

---

### Task C2: Expand NLPContext with 42-layer fields

**Files:**
- Modify: `platform/backend/app/nlp/pipeline.py`

**Step 1: Replace the entire NLPContext dataclass and add pipeline runner**

Replace the contents of `pipeline.py` with:

```python
"""NLP Pipeline — 42-layer micro-NLP architecture for AI Tutor.

Architecture (7 groups, 42 layers):
  A. Text Preprocessing (L1-6): segmentation, POS, sentences, language, normalize, stopwords
  B. Student Understanding (L7-14): intent, sub-intent, emotion, sentiment, frustration, confidence, urgency, politeness
  C. Student Level (L15-20): difficulty, vocabulary, fluency, learning-style, misconception, knowledge-gap
  D. Content Analysis (L21-27): keywords, concepts, NER, code-detect, math-detect, question-quality, readability
  E. Context & Memory (L28-32): conversation, topic-continuity, hint-ladder, session-summary, knowledge-state
  F. Retrieval (L33-36): query-expand, RAG, semantic-rerank, cross-week
  G. Response (L37-42): assemble, complexity-adjust, citation, follow-up, encouragement, completeness-check
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NLPContext:
    """Shared context object passed through all 42 NLP layers."""
    # ── Input ──
    user_message: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    week: int = 1
    topic: str = ""
    is_homework: bool = False
    student_id: str = ""

    # ── A. Text Preprocessing (L1-6) ──
    tokens: list[str] = field(default_factory=list)           # L1 ChineseSegmenter
    pos_tags: list[tuple[str, str]] = field(default_factory=list)  # L2 POSTagger
    sentences: list[str] = field(default_factory=list)         # L3 SentenceSplitter
    language: str = "zh"                                       # L4 LanguageDetector
    language_confidence: float = 0.0
    normalized_text: str = ""                                  # L5 TextNormalizer
    filtered_tokens: list[str] = field(default_factory=list)   # L6 StopwordFilter

    # ── B. Student Understanding (L7-14) ──
    intent: str = "general"                                    # L7 IntentClassifier
    intent_confidence: float = 0.0
    sub_intent: str = ""                                       # L8 SubIntentDetector
    sub_intent_confidence: float = 0.0
    emotion: str = "neutral"                                   # L9 EmotionClassifier
    emotion_confidence: float = 0.0
    sentiment_score: float = 0.5                               # L10 SentimentScorer (0=negative, 1=positive)
    frustration_level: int = 0                                 # L11 FrustrationEscalator (0-5)
    confidence_level: float = 0.5                              # L12 ConfidenceEstimator (0-1)
    urgency: str = "normal"                                    # L13 UrgencyDetector
    politeness_score: float = 0.5                              # L14 PolitenessDetector (0-1)

    # ── C. Student Level (L15-20) ──
    student_level: str = "beginner"                            # L15 DifficultyAssessor
    uses_technical_terms: bool = False
    question_complexity: str = "simple"
    vocabulary_score: float = 0.0                              # L16 VocabularyLevelScorer (0-1)
    technical_fluency: float = 0.0                             # L17 TechnicalFluencyScorer (0-1)
    learning_style: str = "balanced"                           # L18 LearningStyleDetector
    misconceptions: list[str] = field(default_factory=list)    # L19 MisconceptionDetector
    knowledge_gaps: list[str] = field(default_factory=list)    # L20 KnowledgeGapDetector

    # ── D. Content Analysis (L21-27) ──
    keywords: list[str] = field(default_factory=list)          # L21 KeywordExtractor
    keyword_scores: list[tuple[str, float]] = field(default_factory=list)
    domain_concepts: list[str] = field(default_factory=list)   # L22 DomainConceptMatcher
    named_entities: list[dict] = field(default_factory=list)   # L23 NER
    has_code: bool = False                                     # L24 CodeBlockDetector
    code_language: str = ""
    code_blocks: list[str] = field(default_factory=list)
    has_math: bool = False                                     # L25 MathExpressionDetector
    math_expressions: list[str] = field(default_factory=list)
    question_quality: float = 0.5                              # L26 QuestionQualityScorer (0-1)
    quality_feedback: str = ""
    readability_score: float = 0.0                             # L27 ReadabilityScorer

    # ── E. Context & Memory (L28-32) ──
    turn_count: int = 0                                        # L28 ConversationTracker
    is_followup: bool = False
    previous_intent: str = ""
    topic_continuity: float = 0.0                              # L29 TopicContinuityDetector (0-1)
    continued_topic: str = ""
    hint_level: int = 1                                        # L30 HintLadderManager (1-4)
    session_summary: str = ""                                  # L31 SessionSummarizer
    known_concepts: list[str] = field(default_factory=list)    # L32 KnowledgeStateTracker
    unknown_concepts: list[str] = field(default_factory=list)

    # ── F. Retrieval (L33-36) ──
    expanded_query: str = ""                                   # L33 QueryExpander
    rag_context: str = ""                                      # L34 RAGRetriever
    rag_sources: list[str] = field(default_factory=list)
    reranked_results: list[dict] = field(default_factory=list) # L35 SemanticReranker
    cross_week_links: list[dict] = field(default_factory=list) # L36 CrossWeekLinker

    # ── G. Response (L37-42) ──
    response: str = ""                                         # L37 ResponseAssembler
    response_complexity: str = "moderate"                      # L38 ComplexityAdjuster
    citations: list[str] = field(default_factory=list)         # L39 CitationInjector
    follow_up_questions: list[str] = field(default_factory=list)  # L40 FollowUpGenerator
    encouragement: str = ""                                    # L41 EncouragementGenerator
    completeness_score: float = 0.0                            # L42 ResponseCompletenessChecker
    completeness_missing: list[str] = field(default_factory=list)

    # ── Pipeline metadata ──
    layers_executed: list[str] = field(default_factory=list)
    total_processing_ms: float = 0.0


def run_pipeline(ctx: NLPContext, layers: list) -> NLPContext:
    """Execute all NLP layers sequentially, collecting timing and layer names."""
    import time
    start = time.time()
    for layer_fn in layers:
        layer_name = layer_fn.__name__
        try:
            ctx = layer_fn(ctx)
            ctx.layers_executed.append(layer_name)
        except Exception as e:
            logger.warning("NLP layer %s failed: %s", layer_name, e)
            ctx.layers_executed.append(f"{layer_name}:ERROR")
    ctx.total_processing_ms = (time.time() - start) * 1000
    return ctx
```

**Step 2: Verify import**

Run: `cd platform/backend && python -c "from app.nlp.pipeline import NLPContext, run_pipeline; print('OK')"`
Expected: `OK`

---

### Task C3: Create Group A — Text Preprocessing layers (L1-6)

**Files:**
- Create: `platform/backend/app/nlp/preprocessing.py`

**Step 1: Write the 6 preprocessing layers**

```python
"""Group A: Text Preprocessing Layers (L1-6).

L1 ChineseSegmenter — jieba
L2 POSTagger — jieba.posseg
L3 SentenceSplitter — nltk.punkt / regex
L4 LanguageDetector — langdetect
L5 TextNormalizer — custom
L6 StopwordFilter — nltk + custom Chinese
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Lazy imports for heavy libraries ──
_jieba = None
_posseg = None
_langdetect = None
_sent_tokenize = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
    return _jieba


def _get_posseg():
    global _posseg
    if _posseg is None:
        import jieba.posseg as pseg
        _posseg = pseg
    return _posseg


def _get_langdetect():
    global _langdetect
    if _langdetect is None:
        import langdetect
        _langdetect = langdetect
    return _langdetect


def _get_sent_tokenize():
    global _sent_tokenize
    if _sent_tokenize is None:
        try:
            from nltk.tokenize import sent_tokenize
            _sent_tokenize = sent_tokenize
        except Exception:
            _sent_tokenize = None
    return _sent_tokenize


# ── Chinese + English stopwords ──

STOPWORDS_ZH = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "這",
    "上", "也", "到", "說", "要", "會", "可以", "請問", "想", "個", "中", "嗎",
    "怎麼", "什麼", "為什麼", "如何", "能", "用", "讓", "吧", "呢", "跟", "從",
    "那", "被", "把", "給", "很", "太", "再", "還", "去", "來", "做", "看",
    "但", "所以", "因為", "如果", "然後", "或", "而", "對", "比", "以", "其",
    "你", "他", "她", "它", "們", "這個", "那個", "哪", "嗯", "喔", "啊",
}

STOPWORDS_EN = set()
try:
    from nltk.corpus import stopwords as _nltk_sw
    STOPWORDS_EN = set(_nltk_sw.words("english"))
except Exception:
    STOPWORDS_EN = {"the", "is", "a", "an", "in", "of", "to", "and", "for", "it", "this",
                    "that", "how", "what", "why", "can", "do", "i", "my", "me", "be", "are"}


# ── L1: Chinese Segmenter ──

def chinese_segmenter(ctx):
    """L1: Segment text using jieba."""
    jieba = _get_jieba()
    ctx.tokens = list(jieba.cut(ctx.user_message))
    return ctx


# ── L2: POS Tagger ──

def pos_tagger(ctx):
    """L2: POS tagging using jieba.posseg."""
    pseg = _get_posseg()
    ctx.pos_tags = [(word, flag) for word, flag in pseg.cut(ctx.user_message)]
    return ctx


# ── L3: Sentence Splitter ──

_SENT_SPLIT_RE = re.compile(r'(?<=[。！？.!?])\s*|(?<=\n)\s*')


def sentence_splitter(ctx):
    """L3: Split text into sentences using nltk (English) and regex (Chinese)."""
    text = ctx.user_message.strip()
    if not text:
        ctx.sentences = []
        return ctx

    # Try nltk for English-heavy text
    sent_tok = _get_sent_tokenize()
    if sent_tok and ctx.language == "en":
        ctx.sentences = sent_tok(text)
    else:
        # Regex-based for Chinese / mixed
        parts = _SENT_SPLIT_RE.split(text)
        ctx.sentences = [s.strip() for s in parts if s.strip()]

    if not ctx.sentences:
        ctx.sentences = [text]

    return ctx


# ── L4: Language Detector ──

def language_detector(ctx):
    """L4: Detect language using langdetect."""
    ld = _get_langdetect()
    try:
        result = ld.detect_langs(ctx.user_message)
        if result:
            ctx.language = result[0].lang
            ctx.language_confidence = result[0].prob
        else:
            ctx.language = "zh"
            ctx.language_confidence = 0.5
    except Exception:
        # Default to Chinese for this course
        ctx.language = "zh"
        ctx.language_confidence = 0.5
    return ctx


# ── L5: Text Normalizer ──

_FULLWIDTH_MAP = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ（）【】",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[]",
)


def text_normalizer(ctx):
    """L5: Normalize text — fullwidth→halfwidth, collapse whitespace."""
    text = ctx.user_message
    # Fullwidth to halfwidth
    text = text.translate(_FULLWIDTH_MAP)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    ctx.normalized_text = text
    return ctx


# ── L6: Stopword Filter ──

def stopword_filter(ctx):
    """L6: Remove stopwords from token list."""
    all_stops = STOPWORDS_ZH | STOPWORDS_EN
    ctx.filtered_tokens = [t for t in ctx.tokens if t.strip() and t not in all_stops and len(t.strip()) > 0]
    return ctx
```

**Step 2: Verify import**

Run: `cd platform/backend && python -c "from app.nlp.preprocessing import chinese_segmenter, pos_tagger, sentence_splitter, language_detector, text_normalizer, stopword_filter; print('6 layers OK')"`
Expected: `6 layers OK`

---

### Task C4: Create Group B — Student Understanding layers (L7-14)

**Files:**
- Modify: `platform/backend/app/nlp/intent.py` (keep existing, it's L7)
- Modify: `platform/backend/app/nlp/emotion.py` (keep existing, it's L9)
- Create: `platform/backend/app/nlp/understanding.py` (L8, L10-14)

**Step 1: Create understanding.py with new layers**

```python
"""Group B: Student Understanding Layers (L8, L10-14).

L7 IntentClassifier — existing intent.py
L8 SubIntentDetector — sklearn sub-categories
L9 EmotionClassifier — existing emotion.py
L10 SentimentScorer — snownlp
L11 FrustrationEscalator — multi-turn escalation
L12 ConfidenceEstimator — snownlp + pattern
L13 UrgencyDetector — pattern + time NER
L14 PolitenessDetector — snownlp + pattern
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_snownlp = None


def _get_snownlp():
    global _snownlp
    if _snownlp is None:
        from snownlp import SnowNLP
        _snownlp = SnowNLP
    return _snownlp


# ── L8: Sub-Intent Detector ──

SUB_INTENTS = {
    "definition": ["basic_definition", "formal_definition", "intuitive_explanation"],
    "how": ["step_by_step", "code_implementation", "tool_usage"],
    "debug": ["error_message", "wrong_output", "environment_issue"],
    "compare": ["pros_cons", "when_to_use", "performance_comparison"],
    "code": ["syntax_help", "library_usage", "full_example"],
}


def sub_intent_detector(ctx: NLPContext) -> NLPContext:
    """L8: Detect sub-intent within the main intent category."""
    text_lower = ctx.user_message.lower()
    subs = SUB_INTENTS.get(ctx.intent, [])
    if not subs:
        ctx.sub_intent = ""
        ctx.sub_intent_confidence = 0.0
        return ctx

    # Simple heuristic matching
    if ctx.intent == "definition":
        if any(w in text_lower for w in ["直覺", "簡單", "白話", "intuition", "simple"]):
            ctx.sub_intent = "intuitive_explanation"
        elif any(w in text_lower for w in ["正式", "數學", "formal", "rigorous"]):
            ctx.sub_intent = "formal_definition"
        else:
            ctx.sub_intent = "basic_definition"
    elif ctx.intent == "how":
        if any(w in text_lower for w in ["程式", "code", "python", "import"]):
            ctx.sub_intent = "code_implementation"
        elif any(w in text_lower for w in ["步驟", "step", "流程"]):
            ctx.sub_intent = "step_by_step"
        else:
            ctx.sub_intent = "tool_usage"
    elif ctx.intent == "debug":
        if any(w in text_lower for w in ["error", "錯誤", "traceback", "exception"]):
            ctx.sub_intent = "error_message"
        elif any(w in text_lower for w in ["裝", "install", "版本", "version"]):
            ctx.sub_intent = "environment_issue"
        else:
            ctx.sub_intent = "wrong_output"
    elif ctx.intent == "compare":
        if any(w in text_lower for w in ["優缺", "pros", "cons", "好壞"]):
            ctx.sub_intent = "pros_cons"
        elif any(w in text_lower for w in ["何時", "when", "場景", "scenario"]):
            ctx.sub_intent = "when_to_use"
        else:
            ctx.sub_intent = "performance_comparison"
    elif ctx.intent == "code":
        if any(w in text_lower for w in ["語法", "syntax", "怎麼寫"]):
            ctx.sub_intent = "syntax_help"
        elif any(w in text_lower for w in ["套件", "library", "模組", "import"]):
            ctx.sub_intent = "library_usage"
        else:
            ctx.sub_intent = "full_example"
    else:
        ctx.sub_intent = subs[0] if subs else ""

    ctx.sub_intent_confidence = 0.7
    return ctx


# ── L10: Sentiment Scorer ──

def sentiment_scorer(ctx: NLPContext) -> NLPContext:
    """L10: Continuous sentiment score (0=negative, 1=positive) using SnowNLP."""
    SnowNLP = _get_snownlp()
    try:
        s = SnowNLP(ctx.user_message)
        ctx.sentiment_score = round(s.sentiments, 3)
    except Exception:
        ctx.sentiment_score = 0.5
    return ctx


# ── L11: Frustration Escalator ──

def frustration_escalator(ctx: NLPContext) -> NLPContext:
    """L11: Track frustration level across conversation turns (0-5)."""
    level = 0

    # Base from emotion
    if ctx.emotion == "frustrated":
        level = 3
    elif ctx.emotion == "confused":
        level = 1

    # Escalate based on sentiment
    if ctx.sentiment_score < 0.2:
        level += 1
    elif ctx.sentiment_score < 0.1:
        level += 2

    # Multi-turn escalation
    frustrated_turns = sum(
        1 for m in ctx.conversation_history
        if m.get("role") == "user" and any(
            w in m.get("content", "").lower()
            for w in ["不懂", "不行", "失敗", "stuck", "frustrated"]
        )
    )
    level += min(frustrated_turns, 2)

    ctx.frustration_level = min(level, 5)
    return ctx


# ── L12: Confidence Estimator ──

CONFIDENCE_HIGH = [
    r"我覺得", r"我認為", r"應該是", r"我確定", r"我知道",
    r"i think", r"i believe", r"i'm sure", r"definitely",
]
CONFIDENCE_LOW = [
    r"不確定", r"可能", r"也許", r"不知道", r"大概",
    r"not sure", r"maybe", r"perhaps", r"i guess",
]


def confidence_estimator(ctx: NLPContext) -> NLPContext:
    """L12: Estimate student confidence level (0-1)."""
    text_lower = ctx.user_message.lower()

    high = sum(1 for p in CONFIDENCE_HIGH if re.search(p, text_lower))
    low = sum(1 for p in CONFIDENCE_LOW if re.search(p, text_lower))

    # Combine with sentiment
    base = ctx.sentiment_score * 0.3
    if high > low:
        ctx.confidence_level = min(base + 0.4 + high * 0.1, 1.0)
    elif low > high:
        ctx.confidence_level = max(base + 0.1 - low * 0.1, 0.0)
    else:
        ctx.confidence_level = 0.5

    return ctx


# ── L13: Urgency Detector (enhanced from existing) ──

URGENCY_HIGH = [
    r"急", r"趕", r"明天", r"今天", r"馬上", r"快", r"來不及", r"deadline",
    r"due", r"urgent", r"asap", r"tonight", r"tomorrow", r"趕快", r"拜託",
]
URGENCY_LOW = [
    r"順便", r"閒聊", r"好奇而已", r"隨便問", r"有空",
    r"just wondering", r"no rush", r"by the way", r"whenever",
]


def urgency_detector(ctx: NLPContext) -> NLPContext:
    """L13: Detect urgency level."""
    text_lower = ctx.user_message.lower()
    if any(re.search(p, text_lower) for p in URGENCY_HIGH):
        ctx.urgency = "high"
    elif any(re.search(p, text_lower) for p in URGENCY_LOW):
        ctx.urgency = "low"
    else:
        ctx.urgency = "normal"
    return ctx


# ── L14: Politeness Detector ──

POLITE_SIGNALS = [
    r"請", r"麻煩", r"謝謝", r"感謝", r"不好意思", r"打擾",
    r"please", r"thank", r"sorry", r"appreciate", r"kindly",
]


def politeness_detector(ctx: NLPContext) -> NLPContext:
    """L14: Detect politeness level (0-1) using SnowNLP + patterns."""
    text_lower = ctx.user_message.lower()

    polite_count = sum(1 for p in POLITE_SIGNALS if re.search(p, text_lower))
    pattern_score = min(polite_count * 0.15, 0.5)

    # SnowNLP sentiment as proxy (positive sentiment correlates with politeness)
    sentiment_component = ctx.sentiment_score * 0.3

    ctx.politeness_score = min(0.3 + pattern_score + sentiment_component, 1.0)
    return ctx
```

**Step 2: Verify import**

Run: `cd platform/backend && python -c "from app.nlp.understanding import sub_intent_detector, sentiment_scorer, frustration_escalator, confidence_estimator, urgency_detector, politeness_detector; print('6 layers OK')"`
Expected: `6 layers OK`

---

### Task C5: Create Group C — Student Level layers (L15-20)

**Files:**
- Create: `platform/backend/app/nlp/student_level.py`

**Step 1: Write the 6 student level layers**

```python
"""Group C: Student Level Assessment Layers (L15-20).

L15 DifficultyAssessor — textstat + jieba (enhanced from existing)
L16 VocabularyLevelScorer — jieba + domain term bank
L17 TechnicalFluencyScorer — POS analysis
L18 LearningStyleDetector — pattern-based
L19 MisconceptionDetector — rapidfuzz + misconception bank
L20 KnowledgeGapDetector — concept coverage analysis
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_textstat = None
_rapidfuzz = None


def _get_textstat():
    global _textstat
    if _textstat is None:
        import textstat
        _textstat = textstat
    return _textstat


def _get_rapidfuzz():
    global _rapidfuzz
    if _rapidfuzz is None:
        from rapidfuzz import fuzz
        _rapidfuzz = fuzz
    return _rapidfuzz


# ── Domain term bank ──

BEGINNER_TERMS = {"變數", "迴圈", "函數", "列表", "字典", "variable", "loop", "function", "list", "print"}

INTERMEDIATE_TERMS = {
    "過擬合", "欠擬合", "交叉驗證", "損失函數", "梯度下降", "學習率",
    "決策邊界", "特徵工程", "標準化", "SVM", "隨機森林",
    "overfitting", "underfitting", "cross-validation", "loss function",
    "gradient descent", "learning rate", "decision boundary", "random forest",
}

ADVANCED_TERMS = {
    "反向傳播", "注意力機制", "Transformer", "遷移學習", "批次正規化",
    "SHAP", "Grad-CAM", "嵌入空間", "自注意力", "位置編碼",
    "backpropagation", "attention mechanism", "transfer learning",
    "batch normalization", "positional encoding", "ablation", "SOTA",
}

# ── Common misconceptions in ML/DL ──

MISCONCEPTIONS = {
    "accuracy 越高越好": "在類別不平衡時，accuracy 可能誤導。應搭配 F1、AUC 等指標。",
    "更多特徵一定更好": "過多特徵可能導致維度災難和過擬合，需做特徵選擇。",
    "深度學習一定比傳統ML好": "資料少或結構化資料上，傳統 ML（如 XGBoost）常常更好。",
    "訓練損失越低模型越好": "訓練損失過低可能是過擬合，需看驗證損失。",
    "學習率越小越好": "學習率太小會收斂極慢或卡在局部最小值。",
    "batch size 越大越好": "過大的 batch size 可能導致泛化能力下降。",
    "dropout 永遠有幫助": "在資料量充足且模型不過擬合時，dropout 可能降低效能。",
    "CNN 只能用在影像": "CNN 也可用於文本分類、時序資料等一維序列。",
    "RNN 已經被淘汰": "RNN/LSTM 在某些小資料序列任務上仍有優勢。",
    "正則化就是 L2": "正則化包含 L1、L2、Dropout、Early Stopping、資料增強等多種技術。",
}

# ── Week-concept mapping for knowledge gap detection ──

WEEK_CONCEPTS = {
    1: ["Python", "NumPy", "Pandas", "Jupyter"],
    2: ["EDA", "Matplotlib", "Seaborn", "Plotly", "散佈圖", "直方圖"],
    3: ["監督式學習", "訓練集", "測試集", "交叉驗證", "過擬合", "欠擬合"],
    4: ["線性回歸", "損失函數", "MSE", "梯度下降", "學習率"],
    5: ["邏輯迴歸", "Sigmoid", "決策邊界", "ROC", "AUC", "F1"],
    6: ["SVM", "核方法", "RBF", "間隔最大化", "支撐向量"],
    7: ["決策樹", "隨機森林", "GBDT", "Bagging", "Boosting"],
    8: ["特徵重要度", "SHAP", "排列重要度", "可解釋性"],
    9: ["特徵工程", "Pipeline", "One-Hot", "StandardScaler"],
    10: ["超參數", "GridSearch", "RandomSearch", "學習曲線"],
    11: ["神經網路", "激活函數", "ReLU", "Dropout", "BatchNorm"],
    12: ["CNN", "卷積", "池化", "特徵圖", "Grad-CAM"],
    13: ["RNN", "LSTM", "GRU", "Transformer", "注意力機制"],
    14: ["學習率排程", "早停", "資料增強", "訓練曲線"],
    15: ["混淆矩陣", "公平性", "偏誤", "穩健性"],
    16: ["MLOps", "MLflow", "模型版本", "資料漂移"],
    17: ["LLM", "嵌入", "RAG", "提示工程"],
    18: ["專題報告", "可重現性", "倫理"],
}


# ── L15: Difficulty Assessor (enhanced) ──

def difficulty_assessor(ctx: NLPContext) -> NLPContext:
    """L15: Assess student difficulty level from vocabulary and text complexity."""
    text_lower = ctx.user_message.lower()

    # Count domain terms by level
    beginner_count = sum(1 for t in BEGINNER_TERMS if t.lower() in text_lower)
    intermediate_count = sum(1 for t in INTERMEDIATE_TERMS if t.lower() in text_lower)
    advanced_count = sum(1 for t in ADVANCED_TERMS if t.lower() in text_lower)

    ctx.uses_technical_terms = (intermediate_count + advanced_count) > 0

    # Determine level
    if advanced_count >= 2 or (advanced_count >= 1 and intermediate_count >= 2):
        ctx.student_level = "advanced"
    elif intermediate_count >= 2 or (intermediate_count >= 1 and beginner_count >= 1):
        ctx.student_level = "intermediate"
    else:
        ctx.student_level = "beginner"

    # Question complexity
    msg_len = len(ctx.user_message)
    q_marks = ctx.user_message.count("？") + ctx.user_message.count("?")

    if q_marks > 1 or ctx.has_code or msg_len > 200:
        ctx.question_complexity = "complex"
    elif msg_len > 80 or (intermediate_count + advanced_count) >= 2:
        ctx.question_complexity = "moderate"
    else:
        ctx.question_complexity = "simple"

    return ctx


# ── L16: Vocabulary Level Scorer ──

def vocabulary_level_scorer(ctx: NLPContext) -> NLPContext:
    """L16: Score vocabulary richness based on domain term usage."""
    text_lower = ctx.user_message.lower()
    all_terms = BEGINNER_TERMS | INTERMEDIATE_TERMS | ADVANCED_TERMS
    used = [t for t in all_terms if t.lower() in text_lower]

    if not used:
        ctx.vocabulary_score = 0.0
        return ctx

    # Weight: advanced=3, intermediate=2, beginner=1
    score = 0
    for t in used:
        if t in ADVANCED_TERMS or t.lower() in {x.lower() for x in ADVANCED_TERMS}:
            score += 3
        elif t in INTERMEDIATE_TERMS or t.lower() in {x.lower() for x in INTERMEDIATE_TERMS}:
            score += 2
        else:
            score += 1

    # Normalize to 0-1 (max realistic score ~15)
    ctx.vocabulary_score = min(score / 15.0, 1.0)
    return ctx


# ── L17: Technical Fluency Scorer ──

def technical_fluency_scorer(ctx: NLPContext) -> NLPContext:
    """L17: Score technical expression fluency using POS tags."""
    if not ctx.pos_tags:
        ctx.technical_fluency = 0.0
        return ctx

    # Technical POS: nouns (n, eng), verbs related to tech
    tech_pos = {"eng", "n", "nz", "nr", "vn"}
    tech_count = sum(1 for _, flag in ctx.pos_tags if flag in tech_pos)
    total = len(ctx.pos_tags)

    if total == 0:
        ctx.technical_fluency = 0.0
    else:
        ctx.technical_fluency = min(tech_count / total, 1.0)

    return ctx


# ── L18: Learning Style Detector ──

VISUAL_SIGNALS = ["圖", "畫", "視覺", "看", "顯示", "plot", "chart", "graph", "visual", "show"]
TEXTUAL_SIGNALS = ["解釋", "說明", "描述", "文字", "explain", "describe", "text", "read"]
PRACTICAL_SIGNALS = ["實作", "練習", "跑", "程式", "code", "implement", "try", "run", "practice"]


def learning_style_detector(ctx: NLPContext) -> NLPContext:
    """L18: Detect preferred learning style (visual/textual/practical/balanced)."""
    text_lower = ctx.user_message.lower()

    v = sum(1 for s in VISUAL_SIGNALS if s in text_lower)
    t = sum(1 for s in TEXTUAL_SIGNALS if s in text_lower)
    p = sum(1 for s in PRACTICAL_SIGNALS if s in text_lower)

    total = v + t + p
    if total == 0:
        ctx.learning_style = "balanced"
    elif v > t and v > p:
        ctx.learning_style = "visual"
    elif t > v and t > p:
        ctx.learning_style = "textual"
    elif p > v and p > t:
        ctx.learning_style = "practical"
    else:
        ctx.learning_style = "balanced"

    return ctx


# ── L19: Misconception Detector ──

def misconception_detector(ctx: NLPContext) -> NLPContext:
    """L19: Detect common ML/DL misconceptions using fuzzy matching."""
    fuzz = _get_rapidfuzz()
    text = ctx.user_message
    ctx.misconceptions = []

    for trigger, correction in MISCONCEPTIONS.items():
        score = fuzz.partial_ratio(trigger, text)
        if score > 75:
            ctx.misconceptions.append(f"⚠️ 常見迷思：「{trigger}」— {correction}")

    return ctx


# ── L20: Knowledge Gap Detector ──

def knowledge_gap_detector(ctx: NLPContext) -> NLPContext:
    """L20: Detect knowledge gaps by comparing question with prerequisite concepts."""
    current_week = ctx.week
    text_lower = ctx.user_message.lower()

    # Check if student is asking about concepts from earlier weeks
    gaps = []
    for w in range(1, current_week):
        concepts = WEEK_CONCEPTS.get(w, [])
        for concept in concepts:
            if concept.lower() in text_lower:
                # If asking basic questions about prerequisite concepts, might be a gap
                if ctx.intent in ("definition", "how", "prerequisite") and ctx.student_level == "beginner":
                    gaps.append(f"第{w}週：{concept}")

    ctx.knowledge_gaps = gaps[:3]  # Top 3 gaps

    # Track known concepts (mentioned with confidence)
    current_concepts = WEEK_CONCEPTS.get(current_week, [])
    known = [c for c in current_concepts if c.lower() in text_lower and ctx.confidence_level > 0.6]
    ctx.known_concepts = known

    # Unknown = current week concepts NOT mentioned
    ctx.unknown_concepts = [c for c in current_concepts if c not in known]

    return ctx
```

**Step 2: Verify import**

Run: `cd platform/backend && python -c "from app.nlp.student_level import difficulty_assessor, vocabulary_level_scorer, technical_fluency_scorer, learning_style_detector, misconception_detector, knowledge_gap_detector; print('6 layers OK')"`
Expected: `6 layers OK`

---

### Task C6: Create Group D — Content Analysis layers (L21-27)

**Files:**
- Create: `platform/backend/app/nlp/content_analysis.py`

**Step 1: Write the 7 content analysis layers**

```python
"""Group D: Content Analysis Layers (L21-27).

L21 KeywordExtractor — jieba.analyse TF-IDF + TextRank
L22 DomainConceptMatcher — rapidfuzz + concept map
L23 NamedEntityRecognizer — jieba + custom dict
L24 CodeBlockDetector — regex + heuristics
L25 MathExpressionDetector — regex
L26 QuestionQualityScorer — multi-feature
L27 ReadabilityScorer — textstat
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_jieba_analyse = None
_rapidfuzz = None
_textstat = None


def _get_jieba_analyse():
    global _jieba_analyse
    if _jieba_analyse is None:
        import jieba.analyse
        _jieba_analyse = jieba.analyse
    return _jieba_analyse


def _get_rapidfuzz():
    global _rapidfuzz
    if _rapidfuzz is None:
        from rapidfuzz import fuzz
        _rapidfuzz = fuzz
    return _rapidfuzz


def _get_textstat():
    global _textstat
    if _textstat is None:
        import textstat
        _textstat = textstat
    return _textstat


# ── Domain concept map (expanded from existing topic.py) ──

CONCEPT_MAP = {
    "梯度下降": "梯度下降 (Gradient Descent)", "gradient descent": "梯度下降 (Gradient Descent)",
    "學習率": "學習率 (Learning Rate)", "learning rate": "學習率 (Learning Rate)",
    "損失函數": "損失函數 (Loss Function)", "loss function": "損失函數 (Loss Function)",
    "線性回歸": "線性回歸 (Linear Regression)", "linear regression": "線性回歸 (Linear Regression)",
    "決策邊界": "決策邊界 (Decision Boundary)", "decision boundary": "決策邊界 (Decision Boundary)",
    "邏輯迴歸": "邏輯迴歸 (Logistic Regression)", "logistic regression": "邏輯迴歸 (Logistic Regression)",
    "svm": "支撐向量機 (SVM)", "支撐向量機": "支撐向量機 (SVM)",
    "決策樹": "決策樹 (Decision Tree)", "decision tree": "決策樹 (Decision Tree)",
    "隨機森林": "隨機森林 (Random Forest)", "random forest": "隨機森林 (Random Forest)",
    "shap": "SHAP 值", "過擬合": "過擬合 (Overfitting)", "overfitting": "過擬合 (Overfitting)",
    "欠擬合": "欠擬合 (Underfitting)", "underfitting": "欠擬合 (Underfitting)",
    "交叉驗證": "交叉驗證 (Cross-Validation)", "cross-validation": "交叉驗證 (Cross-Validation)",
    "超參數": "超參數調校 (Hyperparameter Tuning)", "hyperparameter": "超參數調校 (Hyperparameter Tuning)",
    "激活函數": "激活函數 (Activation Function)", "activation function": "激活函數 (Activation Function)",
    "relu": "ReLU 激活函數", "sigmoid": "Sigmoid 函數",
    "神經網路": "神經網路 (Neural Network)", "neural network": "神經網路 (Neural Network)",
    "cnn": "卷積神經網路 (CNN)", "rnn": "循環神經網路 (RNN)",
    "lstm": "LSTM", "transformer": "Transformer",
    "attention": "注意力機制 (Attention)", "注意力": "注意力機制 (Attention)",
    "dropout": "Dropout", "batch normalization": "批次正規化 (BatchNorm)",
    "早停": "早停法 (Early Stopping)", "early stopping": "早停法 (Early Stopping)",
    "特徵工程": "特徵工程 (Feature Engineering)", "feature engineering": "特徵工程 (Feature Engineering)",
    "rag": "檢索增強生成 (RAG)", "embedding": "嵌入 (Embedding)", "嵌入": "嵌入 (Embedding)",
    "mlops": "MLOps", "mlflow": "MLflow",
    "bagging": "Bagging", "boosting": "Boosting", "gbdt": "梯度提升樹 (GBDT)",
    "正則化": "正則化 (Regularization)", "regularization": "正則化 (Regularization)",
}

# ── Known entities (packages, models, algorithms) ──

ENTITY_PACKAGES = {
    "scikit-learn", "sklearn", "numpy", "pandas", "matplotlib", "seaborn",
    "plotly", "pytorch", "tensorflow", "keras", "xgboost", "lightgbm",
    "shap", "mlflow", "huggingface", "transformers",
}

ENTITY_ALGORITHMS = {
    "KNN", "SVM", "GBDT", "XGBoost", "LightGBM", "AdaBoost",
    "ResNet", "VGG", "LeNet", "BERT", "GPT", "GAN", "VAE",
}


# ── L21: Keyword Extractor ──

def keyword_extractor(ctx: NLPContext) -> NLPContext:
    """L21: Extract keywords using jieba TF-IDF + TextRank."""
    analyse = _get_jieba_analyse()

    tfidf_kw = analyse.extract_tags(ctx.user_message, topK=8, withWeight=True)
    textrank_kw = analyse.textrank(ctx.user_message, topK=8, withWeight=True)

    # Merge and deduplicate, keep highest weight
    merged = {}
    for kw, w in tfidf_kw:
        merged[kw] = max(merged.get(kw, 0), w)
    for kw, w in textrank_kw:
        merged[kw] = max(merged.get(kw, 0), w * 0.8)

    sorted_kw = sorted(merged.items(), key=lambda x: -x[1])
    ctx.keywords = [kw for kw, _ in sorted_kw[:10]]
    ctx.keyword_scores = [(kw, round(w, 4)) for kw, w in sorted_kw[:10]]
    return ctx


# ── L22: Domain Concept Matcher ──

def domain_concept_matcher(ctx: NLPContext) -> NLPContext:
    """L22: Match keywords to domain concepts using exact + fuzzy matching."""
    fuzz = _get_rapidfuzz()
    text_lower = ctx.user_message.lower()
    concepts = []
    seen = set()

    # Exact match first
    for trigger, concept in CONCEPT_MAP.items():
        if trigger in text_lower and concept not in seen:
            concepts.append(concept)
            seen.add(concept)

    # Fuzzy match on keywords
    for kw in ctx.keywords[:5]:
        for trigger, concept in CONCEPT_MAP.items():
            if concept in seen:
                continue
            if fuzz.ratio(kw.lower(), trigger) > 80:
                concepts.append(concept)
                seen.add(concept)

    ctx.domain_concepts = concepts
    return ctx


# ── L23: Named Entity Recognizer ──

def named_entity_recognizer(ctx: NLPContext) -> NLPContext:
    """L23: Recognize ML/DL named entities (packages, algorithms, metrics)."""
    text_lower = ctx.user_message.lower()
    entities = []

    for pkg in ENTITY_PACKAGES:
        if pkg.lower() in text_lower:
            entities.append({"text": pkg, "type": "PACKAGE"})

    for algo in ENTITY_ALGORITHMS:
        if algo.lower() in text_lower:
            entities.append({"text": algo, "type": "ALGORITHM"})

    # Detect metric names
    metrics = re.findall(r'\b(accuracy|precision|recall|f1|auc|rmse|mae|mse|r2)\b', text_lower)
    for m in metrics:
        entities.append({"text": m.upper(), "type": "METRIC"})

    ctx.named_entities = entities
    return ctx


# ── L24: Code Block Detector ──

CODE_PATTERNS = [
    r'```[\s\S]*?```',
    r'^\s*(import |from .+ import |def |class |print\(|for .+ in |if __name__)',
    r'^\s*\w+\s*=\s*\w+\.\w+\(',
    r'model\.(fit|predict|transform|score)\(',
    r'plt\.\w+\(',
    r'pd\.(read_csv|DataFrame)',
    r'np\.\w+\(',
]


def code_block_detector(ctx: NLPContext) -> NLPContext:
    """L24: Detect code blocks and identify programming language."""
    text = ctx.user_message

    # Fenced code blocks
    fenced = re.findall(r'```(\w*)\n?([\s\S]*?)```', text)
    if fenced:
        ctx.has_code = True
        ctx.code_blocks = [code for _, code in fenced]
        ctx.code_language = fenced[0][0] or "python"
        return ctx

    # Inline code patterns
    for pattern in CODE_PATTERNS[1:]:
        if re.search(pattern, text, re.MULTILINE):
            ctx.has_code = True
            ctx.code_language = "python"
            return ctx

    ctx.has_code = False
    return ctx


# ── L25: Math Expression Detector ──

MATH_PATTERNS = [
    r'\$\$.+?\$\$',
    r'\$.+?\$',
    r'\\frac\{', r'\\sum', r'\\nabla', r'\\partial',
    r'∑|∏|∫|∂|∇|∈|∉|⊂|⊃|∀|∃',
    r'\b[a-z]\s*=\s*[a-z]\s*[\+\-\*/]\s*[a-z]\b',
    r'argmax|argmin|log\s*\(|exp\s*\(',
]


def math_expression_detector(ctx: NLPContext) -> NLPContext:
    """L25: Detect mathematical expressions in the message."""
    text = ctx.user_message
    expressions = []

    # LaTeX blocks
    for expr in re.findall(r'\$\$(.+?)\$\$', text, re.DOTALL):
        expressions.append(expr.strip())
    for expr in re.findall(r'\$(.+?)\$', text):
        expressions.append(expr.strip())

    if expressions:
        ctx.has_math = True
        ctx.math_expressions = expressions
        return ctx

    # Other math indicators
    for pattern in MATH_PATTERNS[4:]:
        if re.search(pattern, text):
            ctx.has_math = True
            return ctx

    ctx.has_math = False
    return ctx


# ── L26: Question Quality Scorer ──

def question_quality_scorer(ctx: NLPContext) -> NLPContext:
    """L26: Score question quality (0-1) to guide students to ask better questions."""
    score = 0.3  # Base score
    feedback = []

    # Has specific context
    if ctx.keywords and len(ctx.keywords) >= 2:
        score += 0.1
    else:
        feedback.append("可以加入更具體的關鍵詞")

    # References week or topic
    if re.search(r'第\s*\d+\s*週|week\s*\d+', ctx.user_message, re.IGNORECASE):
        score += 0.1

    # Shows prior attempt
    if any(w in ctx.user_message.lower() for w in ["我試了", "我嘗試", "i tried", "my attempt"]):
        score += 0.15
    else:
        feedback.append("描述你已經嘗試過什麼會更有幫助")

    # Includes error message or output
    if ctx.has_code or any(w in ctx.user_message.lower() for w in ["error", "錯誤", "output"]):
        score += 0.1

    # Appropriate length (not too short)
    if len(ctx.user_message) > 30:
        score += 0.1
    else:
        feedback.append("問題可以再描述得更詳細一些")

    # Domain concepts identified
    if ctx.domain_concepts:
        score += 0.1

    ctx.question_quality = min(score, 1.0)
    ctx.quality_feedback = "；".join(feedback) if feedback else ""
    return ctx


# ── L27: Readability Scorer ──

def readability_scorer(ctx: NLPContext) -> NLPContext:
    """L27: Score text readability/complexity using textstat."""
    ts = _get_textstat()
    try:
        # textstat works best on English; for Chinese, use character count heuristic
        if ctx.language == "en":
            ctx.readability_score = ts.flesch_reading_ease(ctx.user_message) / 100.0
        else:
            # Simple heuristic for Chinese: sentence length and term complexity
            avg_sent_len = len(ctx.user_message) / max(len(ctx.sentences), 1)
            ctx.readability_score = min(avg_sent_len / 50.0, 1.0)
    except Exception:
        ctx.readability_score = 0.5
    return ctx
```

**Step 2: Verify import**

Run: `cd platform/backend && python -c "from app.nlp.content_analysis import keyword_extractor, domain_concept_matcher, named_entity_recognizer, code_block_detector, math_expression_detector, question_quality_scorer, readability_scorer; print('7 layers OK')"`
Expected: `7 layers OK`

---

### Task C7: Create Group E — Context & Memory layers (L28-32)

**Files:**
- Create: `platform/backend/app/nlp/context_memory.py`

**Step 1: Write the 5 context layers**

```python
"""Group E: Context & Memory Layers (L28-32).

L28 ConversationTracker — enhanced from existing
L29 TopicContinuityDetector — sentence-transformers
L30 HintLadderManager — state machine
L31 SessionSummarizer — snownlp + jieba
L32 KnowledgeStateTracker — concept tracking
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_sentence_model = None


def _get_sentence_model():
    """Lazy-load sentence-transformers model."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")
            logger.info("Sentence-transformers model loaded")
        except Exception as e:
            logger.warning("Could not load sentence-transformers: %s", e)
            _sentence_model = False  # Mark as failed
    return _sentence_model if _sentence_model is not False else None


# ── L28: Conversation Tracker (enhanced) ──

FOLLOWUP_SIGNALS = [
    r"^那", r"^還有", r"^另外", r"^接著", r"^然後呢", r"^繼續",
    r"^所以", r"^也就是", r"^換句話說", r"^你剛才說",
    r"^上面", r"^前面", r"^剛剛", r"^這個",
    r"^then", r"^also", r"^and ", r"^what about", r"^so ",
    r"^following up", r"^you (said|mentioned)",
]


def conversation_tracker(ctx: NLPContext) -> NLPContext:
    """L28: Track multi-turn conversation state."""
    history = ctx.conversation_history
    ctx.turn_count = sum(1 for m in history if m.get("role") == "user")

    # Detect follow-up
    text = ctx.user_message
    ctx.is_followup = ctx.turn_count > 0 and any(
        re.search(p, text, re.IGNORECASE) for p in FOLLOWUP_SIGNALS
    )

    if ctx.turn_count > 0:
        ctx.previous_intent = ctx.intent

    return ctx


# ── L29: Topic Continuity Detector ──

def topic_continuity_detector(ctx: NLPContext) -> NLPContext:
    """L29: Detect if current question continues the previous topic using embeddings."""
    if ctx.turn_count == 0:
        ctx.topic_continuity = 0.0
        return ctx

    model = _get_sentence_model()
    if model is None:
        # Fallback: simple keyword overlap
        prev_messages = [m["content"] for m in ctx.conversation_history if m.get("role") == "user"]
        if prev_messages:
            prev_words = set(prev_messages[-1].lower().split())
            curr_words = set(ctx.user_message.lower().split())
            overlap = len(prev_words & curr_words)
            ctx.topic_continuity = min(overlap / max(len(curr_words), 1), 1.0)
        return ctx

    try:
        prev_messages = [m["content"] for m in ctx.conversation_history if m.get("role") == "user"]
        if prev_messages:
            embeddings = model.encode([prev_messages[-1], ctx.user_message])
            from sklearn.metrics.pairwise import cosine_similarity
            sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0])
            ctx.topic_continuity = max(sim, 0.0)
            if sim > 0.6:
                ctx.continued_topic = ctx.topic
    except Exception:
        ctx.topic_continuity = 0.0

    return ctx


# ── L30: Hint Ladder Manager ──

PROGRESS_SIGNALS = [
    r"我試了", r"我嘗試", r"我寫了", r"結果是", r"我得到",
    r"但是", r"不過", r"可是", r"出現了", r"變成",
    r"i tried", r"i got", r"my result", r"but then", r"however",
]

STUCK_SIGNALS = [
    r"還是不懂", r"還是不行", r"一樣的錯", r"又失敗",
    r"看不出", r"不知道哪裡", r"完全沒有頭緒",
    r"still", r"again", r"same error", r"doesn't work",
]


def hint_ladder_manager(ctx: NLPContext) -> NLPContext:
    """L30: Manage Hint Ladder progression (1-4)."""
    text = ctx.user_message
    has_progress = any(re.search(p, text, re.IGNORECASE) for p in PROGRESS_SIGNALS)
    still_stuck = any(re.search(p, text, re.IGNORECASE) for p in STUCK_SIGNALS)

    if ctx.turn_count == 0:
        ctx.hint_level = 1
    elif still_stuck and ctx.turn_count >= 3:
        ctx.hint_level = 4
    elif still_stuck:
        ctx.hint_level = min(ctx.hint_level + 1, 4)
    elif has_progress:
        ctx.hint_level = min(ctx.hint_level + 1, 3)
    elif ctx.turn_count >= 2:
        ctx.hint_level = min(ctx.turn_count, 3)
    else:
        ctx.hint_level = 1

    # Frustration-based override
    if ctx.frustration_level >= 4:
        ctx.hint_level = max(ctx.hint_level, 3)

    return ctx


# ── L31: Session Summarizer ──

def session_summarizer(ctx: NLPContext) -> NLPContext:
    """L31: Generate a brief summary of the conversation so far."""
    if ctx.turn_count < 2:
        ctx.session_summary = ""
        return ctx

    user_msgs = [m["content"] for m in ctx.conversation_history if m.get("role") == "user"]
    if not user_msgs:
        return ctx

    try:
        from snownlp import SnowNLP
        combined = "。".join(user_msgs[-3:])  # Last 3 messages
        s = SnowNLP(combined)
        summaries = s.summary(3)
        ctx.session_summary = "；".join(summaries) if summaries else ""
    except Exception:
        # Fallback: just use keywords
        ctx.session_summary = "、".join(ctx.keywords[:5]) if ctx.keywords else ""

    return ctx


# ── L32: Knowledge State Tracker ──

def knowledge_state_tracker(ctx: NLPContext) -> NLPContext:
    """L32: Track which concepts the student has demonstrated understanding of."""
    # Analyze conversation history for confirmed understanding
    for msg in ctx.conversation_history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "").lower()
        # Signals of understanding
        if any(w in content for w in ["我懂了", "了解了", "原來", "i see", "i understand", "got it", "makes sense"]):
            # Find which concepts were in the previous assistant message
            idx = ctx.conversation_history.index(msg)
            if idx > 0:
                prev = ctx.conversation_history[idx - 1].get("content", "").lower()
                for concept in ctx.domain_concepts:
                    if concept.split("(")[0].strip().lower() in prev:
                        if concept not in ctx.known_concepts:
                            ctx.known_concepts.append(concept)

    return ctx
```

**Step 2: Verify import**

Run: `cd platform/backend && python -c "from app.nlp.context_memory import conversation_tracker, topic_continuity_detector, hint_ladder_manager, session_summarizer, knowledge_state_tracker; print('5 layers OK')"`
Expected: `5 layers OK`

---

### Task C8: Create Group F — Retrieval layers (L33-36)

**Files:**
- Create: `platform/backend/app/nlp/retrieval.py`

**Step 1: Write the 4 retrieval layers**

```python
"""Group F: Retrieval Enhancement Layers (L33-36).

L33 QueryExpander — jieba synonyms + sentence-transformers
L34 RAGRetriever — existing FTS5 (delegates to rag.retriever)
L35 SemanticReranker — sentence-transformers cosine similarity
L36 CrossWeekLinker — concept-based cross-week linking
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")
        except Exception:
            _sentence_model = False
    return _sentence_model if _sentence_model is not False else None


# ── Simple synonym map for query expansion ──

SYNONYMS = {
    "準確率": ["accuracy", "精確度"],
    "損失": ["loss", "損失函數", "cost"],
    "梯度下降": ["gradient descent", "GD"],
    "學習率": ["learning rate", "lr"],
    "過擬合": ["overfitting", "過度擬合"],
    "欠擬合": ["underfitting", "擬合不足"],
    "隨機森林": ["random forest", "RF"],
    "決策樹": ["decision tree"],
    "激活函數": ["activation function"],
    "反向傳播": ["backpropagation", "back propagation"],
    "注意力": ["attention", "self-attention"],
    "卷積": ["convolution", "conv"],
}

# ── Week concept links for cross-week navigation ──

CONCEPT_WEEK_MAP = {
    "梯度下降": [4, 11], "損失函數": [4, 14], "學習率": [4, 10, 14],
    "過擬合": [3, 7, 10, 14], "欠擬合": [3, 10],
    "決策邊界": [5, 6], "SVM": [5, 6], "特徵工程": [8, 9],
    "正則化": [11, 14], "CNN": [12], "RNN": [13], "Transformer": [13, 17],
    "SHAP": [8], "公平性": [15], "MLOps": [16], "嵌入": [17],
    "交叉驗證": [3, 10], "超參數": [10],
}


# ── L33: Query Expander ──

def query_expander(ctx: NLPContext) -> NLPContext:
    """L33: Expand query with synonyms for better retrieval."""
    expanded_parts = [ctx.user_message]

    # Add synonyms for detected keywords
    for kw in ctx.keywords[:5]:
        kw_lower = kw.lower()
        for trigger, syns in SYNONYMS.items():
            if trigger in kw_lower or kw_lower in trigger.lower():
                expanded_parts.extend(syns[:2])

    ctx.expanded_query = " ".join(expanded_parts)
    return ctx


# ── L34: RAG Retriever ──

def rag_retriever(ctx: NLPContext) -> NLPContext:
    """L34: Retrieve context using FTS5 (delegates to existing retriever)."""
    from app.rag.retriever import retrieve_context

    # Use expanded query if available
    query = ctx.expanded_query or ctx.user_message
    context = retrieve_context(query, week=ctx.week, top_k=5)

    # If sparse, try with just keywords
    if len(context) < 200 and ctx.keywords:
        kw_query = " ".join(ctx.keywords[:5])
        extra = retrieve_context(kw_query, week=ctx.week, top_k=3)
        if extra:
            context = context + "\n\n---\n\n" + extra if context else extra

    ctx.rag_context = context

    # Track sources
    if context:
        sources = re.findall(r"\[來源：([^\]]+)\]", context)
        ctx.rag_sources = sources

    return ctx


# ── L35: Semantic Reranker ──

def semantic_reranker(ctx: NLPContext) -> NLPContext:
    """L35: Re-rank RAG results by semantic similarity (if sentence-transformers available)."""
    model = _get_sentence_model()
    if model is None or not ctx.rag_context:
        return ctx

    try:
        # Split context into chunks
        chunks = ctx.rag_context.split("\n\n---\n\n")
        if len(chunks) <= 1:
            return ctx

        # Encode query and chunks
        texts = [ctx.user_message] + chunks
        embeddings = model.encode(texts)

        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = embeddings[0:1]
        chunk_embs = embeddings[1:]
        sims = cosine_similarity(query_emb, chunk_embs)[0]

        # Re-order by similarity
        ranked = sorted(zip(chunks, sims), key=lambda x: -x[1])
        ctx.rag_context = "\n\n---\n\n".join([c for c, _ in ranked])
        ctx.reranked_results = [{"chunk": c[:100], "score": round(float(s), 3)} for c, s in ranked]
    except Exception as e:
        logger.warning("Semantic reranking failed: %s", e)

    return ctx


# ── L36: Cross-Week Linker ──

def cross_week_linker(ctx: NLPContext) -> NLPContext:
    """L36: Find related content in other weeks."""
    links = []
    current_week = ctx.week

    for concept in ctx.domain_concepts:
        # Extract the Chinese part of the concept
        clean = concept.split("(")[0].strip()
        related_weeks = CONCEPT_WEEK_MAP.get(clean, [])
        for w in related_weeks:
            if w != current_week:
                links.append({"week": w, "concept": concept, "relation": "相關概念"})

    # Deduplicate by week
    seen_weeks = set()
    unique_links = []
    for link in links:
        if link["week"] not in seen_weeks:
            seen_weeks.add(link["week"])
            unique_links.append(link)

    ctx.cross_week_links = unique_links[:5]
    return ctx
```

**Step 2: Verify import**

Run: `cd platform/backend && python -c "from app.nlp.retrieval import query_expander, rag_retriever, semantic_reranker, cross_week_linker; print('4 layers OK')"`
Expected: `4 layers OK`

---

### Task C9: Create Group G — Response layers (L37-42)

**Files:**
- Create: `platform/backend/app/nlp/response_gen.py`

**Step 1: Write the 6 response generation layers**

```python
"""Group G: Response Generation Layers (L37-42).

L37 ResponseAssembler — enhanced adaptive template
L38 ComplexityAdjuster — adjust based on student level
L39 CitationInjector — add week/source references
L40 FollowUpGenerator — generate guiding questions
L41 EncouragementGenerator — emotion-aware encouragement
L42 ResponseCompletenessChecker — check response quality
"""

from .pipeline import NLPContext
from .response import (
    INTENT_TEMPLATES, EMOTION_PREFIX, LEVEL_GUIDANCE,
    HINT_LADDER, INTENT_GUIDANCE, NO_CONTEXT, HOMEWORK_GUARD,
    _concept_note,
)


# ── L37: Response Assembler (enhanced) ──

def response_assembler(ctx: NLPContext) -> NLPContext:
    """L37: Assemble response using all NLP layer outputs."""
    topic = "、".join(ctx.keywords[:3]) if ctx.keywords else ctx.user_message[:20]

    # No context found
    if not ctx.rag_context:
        ctx.response = NO_CONTEXT.format(
            keywords="、".join(ctx.keywords) if ctx.keywords else ctx.user_message[:30]
        )
        return ctx

    # Homework mode
    if ctx.is_homework:
        ctx.response = HOMEWORK_GUARD.format(context=ctx.rag_context)
        return ctx

    parts = []

    # 1. Emotion prefix
    prefix = EMOTION_PREFIX.get(ctx.emotion, "")
    if prefix:
        parts.append(prefix)

    # 2. Main content
    template = INTENT_TEMPLATES.get(ctx.intent, INTENT_TEMPLATES["general"])
    parts.append(template.format(topic=topic, context=ctx.rag_context))

    # 3. Misconception warnings
    if ctx.misconceptions:
        parts.append("\n\n---\n" + "\n".join(ctx.misconceptions[:2]))

    # 4. Domain concept note
    concept_note = _concept_note(ctx)
    if concept_note:
        parts.append(concept_note)

    # 5. Separator
    parts.append("\n---")

    # 6. Level guidance
    level_guides = LEVEL_GUIDANCE.get(ctx.student_level, LEVEL_GUIDANCE["beginner"])
    guide = level_guides.get(ctx.intent, level_guides.get("general", ""))
    if guide:
        parts.append(guide)

    # 7. Learning style hint
    if ctx.learning_style == "visual":
        parts.append("\n🎨 **建議：** 你偏好視覺化學習，試試平台上的互動視覺化工具！")
    elif ctx.learning_style == "practical":
        parts.append("\n💻 **建議：** 你偏好動手實作，建議先跑一遍 Notebook 再看講義。")

    # 8. Cross-week links
    if ctx.cross_week_links:
        links = ctx.cross_week_links[:3]
        link_text = "、".join([f"第{l['week']}週（{l['concept']}）" for l in links])
        parts.append(f"\n\n🔗 **相關週次：** {link_text}")

    ctx.response = "".join(parts)
    return ctx


# ── L38: Complexity Adjuster ──

def complexity_adjuster(ctx: NLPContext) -> NLPContext:
    """L38: Adjust response complexity based on student level."""
    if not ctx.response:
        return ctx

    if ctx.student_level == "beginner" and ctx.question_complexity == "simple":
        ctx.response_complexity = "simple"
        # Add simplified explanation marker
        if len(ctx.response) > 500:
            ctx.response += "\n\n📝 **簡單來說：** 上面的內容比較多，建議先讀標示為「初學者建議」的部分。"
    elif ctx.student_level == "advanced":
        ctx.response_complexity = "advanced"
    else:
        ctx.response_complexity = "moderate"

    return ctx


# ── L39: Citation Injector ──

def citation_injector(ctx: NLPContext) -> NLPContext:
    """L39: Add explicit references to curriculum materials."""
    if not ctx.rag_sources:
        return ctx

    unique_sources = list(dict.fromkeys(ctx.rag_sources))[:3]
    ctx.citations = unique_sources

    citation_text = "\n\n📖 **參考來源：**\n" + "\n".join([f"- {s}" for s in unique_sources])
    ctx.response += citation_text

    return ctx


# ── L40: Follow-Up Question Generator ──

FOLLOWUP_TEMPLATES = {
    "definition": ["你能用自己的話重新解釋這個概念嗎？", "你覺得這和{related}有什麼關係？"],
    "how": ["你有嘗試自己實作嗎？哪一步卡住了？", "你覺得每一步的目的是什麼？"],
    "why": ["如果不這樣做，你覺得會發生什麼？", "你能想到一個反例嗎？"],
    "debug": ["你能把完整的錯誤訊息貼出來嗎？", "你是在哪一行出錯的？"],
    "compare": ["在你目前的專案中，你會選擇哪一個？為什麼？"],
    "code": ["你能解釋這段程式碼每一行在做什麼嗎？"],
    "general": ["關於這個主題，你最想了解的是什麼？"],
}


def follow_up_generator(ctx: NLPContext) -> NLPContext:
    """L40: Generate follow-up questions to guide student thinking."""
    templates = FOLLOWUP_TEMPLATES.get(ctx.intent, FOLLOWUP_TEMPLATES["general"])

    # Pick template based on hint level
    idx = min(ctx.hint_level - 1, len(templates) - 1)
    question = templates[idx]

    # Fill in related concept if template has {related}
    if "{related}" in question and ctx.domain_concepts:
        question = question.format(related=ctx.domain_concepts[0])
    else:
        question = question.replace("{related}", "其他相關概念")

    ctx.follow_up_questions = [question]
    ctx.response += f"\n\n❓ **思考題：** {question}"

    return ctx


# ── L41: Encouragement Generator ──

ENCOURAGEMENTS = {
    "frustrated": [
        "別擔心，這個概念確實不容易，很多同學也花了不少時間才理解。",
        "你已經很努力了！遇到困難是進步的必經之路。",
        "一步一步來，不需要一次全部搞懂。",
    ],
    "confused": [
        "混淆是正常的！很多概念在初學時確實容易搞混。",
        "花時間釐清這些概念是值得的，之後會越來越清楚。",
    ],
    "curious": [
        "你的好奇心很棒！保持這種求知慾！",
        "問出好問題本身就是一種能力！",
    ],
    "confident": [
        "很高興你有自信！讓我們來驗證你的理解。",
    ],
}


def encouragement_generator(ctx: NLPContext) -> NLPContext:
    """L41: Add emotion-aware encouragement."""
    msgs = ENCOURAGEMENTS.get(ctx.emotion, [])
    if msgs:
        import random
        msg = random.choice(msgs)
        ctx.encouragement = msg
        if ctx.emotion in ("frustrated", "confused"):
            ctx.response += f"\n\n💪 {msg}"

    return ctx


# ── L42: Response Completeness Checker ──

def response_completeness_checker(ctx: NLPContext) -> NLPContext:
    """L42: Check if the response addresses all detected sub-questions."""
    score = 0.5
    missing = []

    # Has RAG content?
    if ctx.rag_context:
        score += 0.2

    # Addresses the intent?
    if ctx.intent != "general" and ctx.intent in ctx.response.lower():
        score += 0.1

    # Has follow-up question?
    if ctx.follow_up_questions:
        score += 0.1

    # Knowledge gaps addressed?
    if ctx.knowledge_gaps and not any(gap in ctx.response for gap in ctx.knowledge_gaps):
        missing.append("先備知識提示")

    # Misconceptions addressed?
    if ctx.misconceptions and not any("迷思" in ctx.response for _ in ctx.misconceptions):
        missing.append("迷思概念提醒")

    ctx.completeness_score = min(score, 1.0)
    ctx.completeness_missing = missing

    return ctx
```

**Step 2: Verify import**

Run: `cd platform/backend && python -c "from app.nlp.response_gen import response_assembler, complexity_adjuster, citation_injector, follow_up_generator, encouragement_generator, response_completeness_checker; print('6 layers OK')"`
Expected: `6 layers OK`

---

### Task C10: Wire up the 42-layer pipeline and update API

**Files:**
- Modify: `platform/backend/app/nlp/__init__.py`
- Modify: `platform/backend/app/api/llm_routes.py`

**Step 1: Update `__init__.py` to export the full pipeline**

```python
"""NLP module — 42-layer micro-NLP pipeline for AI Tutor."""

from .pipeline import NLPContext, run_pipeline
from .preprocessing import (
    chinese_segmenter, pos_tagger, sentence_splitter,
    language_detector, text_normalizer, stopword_filter,
)
from .intent import detect_intent
from .understanding import (
    sub_intent_detector, sentiment_scorer, frustration_escalator,
    confidence_estimator, urgency_detector, politeness_detector,
)
from .emotion import detect_emotion
from .student_level import (
    difficulty_assessor, vocabulary_level_scorer, technical_fluency_scorer,
    learning_style_detector, misconception_detector, knowledge_gap_detector,
)
from .content_analysis import (
    keyword_extractor, domain_concept_matcher, named_entity_recognizer,
    code_block_detector, math_expression_detector, question_quality_scorer,
    readability_scorer,
)
from .context_memory import (
    conversation_tracker, topic_continuity_detector, hint_ladder_manager,
    session_summarizer, knowledge_state_tracker,
)
from .retrieval import (
    query_expander, rag_retriever, semantic_reranker, cross_week_linker,
)
from .response_gen import (
    response_assembler, complexity_adjuster, citation_injector,
    follow_up_generator, encouragement_generator, response_completeness_checker,
)

# Full 42-layer pipeline in execution order
FULL_PIPELINE = [
    # A. Preprocessing (L1-6)
    chinese_segmenter,       # L1
    pos_tagger,              # L2
    sentence_splitter,       # L3
    language_detector,        # L4
    text_normalizer,         # L5
    stopword_filter,         # L6
    # B. Understanding (L7-14)
    detect_intent,           # L7
    sub_intent_detector,     # L8
    detect_emotion,          # L9
    sentiment_scorer,        # L10
    frustration_escalator,   # L11
    confidence_estimator,    # L12
    urgency_detector,        # L13
    politeness_detector,     # L14
    # C. Student Level (L15-20)
    difficulty_assessor,     # L15
    vocabulary_level_scorer, # L16
    technical_fluency_scorer, # L17
    learning_style_detector, # L18
    misconception_detector,  # L19
    knowledge_gap_detector,  # L20
    # D. Content Analysis (L21-27)
    keyword_extractor,       # L21
    domain_concept_matcher,  # L22
    named_entity_recognizer, # L23
    code_block_detector,     # L24
    math_expression_detector, # L25
    question_quality_scorer, # L26
    readability_scorer,      # L27
    # E. Context & Memory (L28-32)
    conversation_tracker,    # L28
    topic_continuity_detector, # L29
    hint_ladder_manager,     # L30
    session_summarizer,      # L31
    knowledge_state_tracker, # L32
    # F. Retrieval (L33-36)
    query_expander,          # L33
    rag_retriever,           # L34
    semantic_reranker,       # L35
    cross_week_linker,       # L36
    # G. Response (L37-42)
    response_assembler,      # L37
    complexity_adjuster,     # L38
    citation_injector,       # L39
    follow_up_generator,     # L40
    encouragement_generator, # L41
    response_completeness_checker, # L42
]
```

**Step 2: Verify the full pipeline imports**

Run: `cd platform/backend && python -c "from app.nlp import FULL_PIPELINE, NLPContext, run_pipeline; print(f'{len(FULL_PIPELINE)} layers loaded'); ctx = NLPContext(user_message='什麼是梯度下降？', week=4, topic='梯度下降'); ctx = run_pipeline(ctx, FULL_PIPELINE); print(f'Intent: {ctx.intent}, Emotion: {ctx.emotion}, Keywords: {ctx.keywords[:3]}')"`
Expected: `42 layers loaded` and pipeline executes with results.

---

### Task C11: Expand training data and retrain models

**Files:**
- Modify: `platform/backend/app/nlp/trainer.py`

**Step 1: Add more diverse training samples**

In `_generate_intent_data()`, after the existing data generation, add additional natural-language training samples that simulate real students:

```python
    # ── Additional natural student queries ──
    add("definition", [
        "老師，loss function 到底是什麼意思啊", "可以用比喻解釋 overfitting 嗎",
        "正則化我一直聽不懂它在幹嘛", "那個 attention mechanism 是什麼東西",
        "batch normalization 是做什麼用的啊", "softmax 跟 sigmoid 是一樣的嗎",
        "embedding 是怎麼把文字變成數字的", "什麼叫做模型的泛化能力",
    ] * 2)
    add("how", [
        "老師我不知道 pipeline 怎麼串起來", "sklearn 的 cross_val_score 怎麼用",
        "我想自己從頭寫一個神經網路", "怎麼用 matplotlib 畫 confusion matrix",
        "SHAP force plot 要怎麼生成", "transfer learning 的步驟是什麼",
    ] * 2)
    add("debug", [
        "老師我的 model.fit 跑到一半就當掉了", "為什麼我的 loss 一直是 nan",
        "shape mismatch 這個錯怎麼解", "我 import torch 結果說找不到",
        "跑 CNN 的時候 GPU 記憶體不夠", "predict 出來的結果全部都一樣",
    ] * 2)
    add("compare", [
        "隨機森林跟 XGBoost 我該用哪個", "Adam 和 SGD 哪個比較好",
        "CNN 跟 Transformer 處理影像誰好", "L1 跟 L2 正則化差在哪",
    ] * 2)
    add("visualization", [
        "我想畫一個很漂亮的 ROC 曲線", "怎麼用 seaborn 畫 heatmap",
        "plotly 可以做 3D 散佈圖嗎", "我想視覺化 CNN 的 feature map",
    ] * 2)
    add("performance", [
        "我的模型 val accuracy 一直卡在 60%", "loss 下降很慢怎麼辦",
        "precision 跟 recall 怎麼取捨", "F1 score 太低了有什麼方法",
    ] * 2)
```

In `_generate_emotion_data()`, add:

```python
    add("frustrated", [
        "老師我真的要崩潰了這個 bug 找了三個小時", "為什麼照著教材做還是不對",
        "明明跟範例一模一樣為什麼就是跑不動", "我覺得我可能不適合學程式",
        "每次改一個 bug 就冒出十個新 bug", "我已經重做了五次了還是一樣的問題",
    ] * 2)
    add("confused", [
        "所以 bias 到底是偏差還是偏移啊", "我搞不清楚 validation set 跟 test set",
        "這個跟上禮拜教的好像不一樣？", "我覺得這兩個演算法好像", "結果很奇怪不確定對不對",
    ] * 2)
    add("curious", [
        "GAN 可以用來生成什麼有趣的東西", "如果把 Transformer 用在其他領域呢",
        "最近有什麼很酷的 AI 應用", "AlphaFold 是怎麼做到的",
    ] * 2)
    add("confident", [
        "我覺得是因為 learning rate 太大所以 loss 爆掉", "這應該用 ReLU 比 sigmoid 好",
        "我理解了，就是用梯度去更新權重對吧", "老師我做出來了！結果 accuracy 有 92%",
    ] * 2)
    add("neutral", [
        "這週的作業什麼時候交", "Notebook 在哪裡下載", "有辦公室時間嗎",
        "可以推薦一些參考書嗎", "考試範圍包含哪些週", "今天教到哪裡了",
    ] * 2)
```

**Step 2: Retrain models**

Run: `cd platform/backend && python -c "from app.nlp.trainer import train_models; r = train_models(); print(r)"`
Expected: Training completes with updated sample counts and accuracy scores.

---

### Task C12: Run all backend tests

**Step 1: Run full backend test suite**

Run: `cd platform/backend && python -m pytest -v`
Expected: All existing 38+ tests pass. New NLP layers don't break existing functionality.

---

### Task C13: Run full frontend test suite

**Step 1: Rebuild and run**

Run: `cd platform/frontend && npm run build && npx vitest run`
Expected: Build succeeds. All 35+ frontend tests pass.

---

### Task C14: Final verification

**Step 1: Start backend and test NLP pipeline endpoint**

Run:
```bash
cd platform/backend
python -c "
from app.nlp import FULL_PIPELINE, NLPContext, run_pipeline

# Test 1: Basic question
ctx = NLPContext(user_message='什麼是梯度下降？', week=4, topic='梯度下降')
ctx = run_pipeline(ctx, FULL_PIPELINE)
print(f'Test 1 — Layers: {len(ctx.layers_executed)}, Intent: {ctx.intent}, Emotion: {ctx.emotion}')
print(f'  Keywords: {ctx.keywords[:3]}, Level: {ctx.student_level}, Style: {ctx.learning_style}')
print(f'  Processing: {ctx.total_processing_ms:.0f}ms')

# Test 2: Frustrated student
ctx2 = NLPContext(user_message='我試了好久 model.fit 一直報錯 shape mismatch 快崩潰了', week=11, topic='神經網路')
ctx2 = run_pipeline(ctx2, FULL_PIPELINE)
print(f'Test 2 — Intent: {ctx2.intent}, Emotion: {ctx2.emotion}, Frustration: {ctx2.frustration_level}')
print(f'  Hint Level: {ctx2.hint_level}, Misconceptions: {len(ctx2.misconceptions)}')

# Test 3: Advanced student
ctx3 = NLPContext(user_message='我想做 ablation study 比較 attention mechanism 和 positional encoding 的影響', week=13, topic='Transformer')
ctx3 = run_pipeline(ctx3, FULL_PIPELINE)
print(f'Test 3 — Level: {ctx3.student_level}, Fluency: {ctx3.technical_fluency:.2f}, Vocab: {ctx3.vocabulary_score:.2f}')

print(f'\\nAll {len(FULL_PIPELINE)} layers verified!')
"
```

Expected: All 42 layers execute, correct intent/emotion/level detection for each test case.
