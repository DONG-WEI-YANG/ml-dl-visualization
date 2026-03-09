import { useState, useRef, useEffect } from "react";
import { useChat } from "../../hooks/useChat";
import MarkdownRenderer from "./MarkdownRenderer";

interface Props {
  week: number;
  topic: string;
}

const SUGGESTED_QUESTIONS: Record<number, string[]> = {
  1: [
    "機器學習和深度學習有什麼差別？",
    "什麼是監督式學習？能舉個醫療上的例子嗎？",
    "過擬合是什麼？為什麼要正則化？",
    "ML 的完整工作流程是怎樣的？",
    "Python 在 ML 領域為什麼這麼受歡迎？",
  ],
  2: [
    "EDA 的基本步驟有哪些？",
    "如何處理資料中的缺失值？",
    "什麼時候用直方圖、什麼時候用箱型圖？",
    "相關係數矩陣怎麼解讀？",
  ],
  3: [
    "為什麼要把資料分成訓練集和測試集？",
    "K-Fold 交叉驗證是怎麼運作的？",
    "什麼是資料洩漏？怎麼避免？",
    "分層抽樣的 stratify 參數什麼時候該用？",
  ],
  4: [
    "梯度下降的學習率要怎麼選？",
    "線性回歸和梯度下降有什麼關係？",
    "什麼是正規化？L1 和 L2 有什麼差別？",
    "損失函數為什麼通常用 MSE？",
  ],
  5: [
    "邏輯迴歸為什麼叫「迴歸」但用在分類？",
    "Sigmoid 函數的作用是什麼？",
    "怎麼視覺化決策邊界？",
    "多類別分類怎麼處理？",
  ],
  6: [
    "SVM 的「最大間距」是什麼意思？",
    "核函數為什麼能處理非線性問題？",
    "超參數 C 和 gamma 怎麼調？",
    "SVM 適合什麼類型的資料？",
  ],
  7: [
    "決策樹怎麼決定用哪個特徵分割？",
    "隨機森林和單棵決策樹差在哪？",
    "Bagging 和 Boosting 的差別是什麼？",
    "XGBoost 為什麼這麼受歡迎？",
    "怎麼防止決策樹過擬合？",
  ],
  8: [
    "特徵重要度有哪幾種計算方式？",
    "SHAP 值是什麼？怎麼解讀？",
    "為什麼模型可解釋性很重要？",
    "LIME 和 SHAP 有什麼差別？",
  ],
  9: [
    "什麼時候需要做特徵縮放？",
    "One-Hot Encoding 和 Label Encoding 怎麼選？",
    "sklearn Pipeline 的好處是什麼？",
    "特徵工程有什麼實用技巧？",
  ],
  10: [
    "Grid Search 和 Random Search 哪個比較好？",
    "學習曲線怎麼判斷過擬合？",
    "什麼是偏差-變異度權衡？",
    "怎麼選擇適合的模型？",
  ],
  11: [
    "神經網路的反向傳播是怎麼運作的？",
    "為什麼需要活化函數？",
    "ReLU 相比 Sigmoid 有什麼優勢？",
    "Dropout 是怎麼防止過擬合的？",
  ],
  12: [
    "CNN 的卷積層在做什麼？",
    "為什麼 CNN 適合處理影像？",
    "什麼是遷移學習？怎麼用？",
    "ResNet 的跳躍連接解決什麼問題？",
  ],
  13: [
    "RNN 為什麼有梯度消失問題？",
    "LSTM 的門控機制是怎麼運作的？",
    "Transformer 的 Self-Attention 是什麼？",
    "為什麼 Transformer 取代了 RNN？",
  ],
  14: [
    "學習率排程有哪些常見策略？",
    "Batch Normalization 的原理是什麼？",
    "梯度爆炸要怎麼解決？",
    "混合精度訓練的好處是什麼？",
  ],
  15: [
    "Accuracy 為什麼不適合不平衡資料？",
    "Precision 和 Recall 怎麼取捨？",
    "ROC 曲線和 AUC 怎麼解讀？",
    "怎麼確保模型的公平性？",
  ],
  16: [
    "MLOps 和 DevOps 有什麼關係？",
    "MLflow 可以用來做什麼？",
    "怎麼把模型部署成 API？",
    "什麼是模型漂移？怎麼偵測？",
  ],
  17: [
    "RAG 是怎麼運作的？",
    "什麼是嵌入向量（Embedding）？",
    "Prompt Engineering 有哪些技巧？",
    "LLM 的幻覺問題怎麼解決？",
  ],
  18: [
    "ML 專題怎麼規劃時程？",
    "技術報告的架構應該怎麼寫？",
    "Demo 簡報有什麼技巧？",
    "學完這門課後可以怎麼繼續學習？",
  ],
};

export default function ChatPanel({ week, topic }: Props) {
  const { messages, isLoading, send, clear } = useChat(week, topic);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"tutor" | "homework">("tutor");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    send(input.trim(), mode);
    setInput("");
  };

  const handleSuggestedQuestion = (question: string) => {
    if (isLoading) return;
    send(question, mode);
  };

  const suggestedQuestions = SUGGESTED_QUESTIONS[week] || [];

  return (
    <div className="flex flex-col h-[calc(100vh-120px)] max-h-[700px] min-h-[400px] border border-gray-200 rounded-xl bg-white" role="region" aria-label="AI Tutor Chat">
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b rounded-t-xl">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900 text-sm">AI 助教</span>
        </div>
        <div className="flex gap-1.5">
          <button
            onClick={() => setMode("tutor")}
            className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
              mode === "tutor"
                ? "bg-blue-500 text-white"
                : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
            }`}
          >
            學習模式
          </button>
          <button
            onClick={() => setMode("homework")}
            className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
              mode === "homework"
                ? "bg-orange-500 text-white"
                : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
            }`}
          >
            作業模式
          </button>
          <button
            onClick={clear}
            className="px-2.5 py-1 text-xs bg-white text-gray-600 border border-gray-200 rounded-md hover:bg-gray-50"
          >
            清除
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3" role="log" aria-label="Chat messages" aria-live="polite">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 text-sm py-8">
            <p className="mb-2">歡迎使用 AI 助教！</p>
            <p>目前模式：{mode === "tutor" ? "學習模式" : "作業模式"}</p>
            <p className="mt-2 text-xs">提示：AI 助教會引導你思考，不會直接給答案</p>
            {suggestedQuestions.length > 0 && (
              <div className="mt-4">
                <p className="text-xs text-gray-400 mb-2">試試以下問題：</p>
                <div className="flex flex-wrap justify-center gap-1.5">
                  {suggestedQuestions.map((question, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSuggestedQuestion(question)}
                      disabled={isLoading}
                      className="px-2.5 py-1.5 text-xs bg-white text-blue-600 border border-blue-200 rounded-full hover:bg-blue-50 hover:border-blue-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-blue-500 text-white rounded-br-md"
                  : "bg-gray-100 text-gray-800 rounded-bl-md"
              }`}
            >
              {m.role === "assistant" ? (
                <MarkdownRenderer content={m.content} />
              ) : (
                <pre className="whitespace-pre-wrap font-sans">{m.content}</pre>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 px-4 py-2 rounded-2xl rounded-bl-md">
              <span className="text-gray-400 text-sm animate-pulse">AI 助教思考中...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 p-3 border-t border-gray-200">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 border border-gray-200 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="輸入你的問題..."
          disabled={isLoading}
          aria-label="Chat input"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          送出
        </button>
      </form>
    </div>
  );
}
