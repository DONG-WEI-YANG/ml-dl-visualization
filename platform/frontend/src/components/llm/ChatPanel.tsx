import { useState, useRef, useEffect } from "react";
import { useChat } from "../../hooks/useChat";
import MarkdownRenderer from "./MarkdownRenderer";

interface Props {
  week: number;
  topic: string;
}

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

  return (
    <div className="flex flex-col h-[500px] border border-gray-200 rounded-xl bg-white" role="region" aria-label="AI Tutor Chat">
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
