import { useState, useCallback, useEffect, useRef } from "react";
import { createWebSocket } from "../lib/api";
import { useAuth } from "./useAuth";
import type { LLMMessage } from "../types";

export type ChatStage = "idle" | "analyzing" | "draft" | "verifying" | "verified" | "unverified";

interface PendingChat {
  messages: LLMMessage[];
  mode: string;
}

export function useChat(week: number, topic: string) {
  const { token } = useAuth();
  const [messages, setMessages] = useState<LLMMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [stage, setStage] = useState<ChatStage>("idle");
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const lastRequestRef = useRef<PendingChat | null>(null);

  const closeActiveSocket = useCallback(() => {
    const active = wsRef.current;
    if (!active) return;
    active.onclose = null;
    active.onerror = null;
    active.close();
    wsRef.current = null;
  }, []);

  const startRequest = useCallback((request: PendingChat) => {
    closeActiveSocket();
    setIsLoading(true);
    setStage("analyzing");
    setError(null);

    const ws = createWebSocket("/api/llm/ws/chat", token || undefined);
    wsRef.current = ws;
    let refinementContent = "";
    let refinementStarted = false;
    let hasAssistant = false;
    let finished = false;

    const showAssistant = (text: string) => {
      hasAssistant = true;
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant") updated[updated.length - 1] = { ...last, content: text };
        else updated.push({ role: "assistant", content: text });
        return updated;
      });
    };

    const fail = (message: string, unverified = hasAssistant) => {
      if (finished) return;
      finished = true;
      setIsLoading(false);
      setStage(unverified ? "unverified" : "idle");
      setError(message);
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({
        messages: request.messages.map((message) => ({
          role: message.role,
          content: message.content,
        })),
        week,
        topic,
        mode: request.mode,
      }));
    };

    ws.onmessage = (event) => {
      let data: { type?: string; stage?: string; content?: string };
      try {
        data = JSON.parse(event.data);
      } catch {
        fail("收到無法解析的 AI 回覆，請重試。");
        ws.close();
        return;
      }

      if (data.type === "status") {
        setStage(data.stage === "verifying" ? "verifying" : "analyzing");
      } else if (data.type === "draft") {
        showAssistant(data.content || "");
        setStage("draft");
      } else if (data.type === "refinement" || data.type === "chunk") {
        if (!refinementStarted) {
          refinementStarted = true;
          refinementContent = "";
        }
        refinementContent += data.content || "";
        showAssistant(refinementContent);
        setStage("verifying");
      } else if (data.type === "done") {
        finished = true;
        setIsLoading(false);
        setStage("verified");
        setError(null);
        ws.close();
      } else if (data.type === "error") {
        fail(data.content || "AI 回覆暫時無法完成，請稍後重試。", data.stage === "refinement" || hasAssistant);
        ws.close();
      }
    };

    ws.onerror = () => fail("無法連線到 AI 助教，請檢查網路後重試。");

    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
      if (!finished) fail("AI 連線提早中斷，請重試上一題。");
    };
  }, [closeActiveSocket, token, topic, week]);

  const send = useCallback((content: string, mode: string = "tutor") => {
    const userMsg: LLMMessage = { role: "user", content };
    const allMessages = [...messages, userMsg];
    const request = { messages: allMessages, mode };
    setMessages(allMessages);
    lastRequestRef.current = request;
    startRequest(request);
  }, [messages, startRequest]);

  const retry = useCallback(() => {
    if (lastRequestRef.current) startRequest(lastRequestRef.current);
  }, [startRequest]);

  const stop = useCallback(() => {
    closeActiveSocket();
    setIsLoading(false);
    setStage((current) => current === "draft" || current === "verifying" ? "unverified" : "idle");
  }, [closeActiveSocket]);

  const clear = useCallback(() => {
    closeActiveSocket();
    setMessages([]);
    setIsLoading(false);
    setStage("idle");
    setError(null);
    lastRequestRef.current = null;
  }, [closeActiveSocket]);

  useEffect(() => closeActiveSocket, [closeActiveSocket]);

  return { messages, isLoading, stage, error, send, retry, stop, clear };
}
