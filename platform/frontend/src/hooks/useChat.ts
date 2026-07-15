import { useState, useCallback, useRef } from "react";
import { createWebSocket } from "../lib/api";
import { useAuth } from "./useAuth";
import type { LLMMessage } from "../types";

export type ChatStage = "idle" | "analyzing" | "draft" | "verifying" | "verified" | "unverified";

export function useChat(week: number, topic: string) {
  const { token } = useAuth();
  const [messages, setMessages] = useState<LLMMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [stage, setStage] = useState<ChatStage>("idle");
  const wsRef = useRef<WebSocket | null>(null);

  const send = useCallback(
    (content: string, mode: string = "tutor") => {
      const userMsg: LLMMessage = { role: "user", content };
      const allMessages = [...messages, userMsg];
      setMessages(allMessages);
      setIsLoading(true);
      setStage("analyzing");

      const ws = createWebSocket("/api/llm/ws/chat", token || undefined);
      wsRef.current = ws;
      let assistantContent = "";
      let refinementContent = "";
      let refinementStarted = false;

      const showAssistant = (text: string) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.role === "assistant") updated[updated.length - 1] = { ...last, content: text };
          else updated.push({ role: "assistant", content: text });
          return updated;
        });
      };

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            messages: allMessages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
            week,
            topic,
            mode,
          })
        );
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "status") {
          setStage(data.stage === "verifying" ? "verifying" : "analyzing");
        } else if (data.type === "draft") {
          assistantContent = data.content;
          showAssistant(assistantContent);
          setStage("draft");
        } else if (data.type === "refinement" || data.type === "chunk") {
          if (!refinementStarted) {
            refinementStarted = true;
            refinementContent = "";
          }
          refinementContent += data.content;
          showAssistant(refinementContent);
          setStage("verifying");
        } else if (data.type === "done") {
          setIsLoading(false);
          setStage("verified");
          ws.close();
        } else if (data.type === "error") {
          setIsLoading(false);
          setStage(data.stage === "refinement" ? "unverified" : "idle");
        }
      };

      ws.onerror = () => {
        setIsLoading(false);
        setStage(assistantContent ? "unverified" : "idle");
      };

      ws.onclose = () => {
        setIsLoading(false);
      };
    },
    [messages, week, topic, token]
  );

  const clear = useCallback(() => {
    setMessages([]);
    setStage("idle");
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  return { messages, isLoading, stage, send, clear };
}
