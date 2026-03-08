import { useState, useCallback, useRef } from "react";
import { createWebSocket } from "../lib/api";
import { useAuth } from "./useAuth";
import type { LLMMessage } from "../types";

export function useChat(week: number, topic: string) {
  const { token } = useAuth();
  const [messages, setMessages] = useState<LLMMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const send = useCallback(
    (content: string, mode: string = "tutor") => {
      const userMsg: LLMMessage = { role: "user", content };
      const allMessages = [...messages, userMsg];
      setMessages(allMessages);
      setIsLoading(true);

      const ws = createWebSocket("/api/llm/ws/chat", token || undefined);
      wsRef.current = ws;
      let assistantContent = "";

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
        if (data.type === "chunk") {
          assistantContent += data.content;
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant") {
              updated[updated.length - 1] = {
                ...last,
                content: assistantContent,
              };
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

      ws.onerror = () => {
        setIsLoading(false);
      };

      ws.onclose = () => {
        setIsLoading(false);
      };
    },
    [messages, week, topic, token]
  );

  const clear = useCallback(() => {
    setMessages([]);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  return { messages, isLoading, send, clear };
}
