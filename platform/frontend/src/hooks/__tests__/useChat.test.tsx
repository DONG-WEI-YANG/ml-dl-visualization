import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useChat } from "../useChat";

vi.mock("../useAuth", () => ({
  useAuth: () => ({ token: "student-token" }),
}));

const sockets: FakeWebSocket[] = [];

class FakeWebSocket {
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  send = vi.fn();
  close = vi.fn(() => this.onclose?.({} as CloseEvent));
}

vi.mock("../../lib/api", () => ({
  createWebSocket: vi.fn(() => {
    const socket = new FakeWebSocket();
    sockets.push(socket);
    return socket;
  }),
}));

describe("useChat lifecycle", () => {
  beforeEach(() => {
    sockets.length = 0;
    vi.clearAllMocks();
  });

  it("closes the prior socket before starting another request", () => {
    const { result } = renderHook(() => useChat(4, "梯度下降"));

    act(() => result.current.send("first"));
    act(() => result.current.send("second"));

    expect(sockets).toHaveLength(2);
    expect(sockets[0].close).toHaveBeenCalledOnce();
  });

  it("surfaces the server error content and permits retry", () => {
    const { result } = renderHook(() => useChat(4, "梯度下降"));
    act(() => result.current.send("question"));

    act(() => sockets[0].onmessage?.({
      data: JSON.stringify({ type: "error", stage: "refinement", content: "模型暫時忙碌" }),
    } as MessageEvent));

    expect(result.current.error).toBe("模型暫時忙碌");
    expect(result.current.stage).toBe("unverified");

    act(() => result.current.retry());
    expect(sockets).toHaveLength(2);
    expect(result.current.messages.filter((message) => message.role === "user")).toHaveLength(1);
  });

  it("turns malformed frames into a recoverable error", () => {
    const { result } = renderHook(() => useChat(1, "基礎"));
    act(() => result.current.send("question"));

    act(() => sockets[0].onmessage?.({ data: "not-json" } as MessageEvent));

    expect(result.current.error).toContain("無法解析");
    expect(result.current.isLoading).toBe(false);
  });

  it("stops generation and closes the active socket", () => {
    const { result } = renderHook(() => useChat(1, "基礎"));
    act(() => result.current.send("question"));

    act(() => result.current.stop());

    expect(sockets[0].close).toHaveBeenCalledOnce();
    expect(result.current.isLoading).toBe(false);
  });
});
