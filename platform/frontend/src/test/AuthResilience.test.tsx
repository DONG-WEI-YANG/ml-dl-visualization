import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";

describe("cloud auth resilience", () => {
  afterEach(() => {
    localStorage.clear();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("offers offline course access when auth verification times out", async () => {
    vi.useFakeTimers();
    localStorage.setItem("auth_token", "saved-token");
    vi.stubGlobal("fetch", vi.fn((_url, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    })));

    render(<App />);
    await act(() => vi.advanceTimersByTimeAsync(3000));
    expect(screen.getByText("正在喚醒雲端服務")).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(5000));
    const enter = screen.getByRole("button", { name: "進入離線課程" });
    expect(localStorage.getItem("auth_token")).toBe("saved-token");
    fireEvent.click(enter);
    expect(screen.getByRole("list", { name: "18-week curriculum" })).toBeInTheDocument();
  });

  it("removes the token only after an explicit 401", async () => {
    localStorage.setItem("auth_token", "expired-token");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 401 })));
    render(<App />);
    expect(await screen.findByLabelText("Username")).toBeInTheDocument();
    expect(localStorage.getItem("auth_token")).toBeNull();
  });
});
