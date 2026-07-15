import { afterEach, describe, expect, it, vi } from "vitest";
import { APIError, fetchAPI } from "../lib/api";

describe("fetchAPI", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("classifies 401 as unauthorized", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 401 })));
    await expect(fetchAPI("/api/auth/me")).rejects.toMatchObject<Partial<APIError>>({
      kind: "unauthorized",
      status: 401,
    });
  });

  it("classifies 503 as server", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 503 })));
    await expect(fetchAPI("/health")).rejects.toMatchObject<Partial<APIError>>({
      kind: "server",
      status: 503,
    });
  });

  it("aborts a request after its timeout", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn((_url, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    })));
    const request = fetchAPI("/health", undefined, undefined, { timeoutMs: 10 });
    const rejection = expect(request).rejects.toMatchObject<Partial<APIError>>({ kind: "timeout" });
    await vi.advanceTimersByTimeAsync(10);
    await rejection;
  });
});
