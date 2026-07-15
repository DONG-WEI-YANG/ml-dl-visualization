export const API_BASE = import.meta.env.VITE_API_BASE || "";

export type APIErrorKind = "unauthorized" | "timeout" | "offline" | "server" | "unknown";

export class APIError extends Error {
  constructor(public kind: APIErrorKind, message: string, public status?: number) {
    super(message);
    this.name = "APIError";
  }
}

export interface FetchAPIOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

export async function fetchAPI<T>(path: string, body?: unknown, token?: string, options: FetchAPIOptions = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const controller = new AbortController();
  let timedOut = false;
  const relayAbort = () => controller.abort();
  options.signal?.addEventListener("abort", relayAbort, { once: true });
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, options.timeoutMs ?? 15000);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: body ? "POST" : "GET",
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!res.ok) {
      const kind: APIErrorKind = res.status === 401 ? "unauthorized" : res.status >= 500 ? "server" : "unknown";
      throw new APIError(kind, `API error: ${res.status}`, res.status);
    }
    return res.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    if (timedOut) throw new APIError("timeout", "Request timed out");
    if (controller.signal.aborted) throw new APIError("unknown", "Request cancelled");
    if (typeof navigator !== "undefined" && !navigator.onLine) throw new APIError("offline", "Browser is offline");
    throw new APIError("unknown", error instanceof Error ? error.message : "Network request failed");
  } finally {
    window.clearTimeout(timer);
    options.signal?.removeEventListener("abort", relayAbort);
  }
}

export function createWebSocket(path: string, token?: string): WebSocket {
  const base = API_BASE || window.location.origin;
  const wsBase = base.replace(/^http/, "ws");
  const url = token
    ? `${wsBase}${path}?token=${encodeURIComponent(token)}`
    : `${wsBase}${path}`;
  return new WebSocket(url);
}
