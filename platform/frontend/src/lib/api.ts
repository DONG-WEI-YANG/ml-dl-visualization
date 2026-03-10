export const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function fetchAPI<T>(path: string, body?: unknown, token?: string): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, {
    method: body ? "POST" : "GET",
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export function createWebSocket(path: string, token?: string): WebSocket {
  const base = API_BASE || window.location.origin;
  const wsBase = base.replace(/^http/, "ws");
  const url = token
    ? `${wsBase}${path}?token=${encodeURIComponent(token)}`
    : `${wsBase}${path}`;
  return new WebSocket(url);
}
