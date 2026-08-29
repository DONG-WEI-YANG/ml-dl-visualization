import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AdminSettings from "../AdminSettings";

vi.mock("../../hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: 1, username: "admin", display_name: "管理員", role: "admin" },
    token: "admin-token",
    logout: vi.fn(),
  }),
}));

const settingsResponse = {
  settings: {
    llm_provider: "local",
    llm_model: "local-nlp",
    rag_enabled: "true",
    rag_top_k: "5",
  },
  available_providers: [
    { id: "local", name: "本地 NLP", models: ["local-nlp"] },
    { id: "openai", name: "OpenAI", models: ["gpt-test"] },
  ],
};

const ragStats = {
  total_chunks: 10,
  curriculum_chunks: 8,
  web_chunks: 2,
  by_week: [{ week: 1, count: 8 }],
};

function jsonResponse(data: unknown, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(data) } as Response);
}

function installFetch(diagnostics: Record<string, unknown>, putSucceeds = true) {
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/api/admin/settings") && init?.method === "PUT") {
      return putSucceeds
        ? jsonResponse({ settings: settingsResponse.settings })
        : jsonResponse({}, false, 500);
    }
    if (url.includes("/api/admin/settings")) return jsonResponse(settingsResponse);
    if (url.includes("/api/rag/stats")) return jsonResponse(ragStats);
    if (url.includes("/api/llm/diagnostics")) return jsonResponse(diagnostics);
    throw new Error(`Unexpected URL: ${url}`);
  }));
}

describe("AdminSettings AI diagnostics", () => {
  beforeEach(() => vi.unstubAllGlobals());

  it("shows the configured to effective provider path", async () => {
    installFetch({
      configured_provider: "local",
      configured_model: "local-nlp",
      effective_provider: "local",
      effective_model: "local-nlp-v3",
      status: "ready",
      reason: null,
      probe: { attempted: false },
    });

    render(<AdminSettings />);

    expect(await screen.findByText("AI 運作狀態")).toBeInTheDocument();
    expect(screen.getByText("local / local-nlp")).toBeInTheDocument();
    expect(screen.getByText("local / local-nlp-v3")).toBeInTheDocument();
    expect(screen.getByText("正常運作")).toBeInTheDocument();
  });

  it("explains a missing-key fallback without exposing a secret", async () => {
    installFetch({
      configured_provider: "openai",
      configured_model: "gpt-test",
      effective_provider: "local",
      effective_model: "local-nlp-v3",
      status: "degraded",
      reason: "missing_api_key",
      probe: { attempted: false },
    });

    render(<AdminSettings />);

    expect(await screen.findByText("已降級")).toBeInTheDocument();
    expect(screen.getByText("外部 AI 金鑰未設定，已改用本地 NLP。")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("admin-token");
  });

  it("can run a probe and display a safe failure", async () => {
    installFetch({
      configured_provider: "local",
      configured_model: "local-nlp",
      effective_provider: "local",
      effective_model: "local-nlp-v3",
      status: "error",
      reason: null,
      probe: { attempted: true, ok: false, reason: "provider_unreachable", latency_ms: 15 },
    });

    render(<AdminSettings />);
    fireEvent.click(await screen.findByRole("button", { name: "執行 AI 探測" }));

    expect(await screen.findByText("探測失敗")).toBeInTheDocument();
  });

  it("shows truthful local runtime fallback details", async () => {
    installFetch({
      configured_provider: "local",
      configured_model: "local-nlp",
      effective_provider: "local",
      effective_model: "local-nlp-v3",
      status: "degraded",
      reason: "runtime_degraded",
      runtime_warnings: ["sklearn_model_version_mismatch", "semantic_model_uncached"],
      probe: { attempted: true, ok: true, latency_ms: 125 },
    });

    render(<AdminSettings />);

    expect(await screen.findByText("本地 AI 可回覆，但部分能力正在使用備援：")).toBeInTheDocument();
    expect(screen.getByText("分類模型與目前 scikit-learn 版本不一致。")).toBeInTheDocument();
    expect(screen.getByText("語意向量模型尚未快取，改用關鍵字相似度。")).toBeInTheDocument();
    expect(screen.getByText("探測成功 · 125 ms")).toBeInTheDocument();
  });

  it("keeps the confirmed provider when a save fails", async () => {
    installFetch({
      configured_provider: "local",
      configured_model: "local-nlp",
      effective_provider: "local",
      effective_model: "local-nlp-v3",
      status: "ready",
      reason: null,
      probe: { attempted: false },
    }, false);

    render(<AdminSettings />);
    const providerSelect = (await screen.findAllByRole("combobox"))[0] as HTMLSelectElement;
    fireEvent.change(providerSelect, { target: { value: "openai" } });

    expect(await screen.findByText("儲存失敗")).toBeInTheDocument();
    expect(providerSelect.value).toBe("local");
  });
});
