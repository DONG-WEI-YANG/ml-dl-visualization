import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import AuditLog from "../AuditLog";
import { fetchAPI } from "../../lib/api";

const { items, users } = vi.hoisted(() => ({
  items: [
    { id: 2, timestamp: "2026-07-18 08:00:00", actor_username: "admin", actor_role: "admin",
      action: "user.create", target_type: "user", target_id: "9", detail: "{}", ip: "1.1.1.1" },
    { id: 1, timestamp: "2026-07-18 07:00:00", actor_username: "", actor_role: "",
      action: "login.failed", target_type: "", target_id: "", detail: '{"username":"ghost"}', ip: "2.2.2.2" },
  ],
  users: [
    { id: 1, username: "admin", display_name: "管理員" },
    { id: 2, username: "stu1", display_name: "學生一" },
  ],
}));

vi.mock("../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../lib/api")>();
  return {
    ...mod,
    fetchAPI: vi.fn((path: string) => {
      if (path.startsWith("/api/admin/users")) return Promise.resolve(users);
      return Promise.resolve({ items, total: 2, page: 1, page_size: 50 });
    }),
  };
});
vi.mock("../../hooks/useAuth", () => ({
  useAuth: () => ({ token: "t", user: { role: "admin" } }),
}));

describe("AuditLog", () => {
  it("renders audit rows from the API", async () => {
    render(<AuditLog />);
    await waitFor(() => {
      expect(screen.getByText("user.create")).toBeDefined();
      expect(screen.getByText("login.failed")).toBeDefined();
    });
  });

  it("shows tab buttons for the three views", async () => {
    render(<AuditLog />);
    expect(screen.getByRole("button", { name: /管理動作/ })).toBeDefined();
    expect(screen.getByRole("button", { name: /登入歷程/ })).toBeDefined();
    expect(screen.getByRole("button", { name: /學習行為/ })).toBeDefined();
  });

  it("shows total count", async () => {
    render(<AuditLog />);
    await waitFor(() => expect(screen.getByText(/共 2 筆/)).toBeDefined());
  });

  it("exports CSV with an authenticated fetch", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["x"])),
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn().mockReturnValue("blob:mock"),
      revokeObjectURL: vi.fn(),
    });

    render(<AuditLog />);
    await waitFor(() => expect(screen.getByText("user.create")).toBeDefined());
    fireEvent.click(screen.getByRole("button", { name: "匯出 CSV" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const [url, options] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/api/admin/audit-logs/export");
    expect(options.headers.Authorization).toBe("Bearer t");

    vi.unstubAllGlobals();
  });

  it("renders user filter options fetched from the users API", async () => {
    render(<AuditLog />);
    await waitFor(() => {
      expect(screen.getByText("學生一 (stu1)")).toBeDefined();
    });
  });

  it("refetches with actor_id when the user filter changes, resetting to page 1", async () => {
    render(<AuditLog />);
    await waitFor(() => expect(screen.getByText("user.create")).toBeDefined());
    vi.mocked(fetchAPI).mockClear();

    fireEvent.change(screen.getByLabelText("使用者篩選"), { target: { value: "2" } });

    await waitFor(() => {
      const call = vi.mocked(fetchAPI).mock.calls.find(([url]) =>
        String(url).includes("/api/admin/audit-logs?"),
      );
      expect(call?.[0]).toContain("actor_id=2");
      expect(call?.[0]).toContain("page=1");
    });
  });

  it("refetches with from/to when the date range filters change", async () => {
    render(<AuditLog />);
    await waitFor(() => expect(screen.getByText("user.create")).toBeDefined());
    vi.mocked(fetchAPI).mockClear();

    fireEvent.change(screen.getByLabelText("起"), { target: { value: "2026-07-01" } });
    fireEvent.change(screen.getByLabelText("訖"), { target: { value: "2026-07-19" } });

    await waitFor(() => {
      const calls = vi.mocked(fetchAPI).mock.calls.filter(([url]) =>
        String(url).includes("/api/admin/audit-logs?"),
      );
      const lastCall = calls[calls.length - 1];
      expect(String(lastCall?.[0])).toContain("from=2026-07-01");
      expect(String(lastCall?.[0])).toContain("to=2026-07-19");
    });
  });

  it("carries user and date filters into the CSV export URL", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["x"])),
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn().mockReturnValue("blob:mock"),
      revokeObjectURL: vi.fn(),
    });

    render(<AuditLog />);
    await waitFor(() => expect(screen.getByText("user.create")).toBeDefined());
    fireEvent.change(screen.getByLabelText("使用者篩選"), { target: { value: "2" } });
    await waitFor(() => expect(screen.getByLabelText("使用者篩選")).toHaveProperty("value", "2"));

    fireEvent.click(screen.getByRole("button", { name: "匯出 CSV" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("actor_id=2");

    vi.unstubAllGlobals();
  });
});
