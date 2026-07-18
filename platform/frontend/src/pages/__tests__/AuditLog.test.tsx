import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import AuditLog from "../AuditLog";

const { items } = vi.hoisted(() => ({
  items: [
    { id: 2, timestamp: "2026-07-18 08:00:00", actor_username: "admin", actor_role: "admin",
      action: "user.create", target_type: "user", target_id: "9", detail: "{}", ip: "1.1.1.1" },
    { id: 1, timestamp: "2026-07-18 07:00:00", actor_username: "", actor_role: "",
      action: "login.failed", target_type: "", target_id: "", detail: '{"username":"ghost"}', ip: "2.2.2.2" },
  ],
}));

vi.mock("../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../lib/api")>();
  return {
    ...mod,
    fetchAPI: vi.fn().mockResolvedValue({ items, total: 2, page: 1, page_size: 50 }),
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
});
