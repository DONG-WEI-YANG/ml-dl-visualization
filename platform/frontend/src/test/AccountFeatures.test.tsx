import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { stripHeader } from "../lib/csv";

afterEach(() => {
  localStorage.clear();
  window.history.replaceState({}, "", "/");
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const admin = { id: 1, username: "admin", display_name: "Admin", role: "admin", semester: "115-1", must_change_password: false };
const student = (id: number, username: string, extra = {}) => ({
  id, username, display_name: username, email: "", role: "student", semester: "115-1", class_name: "護一甲",
  is_active: true, approval_status: "approved", created_at: "2026-11-03 08:00:00", updated_at: "", ...extra,
});

describe("login page sign-up", () => {
  it("submits a sign-up request and shows the pending notice", async () => {
    window.history.replaceState({}, "", "/login");
    const fetcher = vi.fn(async (url: string, _init?: RequestInit) => {
      void _init;
      if (url.endsWith("/api/auth/signup")) return Response.json(student(9, "s1150009", { approval_status: "pending" }), { status: 201 });
      return Response.json({}, { status: 401 });
    });
    vi.stubGlobal("fetch", fetcher);
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "申請帳號" }));
    const form = screen.getByRole("form", { name: "申請帳號" });
    fireEvent.change(within(form).getByLabelText(/學號/), { target: { value: "s1150009" } });
    fireEvent.change(within(form).getByLabelText(/姓名/), { target: { value: "王小明" } });
    fireEvent.change(within(form).getByLabelText(/^密碼/), { target: { value: "longpassword" } });
    fireEvent.change(within(form).getByLabelText(/確認密碼/), { target: { value: "longpassword" } });
    fireEvent.click(within(form).getByRole("button", { name: "送出申請" }));
    expect(await screen.findByRole("status")).toHaveTextContent("申請已送出");
    const call = fetcher.mock.calls.find(([u]) => String(u).endsWith("/api/auth/signup"));
    expect(JSON.parse(String((call?.[1] as RequestInit).body))).toMatchObject({ username: "s1150009", display_name: "王小明" });
  });

  it("rejects mismatched passwords without calling the API", async () => {
    window.history.replaceState({}, "", "/login");
    const fetcher = vi.fn(async (_url: string) => { void _url; return Response.json({}, { status: 401 }); });
    vi.stubGlobal("fetch", fetcher);
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "申請帳號" }));
    const form = screen.getByRole("form", { name: "申請帳號" });
    fireEvent.change(within(form).getByLabelText(/學號/), { target: { value: "s1150010" } });
    fireEvent.change(within(form).getByLabelText(/姓名/), { target: { value: "李" } });
    fireEvent.change(within(form).getByLabelText(/^密碼/), { target: { value: "longpassword" } });
    fireEvent.change(within(form).getByLabelText(/確認密碼/), { target: { value: "different1" } });
    fireEvent.click(within(form).getByRole("button", { name: "送出申請" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("不一致");
    expect(fetcher.mock.calls.some(([u]) => String(u).endsWith("/api/auth/signup"))).toBe(false);
  });

  it("shows the server's pending-approval reason on 403 login", async () => {
    window.history.replaceState({}, "", "/login");
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (url.endsWith("/api/auth/login")) return Response.json({ detail: "帳號申請審核中，請等候老師或管理員核准" }, { status: 403 });
      return Response.json({}, { status: 401 });
    }));
    render(<App />);
    fireEvent.change(await screen.findByLabelText("Username"), { target: { value: "s1150009" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "longpassword" } });
    fireEvent.click(screen.getByRole("button", { name: "登入" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("審核中");
  });
});

function adminFetcher(handlers: Record<string, (init?: RequestInit) => unknown> = {}) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/api/auth/me")) return Response.json(admin);
    for (const [suffix, fn] of Object.entries(handlers)) {
      if (url.includes(suffix)) return Response.json(fn(init));
    }
    if (url.includes("/api/admin/users?status=pending")) return Response.json([student(7, "s_pending", { approval_status: "pending", is_active: false })]);
    if (url.includes("/api/admin/users")) return Response.json([admin && student(1, "admin", { role: "admin" }), student(2, "s1"), student(3, "s2")]);
    return Response.json([]);
  });
}

describe("account management", () => {
  it("lists pending sign-ups and approves one", async () => {
    window.history.replaceState({}, "", "/admin/users");
    localStorage.setItem("auth_token", "t");
    const fetcher = adminFetcher({ "/users/7/approve": () => student(7, "s_pending") });
    vi.stubGlobal("fetch", fetcher);
    render(<App />);
    const panel = await screen.findByRole("region", { name: "待審核申請" });
    fireEvent.click(within(panel).getByRole("button", { name: "核准" }));
    await waitFor(() =>
      expect(fetcher.mock.calls.some(([u, i]) => String(u).endsWith("/api/admin/users/7/approve") && i?.method === "POST")).toBe(true),
    );
  });

  it("debounces the search box into a q= query", async () => {
    window.history.replaceState({}, "", "/admin/users");
    localStorage.setItem("auth_token", "t");
    const fetcher = adminFetcher();
    vi.stubGlobal("fetch", fetcher);
    render(<App />);
    fireEvent.change(await screen.findByLabelText("搜尋使用者"), { target: { value: "王" } });
    await waitFor(() =>
      expect(fetcher.mock.calls.some(([u]) => String(u).includes("q=%E7%8E%8B"))).toBe(true),
    );
  });

  it("bulk-resets passwords for selected rows and shows the new passwords", async () => {
    window.history.replaceState({}, "", "/admin/users");
    localStorage.setItem("auth_token", "t");
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const fetcher = adminFetcher({
      "/api/admin/users/bulk": () => ({ updated: 1, passwords: [{ username: "s1", initial_password: "NewPass-12345" }] }),
    });
    vi.stubGlobal("fetch", fetcher);
    render(<App />);
    fireEvent.click(await screen.findByLabelText("選取 s1"));
    // own account (admin) is never selectable
    expect(screen.queryByLabelText("選取 admin")).not.toBeInTheDocument();
    const bar = screen.getByRole("toolbar", { name: "批次操作" });
    fireEvent.click(within(bar).getByRole("button", { name: "重設密碼" }));
    expect(await screen.findByText("NewPass-12345")).toBeInTheDocument();
    const call = fetcher.mock.calls.find(([u]) => String(u).endsWith("/api/admin/users/bulk"));
    expect(JSON.parse(String((call?.[1] as RequestInit).body))).toEqual({ ids: [2], action: "reset_password" });
  });
});

describe("CSV import helper", () => {
  it("strips BOM and header row", () => {
    expect(stripHeader("﻿學號,姓名\ns1,王\ns2,李")).toBe("s1,王\ns2,李");
    expect(stripHeader("s1,王\ns2,李")).toBe("s1,王\ns2,李");
  });
});
