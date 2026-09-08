import { expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import AuditLog from "../pages/AuditLog";

vi.mock("../hooks/useAuth", () => ({ useAuth: () => ({ token: "test-token" }) }));
vi.mock("../lib/api", () => ({
  API_BASE: "",
  fetchAPI: vi.fn((path: string) => Promise.resolve(
    path === "/api/admin/users" ? [] : { items: [], total: 0, page: 1, page_size: 50 },
  )),
}));

it("keeps the GitHub Pages project path when opening learning analytics", async () => {
  render(
    <MemoryRouter basename="/ml-dl-visualization" initialEntries={["/ml-dl-visualization/admin/audit"]}>
      <Routes>
        <Route path="/admin/audit" element={<AuditLog />} />
        <Route path="/dashboard" element={<h1>Student progress destination</h1>} />
      </Routes>
    </MemoryRouter>,
  );
  fireEvent.click(screen.getByRole("button", { name: "學習行為" }));
  const link = screen.getByRole("link", { name: "學習儀表板" });
  expect(link).toHaveAttribute("href", "/ml-dl-visualization/dashboard");
  fireEvent.click(link);
  expect(await screen.findByRole("heading", { name: "Student progress destination" })).toBeInTheDocument();
});
