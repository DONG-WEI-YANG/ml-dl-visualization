import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { beforeEach } from "vitest";
import { MemoryRouter } from "react-router-dom";
import Sidebar from "../components/Sidebar";

vi.mock("../hooks/useAuth", () => ({
  useAuth: () => ({
    user: { username: "admin", display_name: "管理員", role: "admin" },
    logout: vi.fn(),
  }),
}));

describe("Sidebar", () => {
  it("renders platform title", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.getByText("ML/DL 視覺化")).toBeInTheDocument();
  });

  it("renders 18 week navigation links", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    const links = screen.getAllByRole("link");
    const weekLinks = links.filter((l) => l.getAttribute("href")?.match(/\/week\/\d+/));
    expect(weekLinks.length).toBe(18);
  });

  it("shows dashboard link", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.getByText("學習分析")).toBeInTheDocument();
  });

  it("shows admin link for admin user", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.getByText("系統管理")).toBeInTheDocument();
  });

  it("shows user display name", () => {
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.getByText("管理員")).toBeInTheDocument();
  });

  describe("collapse", () => {
    beforeEach(() => localStorage.clear());

    it("collapses to icons and remembers the choice", () => {
      const { unmount } = render(<MemoryRouter><Sidebar /></MemoryRouter>);
      fireEvent.click(screen.getByRole("button", { name: "收合側邊欄" }));
      expect(screen.queryByText("ML/DL 視覺化")).not.toBeInTheDocument();
      // week links stay reachable by accessible name
      expect(screen.getByRole("link", { name: /第 1 週/ })).toBeInTheDocument();
      expect(screen.getByRole("link", { name: "帳號管理" })).toBeInTheDocument();
      expect(localStorage.getItem("sidebar-collapsed")).toBe("1");
      unmount();
      render(<MemoryRouter><Sidebar /></MemoryRouter>);
      expect(screen.getByRole("button", { name: "展開側邊欄" })).toHaveAttribute("aria-expanded", "false");
    });
  });

  describe("theme toggle", () => {
    beforeEach(() => {
      localStorage.clear();
      document.documentElement.classList.remove("dark");
    });

    it("cycles light -> dark -> system and toggles the dark class", () => {
      localStorage.setItem("theme-preference", "light");
      render(<MemoryRouter><Sidebar /></MemoryRouter>);
      expect(document.documentElement).not.toHaveClass("dark");
      fireEvent.click(screen.getByRole("button", { name: /主題：淺色/ }));
      expect(document.documentElement).toHaveClass("dark");
      expect(localStorage.getItem("theme-preference")).toBe("dark");
      fireEvent.click(screen.getByRole("button", { name: /主題：深色/ }));
      expect(screen.getByRole("button", { name: /主題：跟隨系統/ })).toBeInTheDocument();
    });
  });
});
