import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
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
});
