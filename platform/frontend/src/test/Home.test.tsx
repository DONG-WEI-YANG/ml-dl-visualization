import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Home from "../pages/Home";

describe("Home", () => {
  it("renders page title", () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(screen.getByText("ML/DL 視覺化互動教學平台")).toBeInTheDocument();
  });

  it("renders all 18 week cards", () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    const list = screen.getByRole("list", { name: /18-week curriculum/ });
    expect(list.children.length).toBe(18);
  });

  it("renders core and advanced labels", () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(screen.getAllByText("進階").length).toBeGreaterThan(0);
    expect(screen.getAllByText("核心").length).toBeGreaterThan(0);
  });

  it("has links to week pages", () => {
    render(<MemoryRouter><Home /></MemoryRouter>);
    const links = screen.getAllByRole("link");
    const weekLinks = links.filter((l) => l.getAttribute("href")?.startsWith("/week/"));
    expect(weekLinks.length).toBe(18);
  });
});
