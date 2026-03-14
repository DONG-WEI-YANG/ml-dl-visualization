import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import WeekPage from "../pages/WeekPage";

vi.mock("../hooks/useChat", () => ({
  useChat: () => ({ messages: [], isLoading: false, send: vi.fn(), clear: vi.fn() }),
}));

vi.mock("../lib/api", () => ({
  fetchAPI: vi.fn().mockResolvedValue({ questions: [] }),
  API_BASE: "http://localhost:8000",
}));

function renderWeekPage(weekId: string) {
  return render(
    <MemoryRouter initialEntries={[`/week/${weekId}`]}>
      <Routes>
        <Route path="/week/:weekId" element={<WeekPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("WeekPage", () => {
  it("renders week title for valid week", () => {
    renderWeekPage("4");
    expect(screen.getByText(/第 4 週/)).toBeInTheDocument();
    expect(screen.getByText(/線性回歸與梯度下降/)).toBeInTheDocument();
  });

  it("shows error for invalid week", () => {
    renderWeekPage("99");
    expect(screen.getByText(/找不到第 99 週的內容/)).toBeInTheDocument();
  });

  it("renders visualization section", () => {
    renderWeekPage("1");
    expect(screen.getByText("視覺化互動區")).toBeInTheDocument();
  });

  it("renders curriculum download links", () => {
    renderWeekPage("1");
    expect(screen.getByText("講義")).toBeInTheDocument();
    expect(screen.getByText("投影片")).toBeInTheDocument();
    expect(screen.getByText("Notebook")).toBeInTheDocument();
    expect(screen.getByText("作業")).toBeInTheDocument();
  });

  it("renders chat panel", () => {
    renderWeekPage("1");
    expect(screen.getByText("AI")).toBeInTheDocument();
    expect(screen.getByText("助教")).toBeInTheDocument();
  });
});
