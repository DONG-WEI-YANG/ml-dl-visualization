import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import Dashboard from "../pages/Dashboard";

const mockFetchAPI = vi.fn();
vi.mock("../lib/api", () => ({
  fetchAPI: (...args: unknown[]) => mockFetchAPI(...args),
}));

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null, XAxis: () => null, YAxis: () => null,
  CartesianGrid: () => null, Tooltip: () => null, Legend: () => null,
}));

const MOCK_SUMMARY = {
  total_students: 30, total_events: 1500, average_score: 78.5,
  popular_llm_topics: [{ topic: "梯度下降", count: 45 }],
};

const MOCK_STUDENT = {
  student_id: "S001", total_weeks_completed: 10, total_time_minutes: 450, average_score: 82.3,
  weekly_progress: [{ week: 1, completed: true, quiz_score: 90, assignment_score: 85, llm_interactions: 5, time_spent_minutes: 30 }],
  llm_topics: [{ topic: "過擬合", count: 3 }],
  error_patterns: [{ type: "TypeError", count: 2 }],
};

describe("Dashboard", () => {
  beforeEach(() => { mockFetchAPI.mockReset(); });

  it("shows loading state", () => {
    mockFetchAPI.mockReturnValue(new Promise(() => {}));
    render(<Dashboard />);
    expect(screen.getByText("載入中...")).toBeInTheDocument();
  });

  it("displays class summary", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_SUMMARY);
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText("30")).toBeInTheDocument();
      expect(screen.getByText("1500")).toBeInTheDocument();
    });
  });

  it("shows error on fetch failure", async () => {
    mockFetchAPI.mockRejectedValueOnce(new Error("fail"));
    render(<Dashboard />);
    await waitFor(() => { expect(screen.getByText("無法載入班級總覽資料")).toBeInTheDocument(); });
  });

  it("looks up student analytics", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_SUMMARY);
    render(<Dashboard />);
    await waitFor(() => screen.getByText("30"));
    const input = screen.getByPlaceholderText("輸入學生 ID");
    fireEvent.change(input, { target: { value: "S001" } });
    mockFetchAPI.mockResolvedValueOnce(MOCK_STUDENT);
    fireEvent.click(screen.getByText("查詢"));
    await waitFor(() => { expect(screen.getByText("10 / 18")).toBeInTheDocument(); });
  });

  it("shows student not found error", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_SUMMARY);
    render(<Dashboard />);
    await waitFor(() => screen.getByText("30"));
    const input = screen.getByPlaceholderText("輸入學生 ID");
    fireEvent.change(input, { target: { value: "X" } });
    mockFetchAPI.mockRejectedValueOnce(new Error("404"));
    fireEvent.click(screen.getByText("查詢"));
    await waitFor(() => { expect(screen.getByText("找不到該學生資料")).toBeInTheDocument(); });
  });
});
