import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import QuizPanel from "../components/quiz/QuizPanel";

const mockFetchAPI = vi.fn();
vi.mock("../lib/api", () => ({
  fetchAPI: (...args: unknown[]) => mockFetchAPI(...args),
}));

const MOCK_QUESTIONS = {
  questions: [
    { id: "w01q1", question: "Python 中用來處理表格資料的主要套件是？", options: ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"] },
    { id: "w01q2", question: "Jupyter Notebook 的副檔名是？", options: [".py", ".ipynb", ".jnb", ".nb"] },
  ],
};

const MOCK_GRADE = {
  score: 1, total: 2, percentage: 50,
  results: [
    { id: "w01q1", correct: true, correct_answer: 1, user_answer: 1 },
    { id: "w01q2", correct: false, correct_answer: 1, user_answer: 0 },
  ],
};

describe("QuizPanel", () => {
  beforeEach(() => { mockFetchAPI.mockReset(); });

  it("loads and displays questions", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => {
      expect(screen.getByText(/Python 中用來處理表格資料/)).toBeInTheDocument();
    });
  });

  it("allows selecting answers", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => screen.getByText(/Python 中用來處理表格資料/));
    fireEvent.click(screen.getByText("Pandas"));
    expect(screen.getByText(/已作答 1\/2/)).toBeInTheDocument();
  });

  it("submits answers and shows results", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => screen.getByText(/Python 中用來處理表格資料/));
    fireEvent.click(screen.getByText("Pandas"));
    fireEvent.click(screen.getByText(".py"));
    mockFetchAPI.mockResolvedValueOnce(MOCK_GRADE);
    fireEvent.click(screen.getByText("提交答案"));
    await waitFor(() => { expect(screen.getByText(/1\/2/)).toBeInTheDocument(); });
  });

  it("shows reset button after grading", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => screen.getByText(/Python 中用來處理表格資料/));
    fireEvent.click(screen.getByText("Pandas"));
    fireEvent.click(screen.getByText(".py"));
    mockFetchAPI.mockResolvedValueOnce(MOCK_GRADE);
    fireEvent.click(screen.getByText("提交答案"));
    await waitFor(() => { expect(screen.getByText("重新作答")).toBeInTheDocument(); });
  });

  it("renders nothing when no questions", async () => {
    mockFetchAPI.mockResolvedValueOnce({ questions: [] });
    const { container } = render(<QuizPanel week={99} />);
    await waitFor(() => { expect(container.innerHTML).toBe(""); });
  });
});
