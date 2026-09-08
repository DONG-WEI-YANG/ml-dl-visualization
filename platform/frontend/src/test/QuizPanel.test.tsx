import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import QuizPanel from "../components/quiz/QuizPanel";

vi.mock("../hooks/useAuth", () => ({ useAuth: () => ({ token: "test-token" }) }));

const mockFetchAPI = vi.fn();
vi.mock("../lib/api", () => ({
  fetchAPI: (...args: unknown[]) => mockFetchAPI(...args),
}));

const MOCK_QUESTIONS = {
  questions: [
    { id: "w01q1", question: "Python 中用來處理表格資料的主要套件是？", options: ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"], category: "concept" },
    { id: "w01q2", question: "Jupyter Notebook 的副檔名是？", options: [".py", ".ipynb", ".jnb", ".nb"], category: "concept" },
  ],
};

const MOCK_GRADE = {
  score: 1, total: 2, percentage: 50,
  results: [
    { id: "w01q1", correct: true, correct_answer: 1, user_answer: 1, explanation: "Pandas 是處理表格資料的主要套件" },
    { id: "w01q2", correct: false, correct_answer: 1, user_answer: 0, explanation: "Jupyter Notebook 使用 .ipynb 副檔名" },
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
    await waitFor(() => { expect(screen.getByText(/1\/2 \(50%\)/)).toBeInTheDocument(); });
  });

  it("shows explanation for wrong answers", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await waitFor(() => screen.getByText(/Python 中用來處理表格資料/));
    fireEvent.click(screen.getByText("Pandas"));
    fireEvent.click(screen.getByText(".py"));
    mockFetchAPI.mockResolvedValueOnce(MOCK_GRADE);
    fireEvent.click(screen.getByText("提交答案"));
    await waitFor(() => { expect(screen.getByText(/Jupyter Notebook 使用 .ipynb/)).toBeInTheDocument(); });
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

  it("shows a recoverable message when questions cannot load", async () => {
    mockFetchAPI.mockRejectedValueOnce(new Error("offline"));

    render(<QuizPanel week={1} />);

    expect(await screen.findByRole("alert")).toHaveTextContent("無法載入測驗");
    expect(screen.getByRole("button", { name: "重新載入測驗" })).toBeInTheDocument();
  });

  it("shows a submit error without discarding selected answers", async () => {
    mockFetchAPI.mockResolvedValueOnce(MOCK_QUESTIONS);
    render(<QuizPanel week={1} />);
    await screen.findByText(/Python 中用來處理表格資料/);
    fireEvent.click(screen.getByText("Pandas"));
    fireEvent.click(screen.getByText(".py"));
    mockFetchAPI.mockRejectedValueOnce(new Error("server"));

    fireEvent.click(screen.getByText("提交答案"));

    expect(await screen.findByRole("alert")).toHaveTextContent("批改失敗");
    expect(screen.getByText(/已作答 2\/2/)).toBeInTheDocument();
  });
});
