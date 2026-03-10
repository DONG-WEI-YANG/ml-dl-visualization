import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ChatPanel from "../components/llm/ChatPanel";

const mockSend = vi.fn();
const mockClear = vi.fn();
const mockOnClose = vi.fn();
const mockOnTogglePin = vi.fn();
const mockOnAutoOpen = vi.fn();
let mockMessages: { role: string; content: string }[] = [];
let mockIsLoading = false;

const defaultProps = {
  week: 1,
  topic: "test",
  pinned: false,
  onClose: mockOnClose,
  onTogglePin: mockOnTogglePin,
  onAutoOpen: mockOnAutoOpen,
};

vi.mock("../hooks/useChat", () => ({
  useChat: () => ({
    messages: mockMessages,
    isLoading: mockIsLoading,
    send: mockSend,
    clear: mockClear,
  }),
}));

describe("ChatPanel", () => {
  beforeEach(() => {
    mockMessages = [];
    mockIsLoading = false;
    mockSend.mockClear();
    mockClear.mockClear();
  });

  it("renders welcome message when no messages", () => {
    render(<ChatPanel {...defaultProps} week={1} topic="Python 環境" />);
    expect(screen.getByText("歡迎使用 AI 助教！")).toBeInTheDocument();
  });

  it("switches between tutor and homework modes", () => {
    render(<ChatPanel {...defaultProps} week={1} topic="Python 環境" />);
    const homeworkBtn = screen.getByText("作業");
    fireEvent.click(homeworkBtn);
    // Homework button should be active (has bg-orange-500 class)
    expect(homeworkBtn.className).toContain("bg-orange-500");
  });

  it("sends a message on form submit", () => {
    render(<ChatPanel {...defaultProps} week={4} topic="梯度下降" />);
    const input = screen.getByPlaceholderText("輸入你的問題...");
    fireEvent.change(input, { target: { value: "什麼是梯度下降？" } });
    fireEvent.submit(input.closest("form")!);
    expect(mockSend).toHaveBeenCalledWith("什麼是梯度下降？", "tutor");
  });

  it("does not send empty message", () => {
    render(<ChatPanel {...defaultProps} />);
    const input = screen.getByPlaceholderText("輸入你的問題...");
    fireEvent.submit(input.closest("form")!);
    expect(mockSend).not.toHaveBeenCalled();
  });

  it("clears messages on clear button", () => {
    mockMessages = [{ role: "user", content: "test" }];
    render(<ChatPanel {...defaultProps} />);
    fireEvent.click(screen.getByTitle("清除對話"));
    expect(mockClear).toHaveBeenCalled();
  });

  it("disables input when loading", () => {
    mockIsLoading = true;
    render(<ChatPanel {...defaultProps} />);
    expect(screen.getByPlaceholderText("輸入你的問題...")).toBeDisabled();
  });

  it("displays user and assistant messages", () => {
    mockMessages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ];
    render(<ChatPanel {...defaultProps} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("shows loading indicator", () => {
    mockIsLoading = true;
    render(<ChatPanel {...defaultProps} />);
    expect(screen.getByText("AI 助教思考中...")).toBeInTheDocument();
  });
});
