import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ChatPanel from "../components/llm/ChatPanel";

const mockSend = vi.fn();
const mockClear = vi.fn();
let mockMessages: { role: string; content: string }[] = [];
let mockIsLoading = false;

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
    render(<ChatPanel week={1} topic="Python 環境" />);
    expect(screen.getByText("歡迎使用 AI 助教！")).toBeInTheDocument();
  });

  it("switches between tutor and homework modes", () => {
    render(<ChatPanel week={1} topic="Python 環境" />);
    const homeworkBtn = screen.getByText("作業模式");
    fireEvent.click(homeworkBtn);
    expect(screen.getByText("目前模式：作業模式")).toBeInTheDocument();
  });

  it("sends a message on form submit", () => {
    render(<ChatPanel week={4} topic="梯度下降" />);
    const input = screen.getByPlaceholderText("輸入你的問題...");
    fireEvent.change(input, { target: { value: "什麼是梯度下降？" } });
    fireEvent.submit(input.closest("form")!);
    expect(mockSend).toHaveBeenCalledWith("什麼是梯度下降？", "tutor");
  });

  it("does not send empty message", () => {
    render(<ChatPanel week={1} topic="test" />);
    const input = screen.getByPlaceholderText("輸入你的問題...");
    fireEvent.submit(input.closest("form")!);
    expect(mockSend).not.toHaveBeenCalled();
  });

  it("clears messages on clear button", () => {
    render(<ChatPanel week={1} topic="test" />);
    fireEvent.click(screen.getByText("清除"));
    expect(mockClear).toHaveBeenCalled();
  });

  it("disables input when loading", () => {
    mockIsLoading = true;
    render(<ChatPanel week={1} topic="test" />);
    expect(screen.getByPlaceholderText("輸入你的問題...")).toBeDisabled();
  });

  it("displays user and assistant messages", () => {
    mockMessages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ];
    render(<ChatPanel week={1} topic="test" />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("shows loading indicator", () => {
    mockIsLoading = true;
    render(<ChatPanel week={1} topic="test" />);
    expect(screen.getByText("AI 助教思考中...")).toBeInTheDocument();
  });
});
