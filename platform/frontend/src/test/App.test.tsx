import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import App from "../App";

describe("App", () => {
  it("renders login page when unauthenticated", () => {
    render(<App />);
    expect(
      screen.getByText("ML/DL 視覺化教學平台")
    ).toBeInTheDocument();
    expect(screen.getByText("請登入以開始學習")).toBeInTheDocument();
  });

  it("renders login form with username and password fields", () => {
    render(<App />);
    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "登入" })).toBeInTheDocument();
  });
});
