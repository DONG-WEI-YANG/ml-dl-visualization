import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import ErrorBoundary from "../components/ErrorBoundary";

function ThrowError(): React.ReactNode {
  throw new Error("Test error");
}

describe("ErrorBoundary", () => {
  it("catches errors and shows fallback UI", () => {
    // Suppress console.error for this test
    vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText("Test error")).toBeInTheDocument();
    expect(screen.getByText("Try Again")).toBeInTheDocument();

    vi.restoreAllMocks();
  });

  it("renders children when no error", () => {
    render(
      <ErrorBoundary>
        <div>Hello</div>
      </ErrorBoundary>
    );
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });
});
