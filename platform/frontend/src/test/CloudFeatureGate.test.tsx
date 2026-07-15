import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import CloudFeatureGate from "../components/CloudFeatureGate";

describe("CloudFeatureGate", () => {
  it("shows a local retry card when cloud is unavailable", () => {
    const retry = vi.fn();
    render(<CloudFeatureGate available={false} title="AI 助教" onRetry={retry}><div>secret cloud feature</div></CloudFeatureGate>);
    expect(screen.queryByText("secret cloud feature")).not.toBeInTheDocument();
    expect(screen.getByText("AI 助教等待雲端服務")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "重新連線" }));
    expect(retry).toHaveBeenCalledOnce();
  });

  it("renders children when cloud is ready", () => {
    render(<CloudFeatureGate available title="測驗"><div>cloud feature</div></CloudFeatureGate>);
    expect(screen.getByText("cloud feature")).toBeInTheDocument();
  });
});
