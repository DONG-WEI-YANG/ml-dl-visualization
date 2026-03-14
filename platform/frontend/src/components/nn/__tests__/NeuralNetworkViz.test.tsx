import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import NeuralNetworkViz from "../NeuralNetworkViz";

describe("NeuralNetworkViz", () => {
  it("renders with default config", () => {
    render(<NeuralNetworkViz />);
    expect(screen.getByText(/神經網路/)).toBeDefined();
  });
});
