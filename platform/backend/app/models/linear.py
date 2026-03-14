import numpy as np
from pydantic import BaseModel


class GradientDescentResult(BaseModel):
    weights_history: list[list[float]]
    loss_history: list[float]
    final_weights: list[float]
    final_loss: float


def run_gradient_descent(
    X: list[list[float]],
    y: list[float],
    learning_rate: float = 0.01,
    epochs: int = 100,
) -> GradientDescentResult:
    X_arr = np.array(X)
    y_arr = np.array(y)
    n, d = X_arr.shape
    w = np.zeros(d)
    b = 0.0
    weights_history = []
    loss_history = []

    for _ in range(epochs):
        pred = X_arr @ w + b
        error = pred - y_arr
        loss = float(np.mean(error**2))
        loss_history.append(loss)
        weights_history.append(w.tolist() + [b])
        grad_w = (2 / n) * (X_arr.T @ error)
        grad_b = (2 / n) * np.sum(error)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return GradientDescentResult(
        weights_history=weights_history,
        loss_history=loss_history,
        final_weights=w.tolist() + [b],
        final_loss=loss_history[-1],
    )


def compute_loss_landscape(
    X: list[list[float]],
    y: list[float],
    w0_range: tuple = (-5, 5),
    w1_range: tuple = (-5, 5),
    resolution: int = 50,
    surface_type: str = "bowl",
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)
    w0s = np.linspace(*w0_range, resolution)
    w1s = np.linspace(*w1_range, resolution)
    Z = np.zeros((resolution, resolution))
    for i, w0 in enumerate(w0s):
        for j, w1 in enumerate(w1s):
            if surface_type == "saddle":
                Z[i, j] = w0**2 - w1**2
            elif surface_type == "local_minima":
                Z[i, j] = ((w0**2 + w1 - 11) ** 2 + (w0 + w1**2 - 7) ** 2) / 100
            else:  # "bowl" — default MSE loss
                pred = X_arr[:, 0] * w0 + w1
                Z[i, j] = float(np.mean((pred - y_arr) ** 2))
    return {"w0": w0s.tolist(), "w1": w1s.tolist(), "loss": Z.tolist()}
