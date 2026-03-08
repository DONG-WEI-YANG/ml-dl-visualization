import numpy as np


def get_activation_functions() -> dict:
    """返回各激活函數的計算結果，用於前端視覺化"""
    x = np.linspace(-5, 5, 200).tolist()
    x_arr = np.array(x)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def tanh(z):
        return np.tanh(z)

    def relu(z):
        return np.maximum(0, z)

    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    def gelu(z):
        return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

    return {
        "x": x,
        "sigmoid": {"y": sigmoid(x_arr).tolist(), "dy": (sigmoid(x_arr) * (1 - sigmoid(x_arr))).tolist()},
        "tanh": {"y": tanh(x_arr).tolist(), "dy": (1 - tanh(x_arr) ** 2).tolist()},
        "relu": {"y": relu(x_arr).tolist(), "dy": np.where(x_arr > 0, 1, 0).tolist()},
        "leaky_relu": {"y": leaky_relu(x_arr).tolist(), "dy": np.where(x_arr > 0, 1, 0.01).tolist()},
        "gelu": {"y": gelu(x_arr).tolist()},
    }
