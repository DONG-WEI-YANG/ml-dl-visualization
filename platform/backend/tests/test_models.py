from app.models.linear import run_gradient_descent, compute_loss_landscape
from app.models.classification import train_and_get_decision_boundary
from app.models.tree import train_tree_model
from app.models.neural import get_activation_functions


def test_gradient_descent():
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [2.0, 4.0, 6.0, 8.0]
    result = run_gradient_descent(X, y, learning_rate=0.01, epochs=100)
    assert len(result.loss_history) == 100
    assert result.final_loss < result.loss_history[0]


def test_loss_landscape():
    X = [[1.0], [2.0], [3.0]]
    y = [2.0, 4.0, 6.0]
    result = compute_loss_landscape(X, y, resolution=10)
    assert len(result["w0"]) == 10
    assert len(result["loss"]) == 10


def test_decision_boundary():
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [0, 1, 0, 1]
    result = train_and_get_decision_boundary(X, y)
    assert "accuracy" in result
    assert "Z" in result


def test_tree_model():
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [0, 1, 0, 1]
    result = train_tree_model(X, y)
    assert "accuracy" in result
    assert "feature_importances" in result


def test_activation_functions():
    result = get_activation_functions()
    assert "x" in result
    assert "sigmoid" in result
    assert "relu" in result
    assert len(result["x"]) == 200
