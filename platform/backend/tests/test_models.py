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


def test_loss_landscape_saddle():
    result = compute_loss_landscape(X=[[1], [2], [3]], y=[2, 4, 6], resolution=10, surface_type="saddle")
    assert "w0" in result and "w1" in result and "loss" in result
    assert len(result["loss"]) == 10


def test_loss_landscape_local_minima():
    result = compute_loss_landscape(X=[[1], [2], [3]], y=[2, 4, 6], resolution=10, surface_type="local_minima")
    assert len(result["loss"]) == 10


def test_decision_boundary_3d():
    import random
    random.seed(42)
    X_3d, y_3d = [], []
    for _ in range(25):
        X_3d.append([random.gauss(-1, 1), random.gauss(-1, 1), random.gauss(-1, 1)])
        y_3d.append(0)
        X_3d.append([random.gauss(1, 1), random.gauss(1, 1), random.gauss(1, 1)])
        y_3d.append(1)
    result = train_and_get_decision_boundary(X_3d, y_3d, n_features=3, resolution=10)
    assert "mesh_vertices" in result
    assert "mesh_faces" in result
    assert len(result["X"][0]) == 3
