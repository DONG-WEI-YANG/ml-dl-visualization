import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler


def train_and_get_decision_boundary(
    X: list[list[float]],
    y: list[int],
    model_type: str = "logistic",
    C: float = 1.0,
    kernel: str = "rbf",
    resolution: int = 100,
    n_features: int = 2,
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    if model_type == "logistic":
        model = LogisticRegression(C=C, max_iter=1000)
    else:
        model = SVC(C=C, kernel=kernel, probability=True)
    model.fit(X_scaled, y_arr)

    if n_features == 3:
        from skimage.measure import marching_cubes

        resolution = min(resolution, 30)

        mins = X_scaled.min(axis=0) - 1
        maxs = X_scaled.max(axis=0) + 1
        grid = np.mgrid[
            mins[0]:maxs[0]:complex(resolution),
            mins[1]:maxs[1]:complex(resolution),
            mins[2]:maxs[2]:complex(resolution),
        ]
        grid_points = grid.reshape(3, -1).T

        if hasattr(model, "decision_function"):
            values = model.decision_function(grid_points)
        else:
            values = model.predict_proba(grid_points)[:, 1] - 0.5

        volume = values.reshape(resolution, resolution, resolution)

        try:
            verts, faces, _, _ = marching_cubes(volume, level=0)
        except (ValueError, RuntimeError):
            # If marching cubes fails (e.g. no isosurface), return empty mesh
            return {
                "mesh_vertices": [],
                "mesh_faces": [],
                "X": X_scaled.tolist(),
                "y": y_arr.tolist(),
                "accuracy": float(model.score(X_scaled, y_arr)),
            }

        # Scale vertices back to data coordinates
        for dim in range(3):
            verts[:, dim] = verts[:, dim] / (resolution - 1) * (maxs[dim] - mins[dim]) + mins[dim]

        return {
            "mesh_vertices": verts.tolist(),
            "mesh_faces": faces.tolist(),
            "X": X_scaled.tolist(),
            "y": y_arr.tolist(),
            "accuracy": float(model.score(X_scaled, y_arr)),
        }

    # 2D case — existing behavior
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    return {
        "xx": xx.tolist(),
        "yy": yy.tolist(),
        "Z": Z.tolist(),
        "X": X_scaled.tolist(),
        "y": y_arr.tolist(),
        "accuracy": float(model.score(X_scaled, y_arr)),
    }


def get_roc_pr_curves(
    X: list[list[float]],
    y: list[int],
    model_type: str = "logistic",
    C: float = 1.0,
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)
    if model_type == "logistic":
        model = LogisticRegression(C=C, max_iter=1000)
    else:
        model = SVC(C=C, probability=True)
    model.fit(X_arr, y_arr)
    proba = model.predict_proba(X_arr)[:, 1]
    fpr, tpr, _ = roc_curve(y_arr, proba)
    precision, recall, _ = precision_recall_curve(y_arr, proba)
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist()},
    }
