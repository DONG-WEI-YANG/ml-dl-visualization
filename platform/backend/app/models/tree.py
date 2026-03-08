import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def train_tree_model(
    X: list[list[float]],
    y: list[int],
    model_type: str = "decision_tree",
    max_depth: int = 5,
    n_estimators: int = 100,
    feature_names: list[str] | None = None,
) -> dict:
    X_arr = np.array(X)
    y_arr = np.array(y)

    if model_type == "decision_tree":
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == "random_forest":
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    else:
        model = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)

    model.fit(X_arr, y_arr)
    result = {
        "accuracy": float(model.score(X_arr, y_arr)),
        "feature_importances": model.feature_importances_.tolist(),
    }
    if model_type == "decision_tree":
        result["tree_text"] = export_text(model, feature_names=feature_names)
    return result
