"""Batch-generate real matplotlib/seaborn figures for the curriculum.

Run from repo root:
    python curriculum/_scripts/generate_figures.py

Outputs go to curriculum/week-XX/figures/*.png with deterministic seeds.
Fonts fall back to a Noto Sans CJK install if present; otherwise labels
are kept English-only to avoid mojibake on plain matplotlib installs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
CURRICULUM = ROOT / "curriculum"
SEED = 42

# Shared palette aligned with existing SVG assets (blue↔red)
PALETTE = {"blue": "#2563eb", "red": "#ef4444", "amber": "#d97706", "green": "#059669"}


def _week_dir(n: int) -> Path:
    d = CURRICULUM / f"week-{n:02d}" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pick_cjk_font() -> str | None:
    """Return the first available CJK font family, or None."""
    from matplotlib import font_manager
    candidates = [
        "Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans TC",
        "Heiti TC", "SimHei", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in installed:
            return c
    return None


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    cjk = _pick_cjk_font()
    if cjk:
        plt.rcParams["font.family"] = [cjk, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"[info] using CJK font: {cjk}")
    else:
        print("[info] no CJK font found; Chinese glyphs may render as boxes")


# --------------------------------------------------------------------------
# Week 2 — EDA
# --------------------------------------------------------------------------

def w02_pairplot() -> Path:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    g = sns.pairplot(df, vars=iris.feature_names, hue="species", palette="deep", height=1.6)
    g.fig.suptitle("Iris pairplot (Seaborn)", y=1.02)
    out = _week_dir(2) / "iris-pairplot.png"
    g.fig.savefig(out)
    plt.close(g.fig)
    return out


def w02_histogram_compare() -> Path:
    rng = np.random.default_rng(SEED)
    data_normal = rng.normal(loc=0, scale=1, size=1000)
    data_skew = rng.gamma(2, 1, size=1000)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(data_normal, bins=30, color=PALETTE["blue"], edgecolor="white")
    axes[0].set_title("常態分布 Normal (skew ≈ 0)")
    axes[1].hist(data_skew, bins=30, color=PALETTE["red"], edgecolor="white")
    axes[1].set_title("右偏分布 Right-skewed Gamma")
    for ax in axes:
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.tight_layout()
    out = _week_dir(2) / "histogram-skew-compare.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Week 4 — Linear Regression + Gradient Descent
# --------------------------------------------------------------------------

def w04_linreg_fit() -> Path:
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0, 10, size=50).reshape(-1, 1)
    y = 2.5 * X.ravel() + 4 + rng.normal(0, 2, size=50)
    model = LinearRegression().fit(X, y)
    xs = np.linspace(0, 10, 100).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, y, color=PALETTE["blue"], s=30, alpha=0.75, label="樣本 samples")
    ax.plot(xs, model.predict(xs), color=PALETTE["red"], lw=2,
            label=f"fit: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("線性迴歸擬合 Linear Regression Fit")
    ax.legend()
    out = _week_dir(4) / "linreg-fit.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def w04_gradient_descent_path() -> Path:
    """Contour of MSE on a 2-param linear model + GD path."""
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0, 10, 50)
    y_true = 3 * X + 5 + rng.normal(0, 2, 50)
    # Loss surface over (w, b)
    ws = np.linspace(-1, 7, 80)
    bs = np.linspace(-2, 12, 80)
    W, B = np.meshgrid(ws, bs)
    L = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            pred = W[i, j] * X + B[i, j]
            L[i, j] = np.mean((pred - y_true) ** 2)
    # GD path
    w, b, lr = -0.5, -1.0, 0.01
    path = [(w, b)]
    for _ in range(80):
        pred = w * X + b
        gw = 2 * np.mean((pred - y_true) * X)
        gb = 2 * np.mean(pred - y_true)
        w -= lr * gw
        b -= lr * gb
        path.append((w, b))
    path = np.array(path)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cs = ax.contour(W, B, L, levels=25, cmap="viridis", alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    ax.plot(path[:, 0], path[:, 1], "o-", color=PALETTE["red"], ms=3, lw=1.2, label="GD 路徑")
    ax.scatter(path[0, 0], path[0, 1], color="black", s=60, marker="s", label="起點", zorder=5)
    ax.scatter(path[-1, 0], path[-1, 1], color=PALETTE["green"], s=80, marker="*", label="收斂", zorder=5)
    ax.set_xlabel("w (斜率)")
    ax.set_ylabel("b (截距)")
    ax.set_title("Gradient Descent 在 MSE 損失面上的軌跡")
    ax.legend()
    out = _week_dir(4) / "gd-path.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Week 5 — Logistic Regression Decision Boundary + ROC
# --------------------------------------------------------------------------

def w05_decision_boundary() -> Path:
    iris = load_iris()
    X = iris.data[:, :2]  # sepal length/width for 2D viz
    y = (iris.target == 0).astype(int)  # setosa vs rest
    clf = LogisticRegression().fit(X, y)
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300),
    )
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.55)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color=PALETTE["blue"], edgecolor="white", label="other")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color=PALETTE["red"], edgecolor="white", label="setosa")
    ax.set_xlabel("sepal length (cm)")
    ax.set_ylabel("sepal width (cm)")
    ax.set_title("Logistic Regression 決策邊界（黑線 = 0.5 等高線）")
    ax.legend()
    out = _week_dir(5) / "decision-boundary.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def w05_roc_curve() -> Path:
    rng = np.random.default_rng(SEED)
    n = 500
    # Two imperfect score distributions
    y = np.concatenate([np.zeros(n), np.ones(n)])
    scores = np.concatenate([rng.normal(0.35, 0.18, n), rng.normal(0.65, 0.18, n)])
    scores = np.clip(scores, 0, 1)
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color=PALETTE["red"], lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC 曲線")
    axes[0].legend()
    axes[1].plot(rec, prec, color=PALETTE["blue"], lw=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall 曲線")
    fig.tight_layout()
    out = _week_dir(5) / "roc-pr-curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Week 7 — Decision Tree visualization
# --------------------------------------------------------------------------

def w07_decision_tree() -> Path:
    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(iris.data, iris.target)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_tree(
        clf,
        ax=ax,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    ax.set_title("Decision Tree on Iris (max_depth=3)")
    out = _week_dir(7) / "decision-tree.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def w07_feature_importance() -> Path:
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(iris.data, iris.target)
    order = np.argsort(clf.feature_importances_)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(np.array(iris.feature_names)[order], clf.feature_importances_[order], color=PALETTE["amber"])
    ax.set_xlabel("importance")
    ax.set_title("Random Forest 特徵重要度 (Iris)")
    out = _week_dir(7) / "feature-importance.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Week 10 — Validation & Learning Curves
# --------------------------------------------------------------------------

def w10_validation_curve() -> Path:
    data = load_diabetes()
    from sklearn.ensemble import RandomForestRegressor
    depths = np.arange(1, 15)
    train_s, val_s = validation_curve(
        RandomForestRegressor(n_estimators=80, random_state=SEED),
        data.data, data.target,
        param_name="max_depth", param_range=depths,
        scoring="r2", cv=5, n_jobs=1,
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(depths, train_s.mean(axis=1), "o-", color=PALETTE["blue"], label="train R²")
    ax.fill_between(depths, train_s.mean(1) - train_s.std(1), train_s.mean(1) + train_s.std(1),
                    color=PALETTE["blue"], alpha=0.15)
    ax.plot(depths, val_s.mean(axis=1), "o-", color=PALETTE["red"], label="val R²")
    ax.fill_between(depths, val_s.mean(1) - val_s.std(1), val_s.mean(1) + val_s.std(1),
                    color=PALETTE["red"], alpha=0.15)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("R²")
    ax.set_title("Validation Curve: RF on Diabetes")
    ax.legend()
    out = _week_dir(10) / "validation-curve.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def w10_learning_curve() -> Path:
    data = load_diabetes()
    from sklearn.ensemble import RandomForestRegressor
    sizes, train_s, val_s = learning_curve(
        RandomForestRegressor(n_estimators=80, max_depth=8, random_state=SEED),
        data.data, data.target,
        train_sizes=np.linspace(0.1, 1.0, 8), cv=5, scoring="r2", n_jobs=1, random_state=SEED,
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sizes, train_s.mean(1), "o-", color=PALETTE["blue"], label="train R²")
    ax.plot(sizes, val_s.mean(1), "o-", color=PALETTE["red"], label="val R²")
    ax.fill_between(sizes, train_s.mean(1) - train_s.std(1), train_s.mean(1) + train_s.std(1),
                    color=PALETTE["blue"], alpha=0.15)
    ax.fill_between(sizes, val_s.mean(1) - val_s.std(1), val_s.mean(1) + val_s.std(1),
                    color=PALETTE["red"], alpha=0.15)
    ax.set_xlabel("training samples")
    ax.set_ylabel("R²")
    ax.set_title("Learning Curve: RF on Diabetes")
    ax.legend()
    out = _week_dir(10) / "learning-curve.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    _style()
    tasks = [
        ("w02 pairplot", w02_pairplot),
        ("w02 histogram", w02_histogram_compare),
        ("w04 linreg", w04_linreg_fit),
        ("w04 GD path", w04_gradient_descent_path),
        ("w05 decision boundary", w05_decision_boundary),
        ("w05 ROC/PR", w05_roc_curve),
        ("w07 decision tree", w07_decision_tree),
        ("w07 feature importance", w07_feature_importance),
        ("w10 validation curve", w10_validation_curve),
        ("w10 learning curve", w10_learning_curve),
    ]
    for name, fn in tasks:
        out = fn()
        print(f"[ok] {name:<30} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
