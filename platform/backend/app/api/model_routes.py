from fastapi import APIRouter
from pydantic import BaseModel
from app.models.linear import run_gradient_descent, compute_loss_landscape
from app.models.classification import train_and_get_decision_boundary, get_roc_pr_curves
from app.models.tree import train_tree_model
from app.models.neural import get_activation_functions

router = APIRouter(prefix="/api/models", tags=["Models"])


class GradientDescentRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    learning_rate: float = 0.01
    epochs: int = 100
    surface_type: str = "bowl"


@router.post("/gradient-descent")
async def gradient_descent(req: GradientDescentRequest):
    return run_gradient_descent(req.X, req.y, req.learning_rate, req.epochs)


class LossLandscapeRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    w0_range: list[float] = [-5, 5]
    w1_range: list[float] = [-5, 5]
    resolution: int = 50
    surface_type: str = "bowl"


@router.post("/loss-landscape")
async def loss_landscape(req: LossLandscapeRequest):
    return compute_loss_landscape(
        req.X, req.y, tuple(req.w0_range), tuple(req.w1_range), req.resolution,
        surface_type=req.surface_type,
    )


class DecisionBoundaryRequest(BaseModel):
    X: list[list[float]]
    y: list[int]
    model_type: str = "logistic"
    C: float = 1.0
    kernel: str = "rbf"
    n_features: int = 2


@router.post("/decision-boundary")
async def decision_boundary(req: DecisionBoundaryRequest):
    return train_and_get_decision_boundary(
        req.X, req.y, req.model_type, req.C, req.kernel,
        n_features=req.n_features,
    )


class RocPrRequest(BaseModel):
    X: list[list[float]]
    y: list[int]
    model_type: str = "logistic"
    C: float = 1.0


@router.post("/roc-pr")
async def roc_pr(req: RocPrRequest):
    return get_roc_pr_curves(req.X, req.y, req.model_type, req.C)


class TreeModelRequest(BaseModel):
    X: list[list[float]]
    y: list[int]
    model_type: str = "decision_tree"
    max_depth: int = 5
    n_estimators: int = 100
    feature_names: list[str] | None = None


@router.post("/tree")
async def tree_model(req: TreeModelRequest):
    return train_tree_model(
        req.X, req.y, req.model_type, req.max_depth, req.n_estimators, req.feature_names
    )


@router.get("/activation-functions")
async def activation_functions():
    return get_activation_functions()
