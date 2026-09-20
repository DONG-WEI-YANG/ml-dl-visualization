from threading import BoundedSemaphore, Lock
from typing import Annotated, Literal

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from app.models.linear import run_gradient_descent, compute_loss_landscape
from app.models.classification import train_and_get_decision_boundary, get_roc_pr_curves
from app.models.tree import train_tree_model
from app.models.neural import get_activation_functions

router = APIRouter(prefix="/api/models", tags=["Models"])

# Per-process admission limit: no unbounded queue of expensive model jobs.
_compute_slots = BoundedSemaphore(2)
Number = Annotated[float, Field(allow_inf_nan=False, ge=-10000, le=10000)]
Row = Annotated[list[Number], Field(min_length=1, max_length=20)]
Matrix = Annotated[list[Row], Field(min_length=1, max_length=500)]
Targets = Annotated[list[Number], Field(min_length=1, max_length=500)]
Labels = Annotated[list[Annotated[int, Field(ge=0, le=19)]], Field(min_length=2, max_length=500)]
Surface = Literal["bowl", "saddle", "local_minima"]


async def _compute(function, *args, **kwargs):
    if not _compute_slots.acquire(blocking=False):
        raise HTTPException(503, "Model workers are busy; retry shortly", headers={"Retry-After": "1"})

    # Dispatch and cancellation transfer slot ownership under one lock. A
    # cancelled queued job must not execute later or return the slot twice.
    ownership = Lock()
    state = "queued"

    def execute():
        nonlocal state
        with ownership:
            if state == "cancelled":
                return None
            state = "running"
        try:
            return function(*args, **kwargs)
        finally:
            _compute_slots.release()

    try:
        return await anyio.to_thread.run_sync(execute)
    except BaseException as exc:
        with ownership:
            if state == "queued":
                state = "cancelled"
                _compute_slots.release()
        if isinstance(exc, (ValueError, FloatingPointError, OverflowError)):
            raise HTTPException(422, "Model input is invalid or computation diverged") from exc
        raise


class DatasetRequest(BaseModel):
    X: Matrix
    y: Targets

    @model_validator(mode="after")
    def validate_shape(self):
        if len(self.X) != len(self.y) or any(len(row) != len(self.X[0]) for row in self.X):
            raise ValueError("X must be rectangular and have one target per row")
        return self


class GradientDescentRequest(DatasetRequest):
    learning_rate: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.01
    epochs: int = Field(default=100, ge=1, le=1000)
    surface_type: Surface = "bowl"


@router.post("/gradient-descent")
async def gradient_descent(req: GradientDescentRequest):
    return await _compute(run_gradient_descent, req.X, req.y, req.learning_rate, req.epochs)


class LossLandscapeRequest(DatasetRequest):
    X: Matrix = [[1], [2], [3]]
    y: Targets = [2, 4, 6]
    w0_range: tuple[Number, Number] = (-5, 5)
    w1_range: tuple[Number, Number] = (-5, 5)
    resolution: int = Field(default=50, ge=2, le=100)
    surface_type: Surface = "bowl"

    @model_validator(mode="after")
    def validate_landscape(self):
        if len(self.X[0]) != 1:
            raise ValueError("Loss landscape requires one feature")
        if self.w0_range[0] >= self.w0_range[1] or self.w1_range[0] >= self.w1_range[1]:
            raise ValueError("Ranges must have increasing endpoints")
        return self


@router.post("/loss-landscape")
async def loss_landscape(req: LossLandscapeRequest):
    return await _compute(compute_loss_landscape, req.X, req.y, req.w0_range,
                          req.w1_range, req.resolution, surface_type=req.surface_type)


class BinaryRequest(DatasetRequest):
    y: Labels
    model_type: Literal["logistic", "svm"] = "logistic"
    C: Annotated[float, Field(gt=0, le=1000, allow_inf_nan=False)] = 1.0

    @model_validator(mode="after")
    def validate_classes(self):
        if set(self.y) != {0, 1}:
            raise ValueError("Both binary classes 0 and 1 are required")
        return self


class DecisionBoundaryRequest(BinaryRequest):
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    n_features: Literal[2, 3] = 2

    @model_validator(mode="after")
    def validate_features(self):
        if len(self.X[0]) != self.n_features:
            raise ValueError("X width must match n_features")
        return self


@router.post("/decision-boundary")
async def decision_boundary(req: DecisionBoundaryRequest):
    return await _compute(train_and_get_decision_boundary, req.X, req.y, req.model_type,
                          req.C, req.kernel, n_features=req.n_features)


class RocPrRequest(BinaryRequest):
    pass


@router.post("/roc-pr")
async def roc_pr(req: RocPrRequest):
    return await _compute(get_roc_pr_curves, req.X, req.y, req.model_type, req.C)


class TreeModelRequest(DatasetRequest):
    y: Labels
    model_type: Literal["decision_tree", "random_forest", "gradient_boosting"] = "decision_tree"
    max_depth: int = Field(default=5, ge=1, le=15)
    n_estimators: int = Field(default=100, ge=1, le=200)
    feature_names: list[Annotated[str, Field(min_length=1, max_length=80)]] | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def validate_tree(self):
        if len(set(self.y)) < 2:
            raise ValueError("At least two classes are required")
        if self.feature_names is not None and len(self.feature_names) != len(self.X[0]):
            raise ValueError("feature_names must match X width")
        return self


@router.post("/tree")
async def tree_model(req: TreeModelRequest):
    return await _compute(train_tree_model, req.X, req.y, req.model_type, req.max_depth,
                          req.n_estimators, req.feature_names)


@router.get("/activation-functions")
async def activation_functions():
    return await _compute(get_activation_functions)
