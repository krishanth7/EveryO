"""Training loop, callbacks, metrics and history."""

from __future__ import annotations

from everyo.training.callbacks import (
    Callback,
    CallbackList,
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ProgressLogger,
)
from everyo.training.history import History
from everyo.training.metrics import (
    METRICS,
    accuracy,
    binary_accuracy,
    confusion_matrix,
    get_metric,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from everyo.training.trainer import Trainer

__all__ = [
    "CSVLogger",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "History",
    "LearningRateScheduler",
    "METRICS",
    "ModelCheckpoint",
    "ProgressLogger",
    "Trainer",
    "accuracy",
    "binary_accuracy",
    "confusion_matrix",
    "get_metric",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
]
