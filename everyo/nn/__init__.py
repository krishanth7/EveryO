"""Neural network building blocks: modules, layers, activations and losses."""

from __future__ import annotations

from everyo.nn.activations import (
    LogSoftmax,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    log_softmax,
    relu,
    sigmoid,
    softmax,
    tanh,
)
from everyo.nn.initialization import (
    he_normal,
    he_uniform,
    xavier_normal,
    xavier_uniform,
)
from everyo.nn.layers import Dropout, Flatten, Linear
from everyo.nn.losses import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Loss,
    MAELoss,
    MSELoss,
    binary_cross_entropy,
    cross_entropy,
    mae_loss,
    mse_loss,
)
from everyo.nn.module import MODULE_REGISTRY, Module, Parameter, register_module
from everyo.nn.sequential import Sequential

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "Dropout",
    "Flatten",
    "Linear",
    "Loss",
    "LogSoftmax",
    "MAELoss",
    "MODULE_REGISTRY",
    "MSELoss",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "binary_cross_entropy",
    "cross_entropy",
    "he_normal",
    "he_uniform",
    "log_softmax",
    "mae_loss",
    "mse_loss",
    "register_module",
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
    "xavier_normal",
    "xavier_uniform",
]
