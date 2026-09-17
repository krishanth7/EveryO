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
from everyo.nn.attention import (
    MultiHeadAttention,
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderBlock,
    causal_mask,
    padding_mask,
    scaled_dot_product_attention,
)
from everyo.nn.embedding import Embedding
from everyo.nn.initialization import (
    he_normal,
    he_uniform,
    xavier_normal,
    xavier_uniform,
)
from everyo.nn.layers import AvgPool2D, Conv2D, Dropout, Flatten, Linear, MaxPool2D
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
from everyo.nn.normalization import BatchNorm1D, BatchNorm2D, LayerNorm
from everyo.nn.recurrent import GRU, LSTM, RNN
from everyo.nn.sequential import Sequential

__all__ = [
    "BatchNorm1D",
    "BatchNorm2D",
    "Embedding",
    "GRU",
    "LSTM",
    "LayerNorm",
    "MultiHeadAttention",
    "PositionalEncoding",
    "RNN",
    "TransformerEncoder",
    "TransformerEncoderBlock",
    "causal_mask",
    "padding_mask",
    "scaled_dot_product_attention",
    "AvgPool2D",
    "BCELoss",
    "BCEWithLogitsLoss",
    "Conv2D",
    "CrossEntropyLoss",
    "Dropout",
    "Flatten",
    "Linear",
    "Loss",
    "LogSoftmax",
    "MAELoss",
    "MODULE_REGISTRY",
    "MaxPool2D",
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
