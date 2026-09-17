"""Optional TensorFlow integration.

TensorFlow plays three specific roles in EveryO, and no others:

1. **Reference numerics.** Operations implemented from scratch in
   :mod:`everyo.core.operations` can be checked against a mature
   implementation (see ``tests/test_tensorflow_backend.py``).
2. **Hardware-accelerated reference training.** :func:`train_reference_model`
   trains the same architecture with Keras, which is useful when comparing
   convergence or when a GPU is available through TensorFlow but not through
   EveryO's own CUDA extension.
3. **Benchmarks.** A well optimised baseline to measure EveryO against.

EveryO is *not* a TensorFlow wrapper: nothing in the core depends on this
module, and importing it never fails when TensorFlow is missing.
"""

from __future__ import annotations

import functools
import os
from types import ModuleType
from typing import Any, Sequence

import numpy as np

from everyo._logging import get_logger
from everyo.exceptions import EveryOBackendError

__all__ = [
    "NAME",
    "get_tensorflow",
    "is_available",
    "unavailable_reason",
    "describe",
    "require",
    "to_tensorflow",
    "from_tensorflow",
    "list_devices",
    "gpu_available",
    "build_keras_model",
    "train_reference_model",
    "matmul",
    "relu",
]

NAME = "tensorflow"
_LOGGER = get_logger(__name__)
_UNAVAILABLE_REASON: str | None = "TensorFlow has not been probed yet."

#: Keras activation name for each EveryO activation module.
_ACTIVATION_NAMES = {
    "ReLU": "relu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softmax": "softmax",
}


@functools.lru_cache(maxsize=1)
def get_tensorflow() -> ModuleType | None:
    """Import TensorFlow once, returning ``None`` when it is unavailable."""
    global _UNAVAILABLE_REASON
    # Keep TensorFlow's own start-up logging quiet unless the user asked for it.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        _UNAVAILABLE_REASON = (
            f"TensorFlow is not installed ({exc}). Install it with "
            "'pip install everyo[tensorflow]' to enable this backend."
        )
        _LOGGER.debug("TensorFlow unavailable: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - broken installations
        _UNAVAILABLE_REASON = f"TensorFlow failed to import ({exc})."
        _LOGGER.warning("TensorFlow import failed: %s", exc)
        return None

    _UNAVAILABLE_REASON = None
    return tf


def is_available() -> bool:
    """Return ``True`` when TensorFlow can be used."""
    return get_tensorflow() is not None


def unavailable_reason() -> str | None:
    """Explain why TensorFlow is unavailable, or ``None`` when it is."""
    get_tensorflow()
    return _UNAVAILABLE_REASON


def require() -> ModuleType:
    """Return the TensorFlow module or raise a helpful error.

    Raises:
        EveryOBackendError: When TensorFlow is not importable.
    """
    module = get_tensorflow()
    if module is None:
        raise EveryOBackendError(
            f"This operation requires TensorFlow, which is unavailable: {unavailable_reason()}"
        )
    return module


def describe() -> dict[str, Any]:
    """Return a JSON-serialisable description of the TensorFlow backend."""
    module = get_tensorflow()
    if module is None:
        return {"name": NAME, "available": False, "reason": unavailable_reason()}
    return {
        "name": NAME,
        "available": True,
        "version": module.__version__,
        "gpu_available": gpu_available(),
        "devices": list_devices(),
    }


def list_devices() -> list[str]:
    """Return the names of devices TensorFlow can see (empty when unavailable)."""
    module = get_tensorflow()
    if module is None:
        return []
    try:
        return [device.name for device in module.config.list_physical_devices()]
    except Exception as exc:  # pragma: no cover - depends on the platform
        _LOGGER.debug("Could not list TensorFlow devices: %s", exc)
        return []


def gpu_available() -> bool:
    """Return ``True`` when TensorFlow reports at least one GPU."""
    module = get_tensorflow()
    if module is None:
        return False
    try:
        return len(module.config.list_physical_devices("GPU")) > 0
    except Exception:  # pragma: no cover - depends on the platform
        return False


def to_tensorflow(tensor: Any) -> Any:
    """Convert an EveryO tensor (or array) into a TensorFlow tensor."""
    tf = require()
    from everyo.core.tensor import Tensor

    array = tensor.data if isinstance(tensor, Tensor) else np.asarray(tensor)
    return tf.convert_to_tensor(array)


def from_tensorflow(value: Any) -> Any:
    """Convert a TensorFlow tensor into an EveryO tensor."""
    require()
    from everyo.core.tensor import Tensor

    return Tensor(np.asarray(value))


def matmul(a: Any, b: Any) -> np.ndarray:
    """Matrix product computed by TensorFlow, returned as a NumPy array."""
    tf = require()
    return np.asarray(tf.matmul(to_tensorflow(a), to_tensorflow(b)))


def relu(x: Any) -> np.ndarray:
    """ReLU computed by TensorFlow, returned as a NumPy array."""
    tf = require()
    return np.asarray(tf.nn.relu(to_tensorflow(x)))


def build_keras_model(model: Any, input_shape: Sequence[int]) -> Any:
    """Build the Keras equivalent of an EveryO :class:`Sequential` model.

    Only the layer types EveryO v0.1.0 implements are translated.  The weights
    are *not* copied: the result is a freshly initialised reference model.

    Args:
        model: An :class:`everyo.nn.sequential.Sequential` instance.
        input_shape: Shape of one sample, without the batch dimension.

    Raises:
        EveryOBackendError: If TensorFlow is missing or a layer has no
            Keras equivalent.
    """
    tf = require()
    from everyo.nn.layers import AvgPool2D, Conv2D, Dropout, Flatten, Linear, MaxPool2D
    from everyo.nn.sequential import Sequential

    if not isinstance(model, Sequential):
        raise EveryOBackendError(
            f"build_keras_model() supports Sequential models, got {type(model).__name__}."
        )

    layers: list[Any] = [tf.keras.layers.Input(shape=tuple(input_shape))]
    for layer in model.layers:
        name = type(layer).__name__
        if isinstance(layer, Linear):
            layers.append(tf.keras.layers.Dense(layer.out_features, use_bias=layer.use_bias))
        elif isinstance(layer, Conv2D):
            # EveryO is NHWC, which is Keras's default data format, so the
            # arguments map across one to one.
            layers.append(
                tf.keras.layers.Conv2D(
                    filters=layer.out_channels,
                    kernel_size=tuple(layer.kernel_size),
                    strides=tuple(layer.stride),
                    padding=layer.padding if isinstance(layer.padding, str) else "valid",
                    use_bias=layer.use_bias,
                )
            )
        elif isinstance(layer, MaxPool2D):
            layers.append(
                tf.keras.layers.MaxPooling2D(
                    pool_size=tuple(layer.pool_size),
                    strides=None if layer.stride is None else tuple(layer.stride),
                    padding=layer.padding if isinstance(layer.padding, str) else "valid",
                )
            )
        elif isinstance(layer, AvgPool2D):
            layers.append(
                tf.keras.layers.AveragePooling2D(
                    pool_size=tuple(layer.pool_size),
                    strides=None if layer.stride is None else tuple(layer.stride),
                    padding=layer.padding if isinstance(layer.padding, str) else "valid",
                )
            )
        elif isinstance(layer, Flatten):
            layers.append(tf.keras.layers.Flatten())
        elif isinstance(layer, Dropout):
            layers.append(tf.keras.layers.Dropout(layer.p))
        elif name in _ACTIVATION_NAMES:
            layers.append(tf.keras.layers.Activation(_ACTIVATION_NAMES[name]))
        else:
            raise EveryOBackendError(
                f"No Keras equivalent is defined for the EveryO layer "
                f"'{name}'. Supported layers: Linear, Conv2D, MaxPool2D, "
                f"AvgPool2D, Flatten, Dropout, "
                f"{', '.join(sorted(_ACTIVATION_NAMES))}."
            )
    return tf.keras.Sequential(layers)


def train_reference_model(
    model: Any,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    loss: str = "sparse_categorical_crossentropy",
    metrics: Sequence[str] = ("accuracy",),
    verbose: int = 0,
) -> dict[str, list[float]]:
    """Train a Keras model with the same architecture and return its history.

    Useful as a sanity check: if EveryO's loss curve is far worse than the
    reference on the same data, something in the EveryO stack is wrong.
    """
    tf = require()
    keras_model = build_keras_model(model, features.shape[1:])
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=list(metrics),
    )
    history = keras_model.fit(
        features,
        targets,
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=int(verbose),
    )
    return {key: [float(v) for v in values] for key, values in history.history.items()}
