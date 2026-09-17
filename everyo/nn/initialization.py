"""Parameter initialisation schemes.

Good initialisation keeps activation variance roughly constant as signals move
through a network.  EveryO ships the two schemes that cover almost all of the
layers implemented in v0.1.0:

* Xavier/Glorot — for symmetric activations such as tanh and sigmoid.
* He/Kaiming — for ReLU-style activations, which halve the variance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from everyo.exceptions import EveryOShapeError

__all__ = [
    "compute_fans",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "zeros_init",
    "ones_init",
    "uniform_init",
    "normal_init",
    "get_initializer",
]


def compute_fans(shape: Sequence[int]) -> tuple[int, int]:
    """Return ``(fan_in, fan_out)`` for a parameter of the given shape.

    Args:
        shape: Parameter shape. A 2-D weight is interpreted as
            ``(fan_in, fan_out)``, matching EveryO's ``x @ W`` convention; a
            4-D weight is a convolution kernel ``(kh, kw, in, out)``.

    Raises:
        EveryOShapeError: If the shape has no dimensions.
    """
    dims = tuple(int(d) for d in shape)
    if not dims:
        raise EveryOShapeError("Cannot compute fan-in/fan-out for a 0-dimensional parameter.")
    if len(dims) == 1:
        return dims[0], dims[0]
    if len(dims) == 4:
        # A convolution kernel is (kh, kw, in_channels, out_channels): every
        # output unit sees kh * kw * in_channels inputs, so the receptive field
        # scales both fans rather than only one.
        receptive = dims[0] * dims[1]
        return dims[2] * receptive, dims[3] * receptive
    receptive = int(np.prod(dims[2:])) if len(dims) > 2 else 1
    return dims[0] * receptive, dims[1] * receptive


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def xavier_uniform(shape: Sequence[int], *, seed=None, gain: float = 1.0) -> np.ndarray:
    """Glorot uniform initialisation, ``U(-limit, limit)``."""
    fan_in, fan_out = compute_fans(shape)
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return _rng(seed).uniform(-limit, limit, size=tuple(shape))


def xavier_normal(shape: Sequence[int], *, seed=None, gain: float = 1.0) -> np.ndarray:
    """Glorot normal initialisation."""
    fan_in, fan_out = compute_fans(shape)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return _rng(seed).normal(0.0, std, size=tuple(shape))


def he_uniform(shape: Sequence[int], *, seed=None, gain: float = 1.0) -> np.ndarray:
    """He uniform initialisation, tuned for ReLU networks."""
    fan_in, _ = compute_fans(shape)
    limit = gain * np.sqrt(6.0 / fan_in)
    return _rng(seed).uniform(-limit, limit, size=tuple(shape))


def he_normal(shape: Sequence[int], *, seed=None, gain: float = 1.0) -> np.ndarray:
    """He normal initialisation, tuned for ReLU networks."""
    fan_in, _ = compute_fans(shape)
    std = gain * np.sqrt(2.0 / fan_in)
    return _rng(seed).normal(0.0, std, size=tuple(shape))


def zeros_init(shape: Sequence[int], *, seed=None) -> np.ndarray:
    """Fill with zeros (the default for biases)."""
    return np.zeros(tuple(shape))


def ones_init(shape: Sequence[int], *, seed=None) -> np.ndarray:
    """Fill with ones."""
    return np.ones(tuple(shape))


def uniform_init(
    shape: Sequence[int], *, seed=None, low: float = -0.05, high: float = 0.05
) -> np.ndarray:
    """Sample uniformly from ``[low, high)``."""
    return _rng(seed).uniform(low, high, size=tuple(shape))


def normal_init(
    shape: Sequence[int], *, seed=None, mean: float = 0.0, std: float = 0.05
) -> np.ndarray:
    """Sample from a normal distribution."""
    return _rng(seed).normal(mean, std, size=tuple(shape))


#: Names accepted by layers that take an ``initializer=`` argument.
INITIALIZERS = {
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "glorot_uniform": xavier_uniform,
    "glorot_normal": xavier_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
    "kaiming_uniform": he_uniform,
    "kaiming_normal": he_normal,
    "zeros": zeros_init,
    "ones": ones_init,
    "uniform": uniform_init,
    "normal": normal_init,
}


def get_initializer(name: str):
    """Look up an initialiser by name.

    Raises:
        EveryOShapeError: Never; unknown names raise :class:`ValueError` with the
            list of valid options.
    """
    key = str(name).lower()
    if key not in INITIALIZERS:
        raise ValueError(
            f"Unknown initializer {name!r}. Available initializers: "
            f"{', '.join(sorted(INITIALIZERS))}."
        )
    return INITIALIZERS[key]
