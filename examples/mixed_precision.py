"""Train with autocast and dynamic loss scaling, and show why scaling matters.

The script does three things: trains the same model in float32 and in mixed
precision so you can compare where they land, measures how many gradient
elements flush to zero with and without a `GradScaler`, and prints the dtypes
so you can see that the parameters never leave float32.

A warning worth repeating: on a CPU this buys **memory, not speed**. NumPy has
no native float16 arithmetic and upcasts to compute, so the float16 path is
usually slower here. The speedup mixed precision is known for comes from GPU
tensor cores.

Run with:
    python examples/mixed_precision.py
"""

from __future__ import annotations

import numpy as np

import everyo as eo
from everyo.datasets import make_regression

STEPS = 400
#: Scaling the loss down by this much pushes the gradients below float16's
#: smallest subnormal (~6e-8), which is exactly the regime loss scaling exists
#: for. A real network reaches it through depth rather than a constant.
TINY_LOSS_WEIGHT = 1e-7


def build_model() -> eo.Sequential:
    """A small MLP, built from fixed seeds so every run is comparable."""
    return eo.Sequential(eo.Linear(4, 32, seed=0), eo.Tanh(), eo.Linear(32, 1, seed=1))


def train(
    inputs: eo.Tensor,
    targets: eo.Tensor,
    *,
    use_autocast: bool,
    use_scaler: bool,
    loss_weight: float = 1.0,
    lr: float = 0.05,
) -> float:
    """Train one model and return its final mean squared error."""
    model = build_model()
    optimizer = eo.SGD(model.parameters(), lr=lr / loss_weight)
    scaler = eo.GradScaler(init_scale=2.0**15, enabled=use_scaler)

    for _ in range(STEPS):
        optimizer.zero_grad()
        with eo.autocast(use_autocast):
            predictions = model(inputs)
        residual = predictions - targets
        loss = eo.mean(residual * residual) * loss_weight
        scaler.backward(loss)
        scaler.step(optimizer)
        scaler.update()

    with eo.no_grad():
        residual = model(inputs) - targets
        return float(eo.mean(residual * residual).item())


def count_flushed_gradients(*, use_scaler: bool) -> tuple[int, int]:
    """Return (zeroed, total) gradient elements after one tiny-loss backward."""
    model = build_model()
    optimizer = eo.SGD(model.parameters(), lr=1e-12)
    scaler = eo.GradScaler(init_scale=2.0**15, enabled=use_scaler)
    inputs, targets = load_data()

    optimizer.zero_grad()
    with eo.autocast():
        predictions = model(inputs)
    residual = predictions - targets
    scaler.backward(eo.mean(residual * residual) * TINY_LOSS_WEIGHT)
    scaler.unscale_(optimizer)

    gradients = np.concatenate([np.asarray(p.grad).ravel() for p in model.parameters()])
    return int((gradients == 0).sum()), gradients.size


def load_data() -> tuple[eo.Tensor, eo.Tensor]:
    """A small regression problem, fixed seed."""
    features, targets = make_regression(n_samples=64, n_features=4, noise=0.05, seed=0)
    return eo.tensor(features), eo.tensor(targets)


def main() -> None:
    inputs, targets = load_data()

    print("Dtypes under autocast")
    print("-" * 60)
    model = build_model()
    with eo.autocast():
        activation = model(inputs)
    eo.sum(activation).backward()
    print(f"  activation           : {activation.dtype}")
    for name, parameter in model.named_parameters():
        print(f"  {name:<21}: {parameter.dtype} (gradient {parameter.grad.dtype})")
    print("  Parameters stay in float32; only the activations drop to float16.")

    print()
    print("Gradient underflow at a very small loss")
    print("-" * 60)
    for use_scaler in (False, True):
        zeroed, total = count_flushed_gradients(use_scaler=use_scaler)
        label = "with GradScaler" if use_scaler else "no GradScaler  "
        print(f"  {label}: {zeroed}/{total} gradient elements flushed to zero")

    print()
    print(f"Training at a loss weight of {TINY_LOSS_WEIGHT:g}")
    print("-" * 60)
    without = train(
        inputs, targets, use_autocast=True, use_scaler=False, loss_weight=TINY_LOSS_WEIGHT
    )
    with_scaler = train(
        inputs, targets, use_autocast=True, use_scaler=True, loss_weight=TINY_LOSS_WEIGHT
    )
    print(f"  autocast, no scaler  : final MSE {without:.4f}")
    print(f"  autocast + GradScaler: final MSE {with_scaler:.4f}")

    print()
    print("At a normal loss scale, mixed precision simply matches float32")
    print("-" * 60)
    full = train(inputs, targets, use_autocast=False, use_scaler=False)
    mixed = train(inputs, targets, use_autocast=True, use_scaler=True)
    print(f"  float32              : final MSE {full:.4f}")
    print(f"  mixed precision      : final MSE {mixed:.4f}")

    print()
    print("On a CPU this saves memory, not time: NumPy upcasts float16 to compute.")


if __name__ == "__main__":
    main()
