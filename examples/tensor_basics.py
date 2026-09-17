"""Tensor creation, arithmetic and automatic differentiation.

Run with:
    python examples/tensor_basics.py
"""

from __future__ import annotations

import everyo as eo


def main() -> None:
    """Walk through the tensor API from creation to gradients."""
    print("=== Creating tensors ===")
    x = eo.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = eo.tensor([[5.0, 6.0], [7.0, 8.0]])
    print(f"x =\n{x}")
    print(f"shape={x.shape}  ndim={x.ndim}  dtype={x.dtype}  device={x.device}  size={x.size}")

    print("\n=== Element-wise arithmetic ===")
    print(f"x + y =\n{x + y}")
    print(f"x * y =\n{x * y}")
    print(f"x / 2 =\n{x / 2}")

    print("\n=== Matrix multiplication ===")
    print(f"x @ y =\n{eo.matmul(x, y)}")

    print("\n=== Broadcasting ===")
    row = eo.tensor([10.0, 20.0])
    print(f"x + [10, 20] =\n{x + row}")

    print("\n=== Reductions ===")
    print(f"sum       = {eo.sum(x).item()}")
    print(f"mean      = {eo.mean(x).item()}")
    print(f"max/axis0 = {eo.max(x, axis=0)}")
    print(f"column mean = {eo.mean(x, axis=0)}")

    print("\n=== Shape manipulation ===")
    print(f"reshape(4, 1) -> {x.reshape(4, 1).shape}")
    print(f"transpose     -> {x.T.shape}")
    print(f"flatten       -> {eo.flatten(x).shape}")

    print("\n=== Random tensors (seeded, so this output is reproducible) ===")
    print(f"normal(2, 3):\n{eo.normal((2, 3), seed=0)}")

    print("\n=== Automatic differentiation ===")
    a = eo.tensor([2.0], requires_grad=True)
    b = a * a
    b.backward()
    print(f"y = x^2 at x=2  ->  dy/dx = {a.grad}  (expected [4.])")

    w = eo.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    loss = eo.mean(eo.relu(eo.matmul(w, eo.tensor([[1.0], [1.0]]))))
    loss.backward()
    print(f"loss = mean(relu(W @ [1, 1])) = {loss.item():.4f}")
    print(f"dloss/dW =\n{w.grad}")

    print("\n=== Disabling gradient tracking ===")
    with eo.no_grad():
        inference = a * a
    print(f"inside no_grad(), requires_grad = {inference.requires_grad}")


if __name__ == "__main__":
    main()
