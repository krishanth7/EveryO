# How automatic differentiation works in EveryO

The engine is roughly 250 lines and is worth reading end to end
(`everyo/core/autograd.py`). This page explains the idea behind it.

## The computation graph

Every differentiable operation does three things:

1. computes the forward result,
2. captures whatever it needs for the backward pass in a closure,
3. attaches a `Node` recording the inputs and that closure.

```python
def multiply(a, b):
    left, right = _binary_inputs(a, b, "multiply")
    data = _dispatch("multiply", device, left.data, right.data)

    def backward_fn(gradient):
        return (
            unbroadcast(gradient * right.data, left.shape),
            unbroadcast(gradient * left.data, right.shape),
        )

    return _make(data, (left, right), "multiply", backward_fn)
```

The closure implements the product rule: `d(ab)/da = b` and `d(ab)/db = a`,
each multiplied by the incoming gradient (the chain rule).

Nodes are created only when gradient mode is enabled *and* at least one input
requires gradients. Inference therefore builds no graph at all.

## The backward pass

`Tensor.backward()` seeds the output gradient (ones, for a scalar), walks the
graph in reverse topological order and accumulates gradients into each tensor's
`.grad`.

```python
for node in topological_order(root):
    incoming = gradients.get(id(node))
    if node.requires_grad and (node.is_leaf or node.retains_grad):
        node.accumulate_grad(incoming)
    for parent, parent_grad in zip(node.grad_node.parents, node.grad_node.backward_fn(incoming)):
        gradients[id(parent)] = gradients.get(id(parent), 0) + parent_grad
```

Two details matter:

* **The traversal is iterative.** A recursive walk would hit Python's
  recursion limit on deep graphs; `tests/test_autograd.py` checks a
  2000-operation chain.
* **Gradients are summed, not replaced.** When a tensor feeds several
  operations, every path contributes. That is why `x + x` gives `dx = 2`, and
  why you must call `zero_grad()` between steps unless you *want* accumulation.

## Broadcasting

NumPy broadcasting means a `(3,)` bias can be added to a `(32, 3)` activation.
The gradient then arrives with the broadcast shape `(32, 3)` and must be summed
back down to `(3,)` — summing over the expanded axes is exactly the derivative
of broadcasting. `unbroadcast()` does this, and every broadcasting operation
routes its gradients through it.

## Only leaves keep gradients

Intermediate tensors do not retain `.grad`, which keeps memory bounded. Call
`.retain_grad()` on one if you want to inspect it:

```python
hidden = (x * 3).retain_grad()
(hidden * 2).backward()
print(hidden.grad)  # [2.]
```

## Turning it off

```python
with eo.no_grad():
    predictions = model(features)  # no graph is built
```

`no_grad` is thread-local, works as a decorator, and restores the previous mode
on exit. `enable_grad()` re-enables tracking inside a `no_grad` block.

## Supported operations

Gradients are implemented and finite-difference tested for: add, subtract,
multiply, divide, negate, power, exp, log, sqrt, abs, clip, matmul (including
the vector cases), dot, sum, mean, max, min, variance, reshape, transpose,
flatten, concatenate, stack, indexing, ReLU, sigmoid, tanh, softmax and
log-softmax.

## How correctness is verified

For a scalar function `f`, the central difference

```
(f(x + h) - f(x - h)) / 2h
```

approximates `df/dx` with error O(h²). `tests/test_autograd.py` compares every
analytic gradient against this approximation in `float64`. If a backward pass
were wrong, the comparison would fail — which is exactly what happened, and got
caught, while this engine was being written.

## What is not implemented in v0.1.0

* Second-order derivatives (the graph is not differentiable itself).
* In-place operations that would invalidate saved tensors.
* Gradients through integer or boolean tensors (these cannot require grad).
