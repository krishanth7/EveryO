# Architecture

EveryO is layered so that each level depends only on the ones below it. You can
read it bottom-up in an afternoon.

```
                everyo.cli            everyo.visualization
                     |                          |
        +------------+--------------------------+
        |
   everyo.training  (Trainer, callbacks, history, metrics)
        |
   everyo.optim     (SGD, Adam)          everyo.serialization  (.evo archives)
        |                                        |
   everyo.nn        (Module, layers, activations, losses)
        |
   everyo.core      (Tensor, operations, autograd, device, dtype)
        |
   everyo.backends  (numpy_backend, tensorflow_backend)  everyo.cuda
        |                                                    |
      NumPy                                     optional native extension
```

## Layers

### `everyo.core`

The foundation.

* `dtype.py` — the small set of supported NumPy dtypes, with `float32` as the
  default floating point type.
* `device.py` — the `Device` abstraction. Buffers always live in host memory;
  the device selects *where operations execute*. Requesting CUDA without a
  working extension degrades to CPU with a warning rather than failing.
* `tensor.py` — the `Tensor` type: data, device, gradient state, operators.
* `operations.py` — every differentiable operation. Each one validates its
  inputs, computes the forward result (dispatching to CUDA when the tensor is
  on a CUDA device and a kernel exists) and records a graph node.
* `autograd.py` — the reverse-mode engine: graph nodes, an iterative
  topological sort, gradient accumulation and the `no_grad` mode flag.
* `convolution.py` — convolution and pooling. These live apart from
  `operations.py` because they need windowing helpers nothing else uses, and
  because the file would otherwise dwarf every other module. The forward pass
  is im2col, which turns a convolution into a single matrix multiplication;
  the backward pass is two more multiplications plus a scatter-add back into
  the padded image.

### `everyo.backends`

`numpy_backend.py` is the reference implementation of every kernel EveryO
needs. `tensorflow_backend.py` is optional and used for cross-checking
numerics, for a Keras reference model and for benchmarks. Importing the
package never imports TensorFlow.

### `everyo.cuda`

`availability.py` answers "can we run CUDA kernels right now?" without ever
raising. `interface.py` mirrors the NumPy kernel signatures and falls back to
NumPy unless `strict=True` is passed. The kernels themselves live in the
top-level `cuda/` directory and are built separately.

### `everyo.nn`

`Module` owns parameters and sub-modules, and can describe itself as a plain
dictionary via `get_config()`. That description is what makes pickle-free
serialisation possible. Layers, activations and losses all build on it.

### `everyo.optim`

`Optimizer` holds the parameter list and per-parameter state keyed by object
identity. `SGD` and `Adam` implement `step()`.

### `everyo.data`

`Dataset` is a two-method contract (`__len__`, `__getitem__`). `DataLoader`
batches and shuffles it, with a fast path that slices whole arrays when the
dataset is an `ArrayDataset`. Preprocessing follows a `fit`/`transform` split so
that scaling statistics come from training data only.

### `everyo.training`

`Trainer` runs the loop; `Callback` observers can stop training, checkpoint,
log or reschedule the learning rate. `History` records one row per epoch and
exports to JSON or CSV.

### `everyo.serialization`

An `.evo` file is a ZIP archive holding `manifest.json` (metadata plus the
architecture as JSON) and `parameters.npz` (saved and loaded with
`allow_pickle=False`). Loading rebuilds the model by looking class names up in
a registry, so opening an untrusted file cannot execute code.

## Design decisions

**Buffers stay in host memory.** A CUDA device tag changes *where kernels run*,
not where data lives. Transfers happen inside each kernel call. This costs
bandwidth, and the benchmarks measure that cost honestly; in exchange the data
model stays simple enough to read.

**Images are `NHWC`.** `(batch, height, width, channels)` is TensorFlow's
native layout, and EveryO uses TensorFlow as its correctness reference — so
convolution results can be compared against `tf.nn.conv2d` without transposing
anything, which is exactly what the test suite does.

**`float32` by default.** Python lists and scalars become `float32`; a NumPy
array keeps its own precision, so `float64` work stays `float64`. Gradient
checks in the test suite use `float64` for accuracy.

**Gradients are recorded only when needed.** A node is created only when
gradient mode is on *and* at least one input requires gradients, so inference
costs nothing extra.

**Errors name the shapes.** `EveryOShapeError` messages state what was
attempted, which shapes were involved and what would have been valid.

**Optional means optional.** TensorFlow and CUDA are detected safely, report
why they are unavailable, and never prevent the core from working.

## Extending EveryO

To add a differentiable operation:

1. Implement the forward pass in `everyo/core/operations.py`.
2. Return the gradients for each input from the `backward_fn` closure.
3. Add a finite-difference check to `tests/test_autograd.py`.

To add a layer:

1. Subclass `Module`, decorate it with `@register_module` so it can be
   serialised, and implement `forward`.
2. Implement `get_config()` if the constructor takes arguments.
3. Export it from `everyo/nn/__init__.py` and the top-level `everyo/__init__.py`.
4. Add tests, including a save/load round trip.
