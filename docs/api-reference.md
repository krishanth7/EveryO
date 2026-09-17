# API reference

Everything listed here is importable from the top-level `everyo` package.
Docstrings in the source are the authoritative reference; this page is a map.

## Tensors

| Name | Purpose |
| --- | --- |
| `eo.tensor(data, *, dtype, device, requires_grad)` | Create a tensor |
| `eo.as_tensor(data)` | Convert only when needed |
| `eo.is_tensor(obj)` | Type check |
| `eo.zeros`, `eo.ones`, `eo.full`, `eo.eye` | Filled tensors |
| `eo.zeros_like`, `eo.ones_like` | Match an existing tensor |
| `eo.arange`, `eo.linspace` | Ranges |
| `eo.uniform`, `eo.normal`, `eo.rand`, `eo.randn` | Random tensors (accept `seed=`) |
| `eo.one_hot(indices, num_classes)` | One-hot encoding |

Tensor properties: `.shape`, `.ndim`, `.size`, `.dtype`, `.device`,
`.requires_grad`, `.is_leaf`, `.grad`, `.T`.

Tensor methods: `.backward()`, `.zero_grad()`, `.retain_grad()`, `.detach()`,
`.numpy()`, `.tolist()`, `.item()`, `.astype()`, `.to()`, `.cpu()`, `.cuda()`,
`.clone()`, `.reshape()`, `.transpose()`, `.flatten()`, `.sum()`, `.mean()`,
`.max()`, `.min()`, `.argmax()`, `.argmin()`, `.exp()`, `.log()`, `.sqrt()`,
`.abs()`, `.clip()`, `.matmul()`, `.dot()`.

## Operations

Arithmetic: `add`, `subtract`, `multiply`, `divide`, `negative`, `power`,
`exp`, `log`, `sqrt`, `abs`, `clip`.

Linear algebra: `matmul`, `dot`.

Reductions: `sum`, `mean`, `max`, `min`, `variance`.

Shape: `reshape`, `transpose`, `flatten`, `concatenate`, `stack`.

Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`.

## Gradient mode

`eo.no_grad()`, `eo.enable_grad()`, `eo.set_grad_enabled(mode)` — all usable as
context managers or decorators.

## Layers and models

| Name | Purpose |
| --- | --- |
| `eo.Module` | Base class for everything with parameters |
| `eo.Parameter` | A tensor an optimizer updates |
| `eo.Sequential(*layers)` | Chain layers |
| `eo.Linear(in_features, out_features, *, bias, initializer, seed)` | Affine layer |
| `eo.Flatten(start_dim=1)` | Collapse trailing dimensions |
| `eo.Dropout(p, *, seed)` | Inverted dropout |
| `eo.ReLU`, `eo.Sigmoid`, `eo.Tanh`, `eo.Softmax`, `eo.LogSoftmax` | Activation modules |

Module methods: `parameters()`, `named_parameters()`, `named_modules()`,
`num_parameters()`, `zero_grad()`, `train()`, `eval()`, `to(device)`,
`state_dict()`, `load_state_dict()`, `get_config()`, `summary()`.

## Losses

`eo.MSELoss`, `eo.MAELoss`, `eo.BCELoss`, `eo.BCEWithLogitsLoss`,
`eo.CrossEntropyLoss` — all accept `reduction` in `{"mean", "sum", "none"}`.
Functional forms live in `everyo.nn`: `mse_loss`, `mae_loss`,
`binary_cross_entropy`, `cross_entropy`.

`CrossEntropyLoss` takes raw logits and applies log-softmax internally, so the
final layer should be a plain `Linear`.

## Optimizers

| Name | Arguments |
| --- | --- |
| `eo.SGD` | `lr`, `momentum`, `weight_decay`, `nesterov` |
| `eo.Adam` | `lr`, `betas`, `eps`, `weight_decay`, `amsgrad` |

Both provide `zero_grad()`, `step()` and `get_config()`.

## Data

`eo.Dataset`, `eo.ArrayDataset`, `eo.TransformDataset`, `eo.Subset`,
`eo.load_csv_dataset`, `eo.DataLoader`.

Preprocessing: `eo.StandardScaler`, `eo.MinMaxScaler`, `eo.standardize`,
`eo.one_hot_encode`; `everyo.data` additionally exports `normalize`,
`min_max_scale` and `shuffle_arrays`.

Splitting: `eo.train_test_split`, `eo.random_split`, `eo.stratified_split`.

## Training

`eo.Trainer(model, optimizer, loss_fn, *, metrics, device, gradient_clip)` with
`fit()`, `evaluate()`, `predict()` and `predict_classes()`.

Callbacks: `eo.EarlyStopping`, `eo.ModelCheckpoint`, `eo.ProgressLogger`,
`eo.CSVLogger`, `eo.LearningRateScheduler`, and `eo.Callback` to write your own.

Metrics: `eo.accuracy`, `eo.confusion_matrix`; `everyo.training` also exports
`binary_accuracy`, `mean_absolute_error`, `mean_squared_error` and `r2_score`.

`eo.History` supports `history["loss"]`, `.best()`, `.last()`, `.to_dict()`,
`.to_json()` and `.to_csv()`.

## Serialization

`eo.save(model, path, *, metadata, overwrite)`, `eo.load(path, *, model, strict)`
and `eo.inspect_archive(path)`.

## Visualization

`eo.plot_loss`, `eo.plot_accuracy`, `eo.plot_history`, `eo.plot_confusion_matrix`,
`eo.plot_predictions`, `eo.plot_benchmark`; `everyo.visualization` also exports
`plot_metric`, `plot_decision_boundary` and `plot_benchmark_bars`. All accept
`save_path=` and work headlessly.

## Devices and CUDA

`eo.device(spec, *, strict)`, `eo.Device`, and the `eo.cuda` namespace:
`is_available()`, `device_count()`, `runtime_info()`, `unavailable_reason()`,
plus the kernels `add`, `multiply`, `matmul`, `relu`, `sum_all` (each taking
`strict=`).

## Datasets

`everyo.datasets`: `load_digits`, `render_digit`, `make_regression`,
`make_blobs`, `make_moons`, `make_spirals`, `make_xor`. All are generated
locally from a seed; nothing is downloaded.

## Configuration and logging

`eo.load_config(path)`, `eo.Config`, `eo.ModelConfig`, `eo.TrainingConfig`,
`eo.DeviceConfig`; `eo.configure_logging(level)` and `eo.get_logger(name)`.

## Exceptions

`EveryOError` is the base class. Subclasses: `EveryOShapeError`,
`EveryODeviceError`, `EveryODTypeError`, `EveryOGradientError`,
`EveryOSerializationError`, `EveryOConfigurationError`, `EveryOBackendError`,
`EveryOCudaError`.

## Command line

```
everyo info        # versions and available backends
everyo doctor      # diagnose the installation
everyo benchmark   # measure matmul across backends
everyo test        # run the bundled test suite
everyo demo        # train the bundled digit classifier
```
