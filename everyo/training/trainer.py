"""The training loop."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from everyo._logging import get_logger
from everyo.core.autograd import no_grad
from everyo.core.device import DeviceLike, resolve_device
from everyo.core.tensor import Tensor, as_tensor
from everyo.data.dataloader import DataLoader
from everyo.data.dataset import ArrayDataset
from everyo.exceptions import EveryOError
from everyo.nn.module import Module
from everyo.optim.optimizer import Optimizer
from everyo.training.callbacks import Callback, CallbackList, ProgressLogger
from everyo.training.history import History
from everyo.training.metrics import get_metric

__all__ = ["Trainer"]

_LOGGER = get_logger(__name__)


class Trainer:
    """Train, evaluate and run inference with an EveryO model.

    Args:
        model: The model to train.
        optimizer: An optimizer already bound to ``model.parameters()``.
        loss_fn: Callable returning a scalar loss tensor.
        metrics: Metric names (see :mod:`everyo.training.metrics`) or callables.
        device: Device used for the forward and backward passes.
        gradient_clip: Optional maximum global gradient norm.

    Example:
        >>> import numpy as np, everyo as eo
        >>> x = np.random.default_rng(0).normal(size=(64, 4)).astype("float32")
        >>> y = (x.sum(axis=1) > 0).astype("int64")
        >>> model = eo.Sequential(eo.Linear(4, 8, seed=0), eo.ReLU(), eo.Linear(8, 2, seed=1))
        >>> trainer = eo.Trainer(model, eo.Adam(model.parameters(), lr=0.01),
        ...                      eo.CrossEntropyLoss(), metrics=["accuracy"])
        >>> history = trainer.fit(eo.DataLoader(eo.ArrayDataset(x, y), batch_size=16),
        ...                       epochs=2, verbose=False)
        >>> history.epochs
        2
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Tensor, Any], Tensor],
        *,
        metrics: Sequence[str | Callable[[Any, Any], float]] = (),
        device: DeviceLike = "cpu",
        gradient_clip: float | None = None,
    ) -> None:
        if not isinstance(model, Module):
            raise TypeError(f"model must be an everyo Module, got {type(model).__name__}.")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"optimizer must be an everyo Optimizer, got {type(optimizer).__name__}."
            )
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable, e.g. eo.CrossEntropyLoss().")
        if gradient_clip is not None and float(gradient_clip) <= 0:
            raise ValueError(f"gradient_clip must be positive, got {gradient_clip}.")

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = resolve_device(device)
        self.gradient_clip = None if gradient_clip is None else float(gradient_clip)
        self.metrics = self._resolve_metrics(metrics)
        self.history = History()
        self.stop_training = False
        self.total_epochs = 0
        self.model.to(self.device)

    @staticmethod
    def _resolve_metrics(
        metrics: Sequence[str | Callable[[Any, Any], float]],
    ) -> dict[str, Callable[[Any, Any], float]]:
        resolved: dict[str, Callable[[Any, Any], float]] = {}
        for metric in metrics:
            if isinstance(metric, str):
                resolved[metric] = get_metric(metric)
            elif callable(metric):
                resolved[getattr(metric, "__name__", "metric")] = metric
            else:
                raise TypeError(f"Metrics must be names or callables, got {type(metric).__name__}.")
        return resolved

    def _loss_is_summed(self) -> bool:
        """Return ``True`` when the loss already sums over the batch.

        A loss built with ``reduction="sum"`` returns the batch total, so
        weighting it by the batch size again would inflate the epoch loss and
        make it depend on how the data was batched. Losses without a
        ``reduction`` attribute are assumed to average, which is the documented
        default and what every EveryO loss does.
        """
        return getattr(self.loss_fn, "reduction", "mean") == "sum"

    def _accumulate(self, batch_loss: float, count: int) -> float:
        """Return this batch's contribution to the epoch's total loss."""
        return batch_loss if self._loss_is_summed() else batch_loss * count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader | Iterable[Any],
        *,
        epochs: int = 10,
        validation_loader: DataLoader | Iterable[Any] | None = None,
        callbacks: Sequence[Callback] = (),
        verbose: bool = True,
        log_every: int = 1,
    ) -> History:
        """Train the model and return its :class:`~everyo.training.history.History`.

        Args:
            train_loader: Loader yielding ``(features, targets)`` batches.
            epochs: Number of passes over the training data.
            validation_loader: Optional loader used after each epoch.
            callbacks: Callbacks observing the run.
            verbose: Print a per-epoch summary.
            log_every: Print every ``n`` epochs when ``verbose`` is set.

        Raises:
            ValueError: If ``epochs`` is not positive.
            EveryOError: If a batch is not a ``(features, targets)`` pair.
        """
        if int(epochs) <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}.")

        callback_list = list(callbacks)
        if verbose and not any(isinstance(cb, ProgressLogger) for cb in callback_list):
            callback_list.append(ProgressLogger(every=log_every))
        handler = CallbackList(callback_list)

        self.total_epochs = int(epochs)
        self.stop_training = False
        handler.on_train_begin(self)

        for epoch in range(1, int(epochs) + 1):
            handler.on_epoch_begin(self, epoch)
            started = time.perf_counter()
            logs = self._run_training_epoch(train_loader, handler)

            if validation_loader is not None:
                for key, value in self.evaluate(validation_loader).items():
                    logs[f"val_{key}"] = value

            logs["time"] = time.perf_counter() - started
            self.history.append(epoch=epoch, **logs)
            handler.on_epoch_end(self, epoch, logs)

            if self.stop_training:
                _LOGGER.info("Training stopped early after epoch %d.", epoch)
                break

        handler.on_train_end(self)
        return self.history

    def evaluate(self, loader: DataLoader | Iterable[Any]) -> dict[str, float]:
        """Return the average loss and metrics over ``loader`` without training."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        with no_grad():
            for batch in loader:
                features, target = self._split_batch(batch)
                output = self.model(features)
                loss = self.loss_fn(output, target)
                count = int(features.shape[0]) if features.ndim else 1
                total_loss += self._accumulate(float(loss.item()), count)
                total_samples += count
                if self.metrics:
                    predictions.append(output.numpy())
                    targets.append(
                        np.asarray(target.data if isinstance(target, Tensor) else target)
                    )

        self.model.train()
        if total_samples == 0:
            raise EveryOError("The evaluation loader produced no batches.")

        results = {"loss": total_loss / total_samples}
        if self.metrics:
            all_predictions = np.concatenate(predictions, axis=0)
            all_targets = np.concatenate(targets, axis=0)
            for name, metric in self.metrics.items():
                results[name] = float(metric(all_predictions, all_targets))
        return results

    def predict(self, data: Any, *, batch_size: int = 64) -> np.ndarray:
        """Run inference and return raw model outputs as a NumPy array.

        Args:
            data: A loader, a dataset, a tensor or a NumPy array.
            batch_size: Batch size used when ``data`` is an array or dataset.
        """
        self.model.eval()
        loader = self._as_loader(data, batch_size)
        outputs: list[np.ndarray] = []
        with no_grad():
            for batch in loader:
                features = batch[0] if isinstance(batch, tuple) else batch
                outputs.append(self.model(as_tensor(features)).numpy())
        self.model.train()
        if not outputs:
            return np.empty((0,))
        return np.concatenate(outputs, axis=0)

    def predict_classes(self, data: Any, *, batch_size: int = 64) -> np.ndarray:
        """Return predicted class indices for a classification model."""
        outputs = self.predict(data, batch_size=batch_size)
        if outputs.ndim == 1 or outputs.shape[-1] == 1:
            return (outputs.reshape(-1) >= 0.5).astype(np.int64)
        return outputs.argmax(axis=-1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_training_epoch(
        self, loader: DataLoader | Iterable[Any], handler: CallbackList
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for index, batch in enumerate(loader):
            handler.on_batch_begin(self, index)
            features, target = self._split_batch(batch)

            self.optimizer.zero_grad()
            output = self.model(features)
            loss = self.loss_fn(output, target)
            loss.backward()
            if self.gradient_clip is not None:
                self._clip_gradients()
            self.optimizer.step()

            count = int(features.shape[0]) if features.ndim else 1
            batch_loss = float(loss.item())
            total_loss += self._accumulate(batch_loss, count)
            total_samples += count
            if self.metrics:
                predictions.append(output.numpy())
                targets.append(np.asarray(target.data if isinstance(target, Tensor) else target))
            handler.on_batch_end(self, index, {"loss": batch_loss})

        if total_samples == 0:
            raise EveryOError(
                "The training loader produced no batches. Check that the "
                "dataset is not empty and that drop_last is not discarding "
                "every batch."
            )

        logs = {"loss": total_loss / total_samples}
        if self.metrics:
            all_predictions = np.concatenate(predictions, axis=0)
            all_targets = np.concatenate(targets, axis=0)
            for name, metric in self.metrics.items():
                logs[name] = float(metric(all_predictions, all_targets))
        return logs

    def _clip_gradients(self) -> None:
        """Scale gradients down so their global L2 norm is at most the clip value."""
        squared = sum(
            float(np.sum(parameter.grad**2))
            for parameter in self.optimizer.parameters
            if parameter.grad is not None
        )
        norm = float(np.sqrt(squared))
        if norm > self.gradient_clip and norm > 0:
            scale = self.gradient_clip / norm
            for parameter in self.optimizer.parameters:
                if parameter.grad is not None:
                    parameter.grad = parameter.grad * scale

    def _split_batch(self, batch: Any) -> tuple[Tensor, Any]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise EveryOError(
                "Training batches must be (features, targets) pairs. Build the "
                "loader from an ArrayDataset created with both features and "
                "targets."
            )
        features, target = batch
        return as_tensor(features, device=self.device), target

    def _as_loader(self, data: Any, batch_size: int) -> Iterable[Any]:
        if isinstance(data, DataLoader):
            return data
        if (
            hasattr(data, "__len__")
            and hasattr(data, "__getitem__")
            and not isinstance(data, (np.ndarray, Tensor))
        ):
            return DataLoader(data, batch_size=batch_size)
        array = data.data if isinstance(data, Tensor) else np.asarray(data)
        return DataLoader(ArrayDataset(array), batch_size=batch_size)

    def __repr__(self) -> str:
        return (
            f"Trainer(model={type(self.model).__name__}, "
            f"optimizer={type(self.optimizer).__name__}, "
            f"loss={type(self.loss_fn).__name__}, device={self.device})"
        )
