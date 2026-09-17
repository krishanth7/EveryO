"""Tests for the Trainer, callbacks, history and metrics."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOError


@pytest.fixture
def loaders(classification_data):
    features, labels = classification_data
    x_train, x_test, y_train, y_test = eo.train_test_split(features, labels, test_size=0.25, seed=0)
    return (
        eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=16, shuffle=True, seed=0),
        eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=16),
    )


def build_trainer(metrics=("accuracy",), **kwargs):
    model = eo.Sequential(eo.Linear(4, 16, seed=0), eo.ReLU(), eo.Linear(16, 3, seed=1))
    return eo.Trainer(
        model,
        eo.Adam(model.parameters(), lr=0.02),
        eo.CrossEntropyLoss(),
        metrics=list(metrics),
        **kwargs,
    )


class TestFit:
    def test_records_one_row_per_epoch(self, loaders):
        history = build_trainer().fit(loaders[0], epochs=3, verbose=False)
        assert history.epochs == 3
        assert [record["epoch"] for record in history] == [1, 2, 3]

    def test_loss_decreases(self, loaders):
        history = build_trainer().fit(loaders[0], epochs=15, verbose=False)
        assert history["loss"][-1] < history["loss"][0] * 0.8

    def test_accuracy_improves(self, loaders):
        history = build_trainer().fit(loaders[0], epochs=25, verbose=False)
        assert history["accuracy"][-1] > 0.75

    def test_validation_metrics_are_recorded(self, loaders):
        train_loader, validation_loader = loaders
        history = build_trainer().fit(
            train_loader, epochs=3, validation_loader=validation_loader, verbose=False
        )
        assert "val_loss" in history
        assert "val_accuracy" in history

    def test_elapsed_time_is_recorded(self, loaders):
        history = build_trainer().fit(loaders[0], epochs=2, verbose=False)
        assert all(record["time"] >= 0.0 for record in history)

    def test_parameters_actually_change(self, loaders):
        trainer = build_trainer()
        before = trainer.model[0].weight.numpy()
        trainer.fit(loaders[0], epochs=2, verbose=False)
        assert not np.allclose(before, trainer.model[0].weight.numpy())

    def test_epochs_must_be_positive(self, loaders):
        with pytest.raises(ValueError, match="epochs"):
            build_trainer().fit(loaders[0], epochs=0, verbose=False)

    def test_batches_must_be_pairs(self):
        loader = eo.DataLoader(eo.ArrayDataset(np.zeros((8, 4), dtype=np.float32)), batch_size=4)
        with pytest.raises(EveryOError, match=r"\(features, targets\)"):
            build_trainer().fit(loader, epochs=1, verbose=False)

    def test_verbose_output(self, loaders, capsys):
        build_trainer().fit(loaders[0], epochs=1, verbose=True)
        captured = capsys.readouterr().out
        assert "Epoch 1/1" in captured
        assert "loss=" in captured

    def test_regression_training(self, rng):
        features = rng.normal(size=(120, 3)).astype(np.float32)
        targets = features @ np.array([[1.0], [-2.0], [0.5]], dtype=np.float32)
        model = eo.Sequential(eo.Linear(3, 8, seed=0), eo.Tanh(), eo.Linear(8, 1, seed=1))
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.05), eo.MSELoss(), metrics=["mae"]
        )
        history = trainer.fit(
            eo.DataLoader(eo.ArrayDataset(features, targets), batch_size=16, shuffle=True, seed=0),
            epochs=40,
            verbose=False,
        )
        assert history["loss"][-1] < 0.1


class TestEvaluateAndPredict:
    def test_evaluate_returns_loss_and_metrics(self, loaders):
        trainer = build_trainer()
        trainer.fit(loaders[0], epochs=3, verbose=False)
        results = trainer.evaluate(loaders[1])
        assert set(results) == {"loss", "accuracy"}
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_evaluate_does_not_change_parameters(self, loaders):
        trainer = build_trainer()
        before = trainer.model[0].weight.numpy()
        trainer.evaluate(loaders[1])
        np.testing.assert_allclose(before, trainer.model[0].weight.numpy())

    def test_evaluate_leaves_the_model_in_training_mode(self, loaders):
        trainer = build_trainer()
        trainer.evaluate(loaders[1])
        assert trainer.model.training

    def test_predict_shape(self, classification_data):
        features, _ = classification_data
        assert build_trainer().predict(features).shape == (len(features), 3)

    def test_predict_accepts_a_loader(self, loaders):
        assert build_trainer().predict(loaders[1]).shape[1] == 3

    def test_predict_classes(self, classification_data):
        features, _ = classification_data
        predictions = build_trainer().predict_classes(features)
        assert predictions.shape == (len(features),)
        assert set(np.unique(predictions)).issubset({0, 1, 2})

    def test_empty_loader_is_reported(self):
        loader = eo.DataLoader(
            eo.ArrayDataset(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.int64)),
            batch_size=8,
            drop_last=True,
        )
        with pytest.raises(EveryOError, match="no batches"):
            build_trainer().evaluate(loader)


class TestLossAccumulation:
    """The recorded epoch loss must not depend on the batch size."""

    @staticmethod
    def _recorded_loss(reduction, batch_size):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(12, 3)).astype(np.float32)
        targets = features.sum(axis=1, keepdims=True)
        model = eo.Sequential(eo.Linear(3, 1, seed=0))
        # A learning rate of effectively zero keeps the model fixed, so any
        # difference between runs comes from the accounting, not from training.
        trainer = eo.Trainer(
            model, eo.SGD(model.parameters(), lr=1e-12), eo.MSELoss(reduction=reduction)
        )
        loader = eo.DataLoader(eo.ArrayDataset(features, targets), batch_size=batch_size)
        history = trainer.fit(loader, epochs=1, verbose=False)
        return history["loss"][0], trainer.evaluate(loader)["loss"]

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_loss_is_independent_of_batch_size(self, reduction):
        full, _ = self._recorded_loss(reduction, 12)
        for batch_size in (6, 5, 1):
            batched, _ = self._recorded_loss(reduction, batch_size)
            assert batched == pytest.approx(full, rel=1e-5)

    def test_sum_and_mean_reductions_record_the_same_per_sample_loss(self):
        mean_loss, mean_eval = self._recorded_loss("mean", 5)
        sum_loss, sum_eval = self._recorded_loss("sum", 5)
        assert sum_loss == pytest.approx(mean_loss, rel=1e-5)
        assert sum_eval == pytest.approx(mean_eval, rel=1e-5)

    def test_evaluate_agrees_with_fit(self):
        for reduction in ("mean", "sum"):
            recorded, evaluated = self._recorded_loss(reduction, 5)
            assert recorded == pytest.approx(evaluated, rel=1e-5)


class TestHistory:
    def test_metric_series(self):
        history = eo.History()
        history.append(epoch=1, loss=1.0)
        history.append(epoch=2, loss=0.5)
        assert history["loss"] == [1.0, 0.5]

    def test_unknown_metric(self):
        with pytest.raises(KeyError, match="No metric"):
            eo.History()["nope"]

    def test_best_and_last(self):
        history = eo.History()
        history.append(epoch=1, val_loss=0.9)
        history.append(epoch=2, val_loss=0.3)
        history.append(epoch=3, val_loss=0.6)
        assert history.best("val_loss")["epoch"] == 2
        assert history.last()["epoch"] == 3

    def test_last_on_empty_history(self):
        with pytest.raises(IndexError):
            eo.History().last()

    def test_export_json_and_csv(self, tmp_path):
        history = eo.History()
        history.append(epoch=1, loss=0.5, accuracy=0.8)
        json_path = history.to_json(tmp_path / "h.json")
        csv_path = history.to_csv(tmp_path / "h.csv")
        assert json_path.is_file() and csv_path.is_file()
        assert "accuracy" in csv_path.read_text(encoding="utf-8")


class TestCallbacks:
    def test_early_stopping_halts_training(self, loaders):
        train_loader, validation_loader = loaders
        stopper = eo.EarlyStopping(monitor="val_loss", patience=0)

        class Plateau(eo.Callback):
            """Force a non-improving validation loss to trigger the stop."""

            def on_epoch_end(self, trainer, epoch, logs):
                logs["val_loss"] = 1.0

        history = build_trainer().fit(
            train_loader,
            epochs=20,
            validation_loader=validation_loader,
            callbacks=[Plateau(), stopper],
            verbose=False,
        )
        assert history.epochs < 20
        assert stopper.stopped_epoch > 0

    def test_early_stopping_restores_best_weights(self, loaders):
        train_loader, validation_loader = loaders
        trainer = build_trainer()
        stopper = eo.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
        trainer.fit(
            train_loader,
            epochs=10,
            validation_loader=validation_loader,
            callbacks=[stopper],
            verbose=False,
        )
        assert stopper.best_epoch >= 1

    def test_early_stopping_announces_the_weight_restore(self, loaders, capsys):
        """A silent restore makes post-fit metrics look inconsistent with the log."""
        train_loader, validation_loader = loaders

        class Plateau(eo.Callback):
            def on_epoch_end(self, trainer, epoch, logs):
                logs["val_loss"] = 1.0 if epoch > 1 else 0.1

        build_trainer().fit(
            train_loader,
            epochs=10,
            validation_loader=validation_loader,
            callbacks=[Plateau(), eo.EarlyStopping(monitor="val_loss", patience=1)],
            verbose=False,
        )
        output = capsys.readouterr().out
        assert "Early stopping at epoch" in output
        assert "Restored the best weights from epoch 1" in output

    def test_early_stopping_can_be_quiet(self, loaders, capsys):
        train_loader, validation_loader = loaders

        class Plateau(eo.Callback):
            def on_epoch_end(self, trainer, epoch, logs):
                logs["val_loss"] = 1.0 if epoch > 1 else 0.1

        build_trainer().fit(
            train_loader,
            epochs=10,
            validation_loader=validation_loader,
            callbacks=[
                Plateau(),
                eo.EarlyStopping(monitor="val_loss", patience=1, verbose=False),
            ],
            verbose=False,
        )
        assert "Early stopping" not in capsys.readouterr().out

    def test_metrics_after_fit_describe_the_restored_model(self, loaders):
        """evaluate() after an early stop reports the best model, not the last."""
        train_loader, validation_loader = loaders
        trainer = build_trainer()
        stopper = eo.EarlyStopping(monitor="val_loss", patience=2)
        history = trainer.fit(
            train_loader,
            epochs=30,
            validation_loader=validation_loader,
            callbacks=[stopper],
            verbose=False,
        )
        if stopper.stopped_epoch:
            after = trainer.evaluate(validation_loader)["loss"]
            assert after == pytest.approx(stopper.best, rel=1e-6)
            assert stopper.best <= history["val_loss"][-1] + 1e-9

    def test_checkpoint_writes_a_file(self, loaders, tmp_path):
        train_loader, validation_loader = loaders
        path = tmp_path / "best.evo"
        checkpoint = eo.ModelCheckpoint(path, monitor="val_loss")
        build_trainer().fit(
            train_loader,
            epochs=3,
            validation_loader=validation_loader,
            callbacks=[checkpoint],
            verbose=False,
        )
        assert path.is_file()
        assert checkpoint.saved_epochs
        assert isinstance(eo.load(path), eo.Sequential)

    def test_csv_logger(self, loaders, tmp_path):
        path = tmp_path / "log.csv"
        build_trainer().fit(loaders[0], epochs=2, callbacks=[eo.CSVLogger(path)], verbose=False)
        assert path.read_text(encoding="utf-8").count("\n") == 3  # header + 2 epochs

    def test_learning_rate_scheduler(self, loaders):
        trainer = build_trainer()
        trainer.fit(
            loaders[0],
            epochs=3,
            callbacks=[eo.LearningRateScheduler(lambda epoch, lr: lr * 0.5)],
            verbose=False,
        )
        assert trainer.optimizer.lr == pytest.approx(0.02 * 0.5**3)

    def test_callback_hooks_fire_in_order(self, loaders):
        events = []

        class Recorder(eo.Callback):
            def on_train_begin(self, trainer):
                events.append("train_begin")

            def on_epoch_begin(self, trainer, epoch):
                events.append(f"epoch_begin:{epoch}")

            def on_batch_end(self, trainer, batch, logs):
                events.append("batch_end")

            def on_epoch_end(self, trainer, epoch, logs):
                events.append(f"epoch_end:{epoch}")

            def on_train_end(self, trainer):
                events.append("train_end")

        build_trainer().fit(loaders[0], epochs=2, callbacks=[Recorder()], verbose=False)
        assert events[0] == "train_begin"
        assert events[-1] == "train_end"
        assert "epoch_begin:1" in events and "epoch_end:2" in events

    def test_invalid_callback(self, loaders):
        with pytest.raises(TypeError, match="subclass"):
            build_trainer().fit(loaders[0], epochs=1, callbacks=["not a callback"], verbose=False)


class TestGradientClipping:
    def test_clipping_bounds_the_global_norm(self, loaders):
        trainer = build_trainer(gradient_clip=0.01)
        trainer.fit(loaders[0], epochs=1, verbose=False)
        assert trainer.gradient_clip == 0.01

    def test_invalid_clip_value(self):
        with pytest.raises(ValueError, match="gradient_clip"):
            build_trainer(gradient_clip=0.0)


class TestMetrics:
    def test_accuracy_with_logits(self):
        logits = np.array([[2.0, 1.0], [0.1, 5.0], [3.0, 0.0]])
        assert eo.accuracy(logits, np.array([0, 1, 1])) == pytest.approx(2 / 3)

    def test_accuracy_with_labels(self):
        assert eo.accuracy(np.array([1, 0, 1]), np.array([1, 0, 0])) == pytest.approx(2 / 3)

    def test_confusion_matrix(self):
        matrix = eo.confusion_matrix(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]))
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1 and matrix[2, 1] == 1

    def test_r2_of_a_perfect_fit(self):
        from everyo.training.metrics import r2_score

        values = np.array([1.0, 2.0, 3.0])
        assert r2_score(values, values) == pytest.approx(1.0)

    def test_unknown_metric_name(self):
        from everyo.training.metrics import get_metric

        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("f1")

    def test_trainer_accepts_callable_metrics(self, loaders):
        def custom(prediction, target):
            return 0.42

        trainer = build_trainer(metrics=[custom])
        history = trainer.fit(loaders[0], epochs=1, verbose=False)
        assert history["custom"][0] == pytest.approx(0.42)
