"""Tests for the plotting helpers.

Figures are written to a temporary directory; nothing requires a display.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOError


@pytest.fixture
def history():
    record = eo.History()
    for epoch, (loss, accuracy) in enumerate(
        [(1.0, 0.4), (0.7, 0.6), (0.5, 0.75), (0.4, 0.82)], start=1
    ):
        record.append(
            epoch=epoch,
            loss=loss,
            val_loss=loss + 0.05,
            accuracy=accuracy,
            val_accuracy=accuracy - 0.03,
            time=0.1,
        )
    return record


@pytest.fixture(autouse=True)
def close_figures():
    yield
    from everyo.visualization._backend import get_pyplot

    get_pyplot().close("all")


class TestTrainingCharts:
    def test_plot_loss_returns_a_figure(self, history):
        figure = eo.plot_loss(history)
        assert figure.axes

    def test_plot_loss_writes_a_png(self, history, tmp_path):
        path = tmp_path / "loss.png"
        eo.plot_loss(history, save_path=path)
        assert path.is_file() and path.stat().st_size > 0

    def test_plot_accuracy_writes_a_png(self, history, tmp_path):
        path = tmp_path / "accuracy.png"
        eo.plot_accuracy(history, save_path=path)
        assert path.is_file()

    def test_plot_history_draws_two_panels(self, history, tmp_path):
        figure = eo.plot_history(history, save_path=tmp_path / "history.png")
        assert len(figure.axes) == 2

    def test_plot_history_falls_back_to_loss_only(self, tmp_path):
        record = eo.History()
        record.append(epoch=1, loss=1.0)
        record.append(epoch=2, loss=0.5)
        figure = eo.plot_history(record, save_path=tmp_path / "loss_only.png")
        assert len(figure.axes) == 1

    def test_accepts_a_plain_dict(self, tmp_path):
        figure = eo.plot_loss({"loss": [1.0, 0.5, 0.2]}, save_path=tmp_path / "d.png")
        assert figure.axes

    def test_missing_metric_is_reported(self):
        with pytest.raises(EveryOError, match="None of the requested metrics"):
            eo.plot_accuracy(eo.History())

    def test_nested_directories_are_created(self, history, tmp_path):
        path = tmp_path / "a" / "b" / "loss.png"
        eo.plot_loss(history, save_path=path)
        assert path.is_file()

    def test_rejects_wrong_input_type(self):
        with pytest.raises(TypeError, match="History"):
            eo.plot_loss(["not", "a", "history"])


class TestMetricCharts:
    def test_confusion_matrix_chart(self, tmp_path):
        predictions = np.array([0, 1, 2, 1, 0])
        targets = np.array([0, 1, 1, 1, 2])
        path = tmp_path / "confusion.png"
        eo.plot_confusion_matrix(predictions, targets, save_path=path)
        assert path.is_file()

    def test_normalized_confusion_matrix(self, tmp_path):
        eo.plot_confusion_matrix(
            np.array([0, 1]), np.array([0, 1]), normalize=True, save_path=tmp_path / "n.png"
        )
        assert (tmp_path / "n.png").is_file()

    def test_prediction_chart_for_one_feature(self, tmp_path, rng):
        features = rng.normal(size=(30, 1))
        targets = features.reshape(-1) * 2
        eo.plot_predictions(features, targets, targets + 0.1, save_path=tmp_path / "p.png")
        assert (tmp_path / "p.png").is_file()

    def test_prediction_chart_for_many_features(self, tmp_path, rng):
        features = rng.normal(size=(30, 4))
        targets = rng.normal(size=30)
        eo.plot_predictions(features, targets, targets, save_path=tmp_path / "p.png")
        assert (tmp_path / "p.png").is_file()

    def test_decision_boundary(self, tmp_path, rng):
        from everyo.visualization import plot_decision_boundary

        features = rng.normal(size=(40, 2))
        labels = (features[:, 0] > 0).astype(np.int64)
        plot_decision_boundary(
            lambda grid: (grid[:, 0] > 0).astype(np.int64),
            features,
            labels,
            resolution=25,
            save_path=tmp_path / "boundary.png",
        )
        assert (tmp_path / "boundary.png").is_file()

    def test_decision_boundary_requires_two_features(self, rng):
        from everyo.visualization import plot_decision_boundary

        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            plot_decision_boundary(lambda g: g, rng.normal(size=(10, 3)), np.zeros(10))


class TestBenchmarkCharts:
    @pytest.fixture
    def results(self):
        return [
            {"backend": "numpy-cpu", "size": 128, "seconds": 0.01, "gflops": 1.0},
            {"backend": "numpy-cpu", "size": 256, "seconds": 0.05, "gflops": 2.0},
            {"backend": "everyo-cpu", "size": 128, "seconds": 0.02, "gflops": 0.5},
            {"backend": "everyo-cpu", "size": 256, "seconds": 0.09, "gflops": 1.1},
        ]

    def test_line_chart(self, results, tmp_path):
        eo.plot_benchmark(results, save_path=tmp_path / "bench.png")
        assert (tmp_path / "bench.png").is_file()

    def test_bar_chart(self, results, tmp_path):
        from everyo.visualization import plot_benchmark_bars

        plot_benchmark_bars(results[:2], save_path=tmp_path / "bars.png")
        assert (tmp_path / "bars.png").is_file()

    def test_empty_results_are_reported(self):
        with pytest.raises(EveryOError, match="empty"):
            eo.plot_benchmark([])

    def test_missing_backend_key(self):
        with pytest.raises(EveryOError, match="backend"):
            eo.plot_benchmark([{"size": 1, "seconds": 1.0}])


class TestEndToEnd:
    def test_charts_from_a_real_training_run(self, tmp_path, classification_data):
        features, labels = classification_data
        model = eo.Sequential(eo.Linear(4, 8, seed=0), eo.ReLU(), eo.Linear(8, 3, seed=1))
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.02), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        history = trainer.fit(
            eo.DataLoader(eo.ArrayDataset(features, labels), batch_size=16),
            epochs=3,
            verbose=False,
        )
        eo.plot_history(history, save_path=tmp_path / "history.png")
        eo.plot_confusion_matrix(
            trainer.predict_classes(features), labels, save_path=tmp_path / "cm.png"
        )
        assert (tmp_path / "history.png").is_file()
        assert (tmp_path / "cm.png").is_file()
