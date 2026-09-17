"""Tests for the .evo model archive format."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOSerializationError
from everyo.serialization import MANIFEST_NAME, PARAMETERS_NAME, inspect_archive


@pytest.fixture
def model():
    return eo.Sequential(
        eo.Linear(4, 6, seed=0),
        eo.ReLU(),
        eo.Dropout(0.25, seed=0),
        eo.Linear(6, 3, seed=1),
    )


class TestRoundTrip:
    def test_predictions_are_identical_after_reload(self, model, tmp_path, rng):
        inputs = eo.tensor(rng.normal(size=(5, 4)).astype(np.float32))
        model.eval()
        expected = model(inputs).numpy()

        path = eo.save(model, tmp_path / "model.evo")
        restored = eo.load(path)
        np.testing.assert_allclose(restored(inputs).numpy(), expected, rtol=1e-6)

    def test_architecture_is_restored(self, model, tmp_path):
        restored = eo.load(eo.save(model, tmp_path / "m.evo"))
        assert isinstance(restored, eo.Sequential)
        assert [type(layer).__name__ for layer in restored] == [
            "Linear",
            "ReLU",
            "Dropout",
            "Linear",
        ]
        assert restored[0].in_features == 4
        assert restored[2].p == 0.25

    def test_parameters_are_restored_exactly(self, model, tmp_path):
        restored = eo.load(eo.save(model, tmp_path / "m.evo"))
        for (name, original), (_, loaded) in zip(
            model.named_parameters(), restored.named_parameters()
        ):
            np.testing.assert_allclose(original.numpy(), loaded.numpy(), err_msg=name)

    def test_loaded_model_is_in_eval_mode(self, model, tmp_path):
        assert not eo.load(eo.save(model, tmp_path / "m.evo")).training

    def test_loading_into_an_existing_model(self, model, tmp_path):
        path = eo.save(model, tmp_path / "m.evo")
        target = eo.Sequential(
            eo.Linear(4, 6, seed=9), eo.ReLU(), eo.Dropout(0.25, seed=9), eo.Linear(6, 3, seed=9)
        )
        eo.load(path, model=target)
        np.testing.assert_allclose(target[0].weight.numpy(), model[0].weight.numpy())

    def test_trained_model_survives_the_round_trip(self, tmp_path, rng):
        features = rng.normal(size=(64, 4)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(np.int64)
        model = eo.Sequential(eo.Linear(4, 8, seed=0), eo.ReLU(), eo.Linear(8, 2, seed=1))
        trainer = eo.Trainer(model, eo.Adam(model.parameters(), lr=0.05), eo.CrossEntropyLoss())
        trainer.fit(
            eo.DataLoader(eo.ArrayDataset(features, labels), batch_size=16), epochs=3, verbose=False
        )
        before = trainer.predict(features)
        restored = eo.load(eo.save(model, tmp_path / "trained.evo"))
        after = eo.Trainer(
            restored, eo.Adam(restored.parameters(), lr=0.05), eo.CrossEntropyLoss()
        ).predict(features)
        np.testing.assert_allclose(before, after, rtol=1e-6)


class TestArchiveContents:
    def test_suffix_is_added(self, model, tmp_path):
        assert eo.save(model, tmp_path / "model").name == "model.evo"

    def test_archive_members(self, model, tmp_path):
        path = eo.save(model, tmp_path / "m.evo")
        with zipfile.ZipFile(path) as archive:
            assert set(archive.namelist()) == {MANIFEST_NAME, PARAMETERS_NAME}

    def test_manifest_records_metadata(self, model, tmp_path):
        path = eo.save(model, tmp_path / "m.evo", metadata={"epoch": 7, "accuracy": 0.9})
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read(MANIFEST_NAME))

        assert manifest["everyo_version"] == eo.__version__
        assert manifest["format_version"] == 1
        assert manifest["architecture"]["class_name"] == "Sequential"
        assert manifest["metadata"] == {"epoch": 7, "accuracy": 0.9}
        assert manifest["num_parameters"] == model.num_parameters()
        assert manifest["parameters"]["0.weight"]["shape"] == [4, 6]
        assert "created_at" in manifest

    def test_inspect_archive(self, model, tmp_path):
        info = inspect_archive(eo.save(model, tmp_path / "m.evo"))
        assert info["class_name"] == "Sequential"
        assert info["num_parameters"] == model.num_parameters()

    def test_overwrite_is_guarded(self, model, tmp_path):
        path = eo.save(model, tmp_path / "m.evo")
        with pytest.raises(EveryOSerializationError, match="already exists"):
            eo.save(model, path, overwrite=False)

    def test_directories_are_created(self, model, tmp_path):
        path = eo.save(model, tmp_path / "nested" / "dir" / "m.evo")
        assert path.is_file()


class TestSafety:
    def test_loading_never_unpickles(self, model, tmp_path):
        """A pickled payload must be refused, not executed."""
        path = tmp_path / "evil.evo"
        manifest = {
            "format_version": 1,
            "everyo_version": eo.__version__,
            "architecture": {"class_name": "Sequential", "config": {"layers": []}},
        }
        payload = np.lib.format.__name__  # any bytes that are not a valid .npz
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("parameters.npz", payload.encode("utf-8"))

        with pytest.raises(EveryOSerializationError):
            eo.load(path)

    def test_unknown_module_type_is_refused(self, tmp_path):
        path = tmp_path / "unknown.evo"
        manifest = {
            "format_version": 1,
            "architecture": {"class_name": "EvilModule", "config": {}},
        }
        import io

        buffer = io.BytesIO()
        np.savez(buffer)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("parameters.npz", buffer.getvalue())

        with pytest.raises(EveryOSerializationError, match="Unknown module type"):
            eo.load(path)


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(EveryOSerializationError, match="not found"):
            eo.load(tmp_path / "missing.evo")

    def test_not_a_zip_archive(self, tmp_path):
        path = tmp_path / "broken.evo"
        path.write_text("this is not a zip file", encoding="utf-8")
        with pytest.raises(EveryOSerializationError, match="not a valid"):
            eo.load(path)

    def test_future_format_version_is_refused(self, model, tmp_path):
        path = tmp_path / "future.evo"
        manifest = {"format_version": 99, "architecture": {"class_name": "Sequential"}}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("parameters.npz", b"")
        with pytest.raises(EveryOSerializationError, match="format version 99"):
            eo.load(path)

    def test_shape_mismatch_is_reported(self, model, tmp_path):
        path = eo.save(model, tmp_path / "m.evo")
        wrong = eo.Sequential(
            eo.Linear(4, 99, seed=0), eo.ReLU(), eo.Dropout(0.25), eo.Linear(99, 3, seed=1)
        )
        with pytest.raises(EveryOSerializationError, match="expects shape"):
            eo.load(path, model=wrong)

    def test_saving_a_non_module_is_rejected(self, tmp_path):
        with pytest.raises(EveryOSerializationError, match="everyo Module"):
            eo.save("not a model", tmp_path / "m.evo")

    def test_non_serialisable_metadata_is_reported(self, model, tmp_path):
        with pytest.raises(EveryOSerializationError, match="JSON-serialisable"):
            eo.save(model, tmp_path / "m.evo", metadata={"bad": object()})

    def test_strict_state_dict_mismatch(self, model):
        with pytest.raises(EveryOSerializationError, match="Missing keys"):
            model.load_state_dict({"nope": np.zeros(3)})
