"""Tests for the command line interface and configuration loading."""

from __future__ import annotations

import json

import pytest

import everyo as eo
from everyo.cli.main import main
from everyo.config import Config, load_config
from everyo.exceptions import EveryOConfigurationError


class TestInfo:
    def test_exits_successfully(self, capsys):
        assert main(["info"]) == 0
        output = capsys.readouterr().out
        assert "EveryO" in output
        assert eo.__version__ in output

    def test_reports_every_backend(self, capsys):
        main(["info"])
        output = capsys.readouterr().out
        for label in ("NumPy", "TensorFlow", "CUDA", "Device"):
            assert label in output

    def test_json_output_is_parseable(self, capsys):
        main(["info", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["everyo_version"] == eo.__version__
        assert "cuda" in payload and "tensorflow" in payload


class TestDoctor:
    def test_reports_healthy_installation(self, capsys):
        assert main(["doctor"]) == 0
        output = capsys.readouterr().out
        assert "PASS" in output
        assert "Autograd" in output

    def test_mentions_optional_components(self, capsys):
        main(["doctor"])
        output = capsys.readouterr().out
        assert "TensorFlow (optional)" in output
        assert "CUDA (optional)" in output


class TestBenchmark:
    @pytest.mark.slow
    def test_small_benchmark_runs(self, capsys):
        assert main(["benchmark", "--sizes", "32", "--repeats", "1"]) == 0
        output = capsys.readouterr().out
        assert "numpy-cpu" in output
        assert "GFLOP/s" in output

    @pytest.mark.slow
    def test_results_can_be_written_to_json(self, tmp_path, capsys):
        target = tmp_path / "results.json"
        main(["benchmark", "--sizes", "32", "--repeats", "1", "--output", str(target)])
        rows = json.loads(target.read_text(encoding="utf-8"))
        assert rows and rows[0]["benchmark"] == "matmul"
        assert rows[0]["seconds"] > 0


class TestDemo:
    @pytest.mark.slow
    def test_demo_trains_and_reports_accuracy(self, capsys):
        assert main(["demo", "--epochs", "3", "--samples", "20"]) == 0
        output = capsys.readouterr().out
        assert "Test accuracy" in output


class TestParser:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as info:
            main(["--version"])
        assert info.value.code == 0
        assert eo.__version__ in capsys.readouterr().out

    def test_missing_command_is_an_error(self):
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command_is_an_error(self):
        with pytest.raises(SystemExit):
            main(["nonsense"])

    def test_log_level_is_accepted(self, capsys):
        assert main(["--log-level", "DEBUG", "info"]) == 0


class TestConfiguration:
    def test_defaults_are_valid(self):
        config = Config()
        assert config.training.optimizer == "adam"
        assert config.device.preferred == "auto"

    def test_yaml_round_trip(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "config.yaml"
        path.write_text(
            "model:\n"
            "  input_size: 784\n"
            "  hidden_size: 256\n"
            "  output_size: 10\n"
            "training:\n"
            "  epochs: 5\n"
            "  batch_size: 64\n"
            "  learning_rate: 0.01\n"
            "device:\n"
            "  preferred: auto\n",
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.model.input_size == 784
        assert config.training.batch_size == 64

    def test_json_config(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"training": {"epochs": 3}}), encoding="utf-8")
        assert load_config(path).training.epochs == 3

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "config.ini"
        path.write_text("[section]", encoding="utf-8")
        with pytest.raises(EveryOConfigurationError, match="Unsupported"):
            load_config(path)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(EveryOConfigurationError, match="Invalid JSON"):
            load_config(path)

    def test_unknown_section(self):
        with pytest.raises(EveryOConfigurationError, match="Unknown configuration section"):
            Config.from_dict({"mystery": {}})

    def test_invalid_values_are_reported(self):
        with pytest.raises(EveryOConfigurationError, match="epochs"):
            Config.from_dict({"training": {"epochs": 0}})
        with pytest.raises(EveryOConfigurationError, match="input_size"):
            Config.from_dict({"model": {"input_size": -1}})
        with pytest.raises(EveryOConfigurationError, match="optimizer"):
            Config.from_dict({"training": {"optimizer": "rmsprop"}})

    def test_to_dict_round_trip(self):
        config = Config.from_dict({"training": {"epochs": 4}})
        assert Config.from_dict(config.to_dict()).training.epochs == 4

    def test_bundled_example_config_is_valid(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        if not path.is_file():
            pytest.skip("Example configuration is only present in the source tree.")
        pytest.importorskip("yaml")
        assert load_config(path).model.input_size > 0
