"""Optional configuration files.

Configuration is a convenience for scripts and the CLI, never a requirement:
every EveryO API can be driven from plain Python.  YAML is supported when
PyYAML is installed; JSON always works.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from everyo.exceptions import EveryOConfigurationError

__all__ = ["ModelConfig", "TrainingConfig", "DeviceConfig", "Config", "load_config"]


@dataclass
class ModelConfig:
    """Shape of a simple feed-forward model."""

    input_size: int = 64
    hidden_size: int = 128
    output_size: int = 10
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("input_size", "hidden_size", "output_size"):
            if int(getattr(self, name)) <= 0:
                raise EveryOConfigurationError(
                    f"model.{name} must be a positive integer, got {getattr(self, name)!r}."
                )
        if not 0.0 <= float(self.dropout) < 1.0:
            raise EveryOConfigurationError(
                f"model.dropout must be in [0, 1), got {self.dropout!r}."
            )


@dataclass
class TrainingConfig:
    """Hyper-parameters of a training run."""

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    validation_split: float = 0.2
    seed: int | None = 0

    def __post_init__(self) -> None:
        if int(self.epochs) <= 0:
            raise EveryOConfigurationError(
                f"training.epochs must be positive, got {self.epochs!r}."
            )
        if int(self.batch_size) <= 0:
            raise EveryOConfigurationError(
                f"training.batch_size must be positive, got {self.batch_size!r}."
            )
        if float(self.learning_rate) <= 0:
            raise EveryOConfigurationError(
                f"training.learning_rate must be positive, got {self.learning_rate!r}."
            )
        if str(self.optimizer).lower() not in ("sgd", "adam"):
            raise EveryOConfigurationError(
                f"training.optimizer must be 'sgd' or 'adam', got {self.optimizer!r}."
            )
        if not 0.0 <= float(self.validation_split) < 1.0:
            raise EveryOConfigurationError(
                f"training.validation_split must be in [0, 1), got {self.validation_split!r}."
            )


@dataclass
class DeviceConfig:
    """Device preference."""

    preferred: str = "auto"

    def __post_init__(self) -> None:
        allowed = ("auto", "cpu", "cuda")
        if str(self.preferred).lower() not in allowed:
            raise EveryOConfigurationError(
                f"device.preferred must be one of {allowed}, got {self.preferred!r}."
            )


@dataclass
class Config:
    """Top-level configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a config from a nested dictionary, validating every section."""
        if not isinstance(data, dict):
            raise EveryOConfigurationError(
                f"Configuration must be a mapping, got {type(data).__name__}."
            )
        known = {"model", "training", "device"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise EveryOConfigurationError(
                f"Unknown configuration section(s): {', '.join(unknown)}. "
                f"Valid sections are: {', '.join(sorted(known))}."
            )
        try:
            return cls(
                model=ModelConfig(**data.get("model", {})),
                training=TrainingConfig(**data.get("training", {})),
                device=DeviceConfig(**data.get("device", {})),
            )
        except TypeError as exc:
            raise EveryOConfigurationError(f"Invalid configuration key: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a nested dictionary."""
        return asdict(self)


def load_config(path: str | Path) -> Config:
    """Load a JSON or YAML configuration file.

    Args:
        path: Path to a ``.json``, ``.yaml`` or ``.yml`` file.

    Raises:
        FileNotFoundError: If the file does not exist.
        EveryOConfigurationError: If the file cannot be parsed, or if YAML is
            requested without PyYAML installed.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EveryOConfigurationError(f"Invalid JSON in {config_path}: {exc}") from exc
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise EveryOConfigurationError(
                f"Reading {config_path} needs PyYAML. Install it with "
                "'pip install everyo[yaml]', or use a .json file instead."
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise EveryOConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    else:
        raise EveryOConfigurationError(
            f"Unsupported configuration format {suffix!r}. Use .json, .yaml or .yml."
        )

    return Config.from_dict(data or {})
