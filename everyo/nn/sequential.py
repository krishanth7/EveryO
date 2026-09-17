"""The :class:`Sequential` container."""

from __future__ import annotations

from typing import Any, Iterator

from everyo.core.tensor import Tensor, as_tensor
from everyo.nn.module import Module, get_module_class, register_module

__all__ = ["Sequential"]


@register_module
class Sequential(Module):
    """Chain modules so that each one consumes the previous one's output.

    Example:
        >>> import everyo as eo
        >>> model = eo.Sequential(
        ...     eo.Linear(4, 16, seed=0),
        ...     eo.ReLU(),
        ...     eo.Linear(16, 3, seed=1),
        ... )
        >>> model(eo.zeros(2, 4)).shape
        (2, 3)
    """

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        for index, layer in enumerate(layers):
            if not isinstance(layer, Module):
                raise TypeError(
                    f"Sequential accepts Module instances, but argument {index} "
                    f"is a {type(layer).__name__}."
                )
            setattr(self, str(index), layer)

    @property
    def layers(self) -> list[Module]:
        """The contained layers, in execution order."""
        return list(self._modules.values())

    def append(self, layer: Module) -> Sequential:
        """Add ``layer`` to the end of the chain and return ``self``."""
        if not isinstance(layer, Module):
            raise TypeError(f"Expected a Module, got {type(layer).__name__}.")
        setattr(self, str(len(self._modules)), layer)
        return self

    def forward(self, x: Any) -> Tensor:
        """Run ``x`` through every layer in order."""
        value = as_tensor(x)
        for layer in self._modules.values():
            value = layer(value)
        return value

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def __getitem__(self, index: int) -> Module:
        return self.layers[index]

    def get_config(self) -> dict[str, Any]:
        """Describe every contained layer so the model can be rebuilt."""
        return {
            "layers": [
                {"class_name": type(layer).__name__, "config": layer.get_config()}
                for layer in self.layers
            ]
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Sequential:
        """Rebuild a Sequential model from :meth:`get_config` output."""
        layers = []
        for entry in config.get("layers", []):
            layer_cls = get_module_class(entry["class_name"])
            layers.append(layer_cls.from_config(entry.get("config", {})))
        return cls(*layers)
