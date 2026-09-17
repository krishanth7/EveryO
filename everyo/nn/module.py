"""The :class:`Module` base class and :class:`Parameter`.

Modules hold parameters and sub-modules, and know how to describe themselves as
a plain dictionary (:meth:`Module.get_config`).  That description is what makes
safe, pickle-free serialisation possible: a saved model is rebuilt by looking
class names up in a registry rather than by executing stored code.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator

import numpy as np

from everyo.core.device import DeviceLike, resolve_device
from everyo.core.tensor import Tensor
from everyo.exceptions import EveryOError, EveryOSerializationError

__all__ = ["Parameter", "Module", "register_module", "get_module_class", "MODULE_REGISTRY"]

#: Maps class names to module classes for safe deserialisation.
MODULE_REGISTRY: dict[str, type[Module]] = {}


def register_module(cls: type[Module]) -> type[Module]:
    """Class decorator registering a module for serialisation."""
    MODULE_REGISTRY[cls.__name__] = cls
    return cls


def get_module_class(name: str) -> type[Module]:
    """Look a registered module class up by name.

    Raises:
        EveryOSerializationError: If the name is not registered.  This is the
            guard that prevents a model file from instantiating arbitrary code.
    """
    if name not in MODULE_REGISTRY:
        raise EveryOSerializationError(
            f"Unknown module type {name!r}. Only modules registered with "
            f"@register_module can be loaded. Known modules: "
            f"{', '.join(sorted(MODULE_REGISTRY))}."
        )
    return MODULE_REGISTRY[name]


class Parameter(Tensor):
    """A tensor that a module owns and an optimizer updates.

    Parameters always require gradients unless explicitly frozen.
    """

    def __init__(self, data: Any, *, requires_grad: bool = True, name: str | None = None) -> None:
        super().__init__(data, requires_grad=requires_grad, name=name)

    def __repr__(self) -> str:
        return "Parameter(" + super().__repr__()[len("Tensor(") :]


class Module:
    """Base class for every layer, container and loss in EveryO.

    Subclasses implement :meth:`forward`.  Calling the module invokes
    :meth:`forward`, so ``model(x)`` and ``model.forward(x)`` are equivalent.

    Example:
        >>> import everyo as eo
        >>> layer = eo.Linear(3, 2, seed=0)
        >>> layer(eo.tensor([[1.0, 2.0, 3.0]])).shape
        (1, 2)
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    # ------------------------------------------------------------------
    # Attribute handling: registering parameters and sub-modules
    # ------------------------------------------------------------------
    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._ensure_initialised()
            self._parameters[key] = value
            self._modules.pop(key, None)
        elif isinstance(value, Module):
            self._ensure_initialised()
            self._modules[key] = value
            self._parameters.pop(key, None)
        else:
            if isinstance(getattr(self, "_parameters", None), dict):
                self._parameters.pop(key, None)
                self._modules.pop(key, None)
        object.__setattr__(self, key, value)

    def _ensure_initialised(self) -> None:
        if not hasattr(self, "_parameters"):
            raise EveryOError(
                f"{type(self).__name__} assigned a parameter before calling "
                "super().__init__(). Add a super().__init__() call to its "
                "__init__ method."
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the module output.  Must be implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} does not implement forward().")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------
    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Yield ``(qualified_name, parameter)`` for this module and children."""
        for name, parameter in self._parameters.items():
            yield (f"{prefix}{name}", parameter)
        for name, child in self._modules.items():
            yield from child.named_parameters(prefix=f"{prefix}{name}.")

    def parameters(self) -> list[Parameter]:
        """Return every parameter in this module tree.

        The result is a list so it can be passed to an optimizer more than once.
        """
        return [parameter for _, parameter in self.named_parameters()]

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        """Yield ``(qualified_name, module)`` for this module and children."""
        yield (prefix.rstrip("."), self)
        for name, child in self._modules.items():
            yield from child.named_modules(prefix=f"{prefix}{name}.")

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count scalar parameters in this module tree."""
        return int(
            sum(
                parameter.size
                for parameter in self.parameters()
                if parameter.requires_grad or not trainable_only
            )
        )

    def zero_grad(self) -> None:
        """Clear accumulated gradients on every parameter."""
        for parameter in self.parameters():
            parameter.zero_grad()

    def to(self, device: DeviceLike) -> Module:
        """Place every parameter on ``device`` in place and return ``self``.

        Tensor buffers always live in host memory, so this only changes which
        backend executes the module's operations.
        """
        target = resolve_device(device)
        for parameter in self._parameters.values():
            parameter._device = target  # noqa: SLF001 - Module owns its parameters
        for child in self._modules.values():
            child.to(target)
        return self

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> Module:
        """Put the module (and children) into training mode."""
        object.__setattr__(self, "training", bool(mode))
        for child in self._modules.values():
            child.train(mode)
        return self

    def eval(self) -> Module:
        """Put the module (and children) into evaluation mode."""
        return self.train(False)

    # ------------------------------------------------------------------
    # Serialisation support
    # ------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        """Return the constructor arguments needed to rebuild this module.

        Subclasses with constructor arguments must override this.
        """
        return {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Module:
        """Rebuild a module from :meth:`get_config` output."""
        return cls(**config)

    def state_dict(self, prefix: str = "") -> OrderedDict[str, np.ndarray]:
        """Return an ordered mapping of parameter name to NumPy array."""
        state: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, parameter in self.named_parameters(prefix=prefix):
            state[name] = parameter.numpy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray], *, strict: bool = True) -> None:
        """Copy arrays from ``state`` into this module's parameters.

        Args:
            state: Mapping produced by :meth:`state_dict`.
            strict: Require that the key sets match exactly.

        Raises:
            EveryOSerializationError: On missing/unexpected keys (when strict) or
                on a shape mismatch.
        """
        own = dict(self.named_parameters())
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if strict and (missing or unexpected):
            raise EveryOSerializationError(
                "State dict does not match the model. "
                f"Missing keys: {missing or 'none'}. "
                f"Unexpected keys: {unexpected or 'none'}."
            )
        for key, parameter in own.items():
            if key not in state:
                continue
            array = np.asarray(state[key])
            if array.shape != parameter.shape:
                raise EveryOSerializationError(
                    f"Parameter '{key}' expects shape {parameter.shape} but the "
                    f"saved value has shape {array.shape}."
                )
            parameter.data = array.astype(parameter.data.dtype)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        """Extra information shown inside the module's repr."""
        config = self.get_config()
        return ", ".join(f"{key}={value!r}" for key, value in config.items())

    def __repr__(self) -> str:
        head = f"{type(self).__name__}({self.extra_repr()})"
        if not self._modules:
            return head
        lines = [f"{type(self).__name__}("]
        for name, child in self._modules.items():
            child_repr = repr(child).replace("\n", "\n  ")
            lines.append(f"  ({name}): {child_repr}")
        lines.append(")")
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a human readable table of layers and parameter counts."""
        rows = [("Layer", "Type", "Parameters")]
        for name, module in self.named_modules():
            if module is self and name == "":
                continue
            own = int(sum(p.size for p in module._parameters.values()))
            rows.append((name or "-", type(module).__name__, f"{own:,}"))
        widths = [max(len(row[i]) for row in rows) for i in range(3)]
        lines = []
        for index, row in enumerate(rows):
            lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
            if index == 0:
                lines.append("  ".join("-" * width for width in widths))
        lines.append("")
        lines.append(f"Total parameters: {self.num_parameters():,}")
        return "\n".join(lines)
