"""Logging helpers.

EveryO uses the standard library :mod:`logging` module.  The library never
configures the root logger on import; applications stay in control.  Calling
:func:`configure_logging` is a convenience for scripts and the CLI.

No telemetry of any kind is collected or transmitted.
"""

from __future__ import annotations

import logging
import os
from typing import Union

__all__ = ["get_logger", "configure_logging", "LOGGER_NAME"]

LOGGER_NAME = "everyo"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the EveryO logger, or a child of it.

    Args:
        name: Optional dotted suffix, typically ``__name__``.

    Returns:
        A :class:`logging.Logger` below the ``everyo`` namespace.
    """
    base = logging.getLogger(LOGGER_NAME)
    if not base.handlers:
        # Avoid "No handlers could be found" noise without hijacking the root
        # logger configuration chosen by the host application.
        base.addHandler(logging.NullHandler())
    if name is None or name == LOGGER_NAME:
        return base
    suffix = name[len(LOGGER_NAME) + 1 :] if name.startswith(LOGGER_NAME + ".") else name
    return base.getChild(suffix)


def configure_logging(level: Union[int, str, None] = None) -> logging.Logger:
    """Attach a stream handler to the EveryO logger.

    Args:
        level: Logging level as an int or name.  Defaults to the ``EVERYO_LOG_LEVEL``
            environment variable, else ``INFO``.

    Returns:
        The configured EveryO logger.
    """
    if level is None:
        level = os.environ.get("EVERYO_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):  # getLevelName returns a str for unknown names
            level = logging.INFO

    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            break
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
