"""
Provide centralized logging utilities for the package.

AATM follows the standard rule for Python libraries: modules only create loggers,
and applications decide what is shown. Importing `aatm` prints nothing; the package
logger only has a `NullHandler`.

Applications opt in by calling `configure_logging()`. The AATM entry points (the CLI,
the FastAPI service and the Streamlit search UI) do this for you, with a level taken
from the ``--log-level`` CLI option or the ``AATM_LOG_LEVEL`` environment variable.

Typical usage in a notebook or script:

```python
import logging
import aatm

aatm.configure_logging(logging.DEBUG)  # show everything AATM logs
# or enable a single module:
logging.getLogger("aatm.data_models.extractors").setLevel(logging.DEBUG)
```

Modules create their loggers with ``get_logger(__name__)`` and do not set levels.
"""

from __future__ import annotations
import logging
import os
import sys
from typing import Optional, TextIO


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

PACKAGE_LOGGER_NAME = "aatm"

LOG_LEVEL_ENV_VAR = "AATM_LOG_LEVEL"

DEFAULT_LOG_LEVEL = logging.INFO

_HANDLER_NAME = "aatm-default-handler"


def resolve_log_level(level: int | str | None = None) -> int:
    """Resolve a logging level from a value, the environment, or the default.

    Args:
        level: A level as an int (``logging.DEBUG``) or a name (``"debug"``,
            ``"INFO"``). If ``None``, the ``AATM_LOG_LEVEL`` environment variable is
            used, falling back to ``logging.INFO``.

    Returns:
        The numeric logging level.

    Raises:
        ValueError: If the level name is not a standard logging level.
    """
    if level is None:
        level = os.environ.get(LOG_LEVEL_ENV_VAR) or DEFAULT_LOG_LEVEL

    if isinstance(level, int):
        return level

    name = level.strip().upper()
    if name.isdigit():
        return int(name)

    resolved = logging.getLevelNamesMapping().get(name)
    if resolved is None:
        raise ValueError(
            f"Unknown log level {level!r}. "
            "Use one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    return resolved


def configure_logging(
    level: int | str | None = None,
    stream: Optional[TextIO] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> None:
    """Show AATM log messages, for applications and interactive use.

    Attaches a stream handler to the ``aatm`` logger (never to the root logger, so
    other libraries' logging is left untouched) and sets its level. Messages are not
    propagated to the root logger, so they are not printed twice if the application
    also configures root logging.

    Calling it again is safe: the existing handler is reused and only the level, and
    the stream if given, are updated. This suits notebooks, where cells are re-run.

    Args:
        level: Level as an int or a name. If ``None``, the ``AATM_LOG_LEVEL``
            environment variable is used, falling back to ``logging.INFO``.
        stream: Output stream. Defaults to standard error, so logs never mix with
            command output on standard output.
        log_format: Format string used to render each log record.
        date_format: Format string used to render timestamps in log messages.

    Raises:
        ValueError: If `level` is not a valid logging level.
    """
    resolved_level = resolve_log_level(level)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    handler = next(
        (
            h
            for h in logger.handlers
            if h.get_name() == _HANDLER_NAME and isinstance(h, logging.StreamHandler)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.set_name(_HANDLER_NAME)
        logger.addHandler(handler)
    elif stream is not None:
        handler.setStream(stream)

    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.setLevel(resolved_level)
    logger.propagate = False


def get_logger(
    name: Optional[str] = None, level: Optional[int] = None
) -> logging.Logger:
    """Return a logger instance for a module or component.

    Use ``get_logger(__name__)`` in package modules and leave `level` unset, so that
    levels are controlled in one place through `configure_logging()`.

    Args:
        name: Name of the logger to retrieve. If not provided, the name of this
            module is used.
        level: Optional logging level to apply to the returned logger. Intended for
            application code, not for package modules.

    Returns:
        A ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name or __name__)

    if level:
        logger.setLevel(level)

    return logger
