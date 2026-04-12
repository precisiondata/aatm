"""
Provide centralized logging utilities for the package.

This module defines a shared logging configuration and a helper for retrieving
module-aware logger instances. It is intended to offer a single, consistent
entry point for logging setup across the package, including timestamped output,
log levels, module names, and source line numbers.

The configuration is designed to be safe in environments where code may be
reloaded multiple times, such as notebooks or interactive sessions, by avoiding
duplicate handler registration.
"""

from __future__ import annotations
import logging
import sys
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOGGING_LEVEL = logging.CRITICAL


def configure_logging(
    level: int = LOGGING_LEVEL,
    stream: Optional[object] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> None:
    """Configure the root logger for the package.

    This function sets up a stream handler with a consistent formatter and
    attaches it to the root logger. It is intended to be called once, typically
    during application startup, so that all package modules share the same
    logging behavior.

    If the root logger already has handlers attached, the function returns
    without making changes. This prevents duplicate log messages in interactive
    or reloaded environments.

    Args:
        level: Logging level to apply to the root logger, such as
            ``logging.INFO`` or ``logging.DEBUG``.
        stream: Output stream for log messages. If not provided, standard output
            is used.
        log_format: Format string used to render each log record.
        date_format: Format string used to render timestamps in log messages.

    Returns:
        None.

    Side Effects:
        Configures the global root logger, attaches a stream handler, and sets
        the logging level for the package logger namespace.

    Notes:
        The ``aatm`` logger namespace is explicitly set to ``logging.DEBUG``
        after the root logger is configured.
    """
    stream = stream or sys.stdout

    root_logger = logging.getLogger()

    # Prevent duplicated handlers in notebooks / reloads
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format,
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    logging.getLogger("aatm").setLevel(logging.DEBUG)


def get_logger(
    name: Optional[str] = None, level: Optional[int] = None
) -> logging.Logger:
    """Return a logger instance for a module or component.

    This helper retrieves a logger by name and optionally overrides its logging
    level. It is typically used to create module-aware loggers, for example
    with ``get_logger(__name__)``.

    Args:
        name: Name of the logger to retrieve. If not provided, the current
            module name is used.
        level: Optional logging level to apply to the returned logger.

    Returns:
        A configured ``logging.Logger`` instance.

    Notes:
        This function does not configure handlers. It assumes logging has
        already been configured elsewhere, typically through
        ``configure_logging()``.
    """
    logger = logging.getLogger(name or __name__)

    if level:
        logger.setLevel(level)

    return logger
