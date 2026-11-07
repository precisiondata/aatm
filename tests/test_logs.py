import io
import logging
import subprocess
import sys

import pytest

import aatm
from aatm.logs import (
    LOG_LEVEL_ENV_VAR,
    configure_logging,
    get_logger,
    resolve_log_level,
)


@pytest.fixture(autouse=True)
def reset_aatm_logger(monkeypatch):
    """Restore the package logger after each test."""
    monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
    logger = logging.getLogger("aatm")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def stream_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]


def test_import_is_silent_and_leaves_root_logger_alone():
    # Run in a fresh interpreter: importing aatm must not configure logging.
    code = (
        "import logging, aatm\n"
        "from aatm.data_models.extractors import logger\n"
        "logger.warning('should not be printed')\n"
        "root = logging.getLogger()\n"
        "pkg = logging.getLogger('aatm')\n"
        "print(len(root.handlers), root.level, pkg.level,"
        " [type(h).__name__ for h in pkg.handlers])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == f"0 {logging.WARNING} 0 ['NullHandler']"
    assert result.stderr == ""


def test_package_exports_configure_logging():
    assert aatm.configure_logging is configure_logging


@pytest.mark.parametrize(
    "value, expected",
    [
        (logging.DEBUG, logging.DEBUG),
        ("debug", logging.DEBUG),
        (" Warning ", logging.WARNING),
        ("20", logging.INFO),
        (None, logging.INFO),
    ],
)
def test_resolve_log_level(value, expected):
    assert resolve_log_level(value) == expected


def test_resolve_log_level_reads_env(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "error")
    assert resolve_log_level() == logging.ERROR


def test_resolve_log_level_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown log level"):
        resolve_log_level("loud")


def test_configure_logging_writes_package_logs_to_stream():
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)

    get_logger("aatm.some_module").debug("hello from aatm")
    logging.getLogger("other_library").warning("not ours")

    output = stream.getvalue()
    assert "DEBUG" in output and "aatm.some_module" in output
    assert "hello from aatm" in output
    assert "not ours" not in output
    assert logging.getLogger("aatm").propagate is False


def test_configure_logging_respects_level():
    stream = io.StringIO()
    configure_logging(logging.WARNING, stream=stream)

    logger = get_logger("aatm.some_module")
    logger.info("hidden")
    logger.warning("shown")

    assert "hidden" not in stream.getvalue()
    assert "shown" in stream.getvalue()


def test_configure_logging_is_idempotent():
    first, second = io.StringIO(), io.StringIO()
    configure_logging("INFO", stream=first)
    configure_logging("DEBUG", stream=second)

    pkg = logging.getLogger("aatm")
    assert len(stream_handlers(pkg)) == 1
    assert pkg.level == logging.DEBUG

    get_logger("aatm.x").debug("once")
    assert first.getvalue() == ""
    assert second.getvalue().count("once") == 1


def test_module_level_can_be_raised_individually():
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    module_logger = logging.getLogger("aatm.data_models.extractors")
    old_level = module_logger.level
    try:
        module_logger.setLevel(logging.DEBUG)
        module_logger.debug("module debug")
        get_logger("aatm.selectors").debug("other debug")
    finally:
        module_logger.setLevel(old_level)

    assert "module debug" in stream.getvalue()
    assert "other debug" not in stream.getvalue()


def test_package_modules_do_not_set_their_own_levels():
    import aatm.data_models.extractors  # noqa: F401
    import aatm.local_database_utils  # noqa: F401
    import aatm.selectors  # noqa: F401
    import aatm.timing_profiler  # noqa: F401

    for name, logger in logging.Logger.manager.loggerDict.items():
        if name.startswith("aatm.") and isinstance(logger, logging.Logger):
            assert logger.level == logging.NOTSET, name
