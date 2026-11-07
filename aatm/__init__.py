import logging

from .logs import configure_logging

# Libraries should not configure logging on import: attach a NullHandler so the
# package stays silent until an application (the CLI, the API, a notebook) calls
# `configure_logging()` or configures logging itself.
logging.getLogger("aatm").addHandler(logging.NullHandler())

__all__ = ["configure_logging"]
