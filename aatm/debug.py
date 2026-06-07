"""
Centralized debug-mode utilities for the package.

This module provides a simple, package-wide mechanism for enabling and
controlling debug behavior using enumerated debug modes. Debug modes are
defined via the `DebugMode` enum and can be activated dynamically through
an environment variable, avoiding the need to modify source code when
debugging specific execution paths.

The active debug mode is resolved at import or runtime using
`get_debug_mode`, which reads from an environment variable and safely
falls back to a default mode if the value is missing or invalid.

Typical usage:

```python title="debug_example.py"
from aatm.debug import DebugMode, get_debug_mode

DEBUG_MODE = get_debug_mode()
if DEBUG_MODE == DebugMode.MY_MODE:
    logger.debug("MY_MODE is active")
```

Environment Variables:
    DEBUG_MODE: Name or value of a `DebugMode` enum member.
        Matching is case-insensitive. If unset or invalid, the debug
        mode defaults to `DebugMode.NONE`.

Notes:
    - Debug modes are intended for diagnostics and development only and
      should not alter core program logic or behavior.
    - Logging output depends on the logging level; ensure it is set to
      DEBUG to see debug messages.
    - New debug modes should be added as enum members of `DebugMode`
      with clear, descriptive docstrings.
"""

from __future__ import annotations

import os
from enum import Enum


class DebugMode(Enum):
    """Custom class to help debug specific sections of the code in this package. To define a new mode, add a new enum value, set it as the DEBUG_MODE variable and add logging statements in the code conditioned on that mode. For example:

    ``` py
    if DEBUG_MODE == DebugMode.MY_MODE:
        logger.debug("My mode is active")
    ```

    Don't forget to add a docstring to the enum value to explain what it does.
    Don't forget to set logging level to DEBUG to see the debug messages."""

    NONE = None
    OPENAI_LLM_SELECTOR = "openai_llm_selector"
    GEMINI_LLM_SELECTOR = "gemini_llm_selector"
    # Add more package-wide modes here, e.g.:
    # DATASET_LOADING = "dataset_loading"
    # TOKEN_ALIGNMENT = "token_alignment"


def get_debug_mode(
    enum: type[DebugMode] = DebugMode,
    env_var: str = "DEBUG_MODE",
    default: Enum = DebugMode.NONE,
) -> Enum:
    """
    Resolve the active debug mode from environment variables.

    Args:
        enum (Enum): Enumeration containing the debug modes.
        env_var (str): Environment variable containing the debug mode.
            Defaults to "DEBUG_MODE".
        default (Enum): Default debug mode if env_var is unset or invalid.
            Defaults to DebugMode.NONE.

    Returns:
        DebugMode: The active debug mode, or the default if not set.
    """
    raw = (os.getenv(env_var) or "").strip().lower()

    if not raw:
        return default

    # allow both names and values
    for m in enum:
        if raw == m.name.lower() or raw == str(m).lower():
            return m

    return default
