"""
Select the PyTorch device used to run local models.

AATM does not require a GPU. Local models (embedding functions and rerankers) run on
the best accelerator PyTorch can see, in this order:

1. ``cuda``: an NVIDIA GPU, when PyTorch was installed with CUDA support.
2. ``mps``: Apple Silicon GPUs, through Metal Performance Shaders.
3. ``cpu``: always available.

Set the ``AATM_DEVICE`` environment variable to ``cuda``, ``mps`` or ``cpu`` to
override the automatic choice, for example to force CPU on a machine with a GPU.

The first selection is logged at ``INFO`` level (see `aatm.logs`), with a hint on how
to enable GPU acceleration when only the CPU is used.

```python
from aatm.device import get_device

device = get_device()  # "cuda", "mps" or "cpu"
```
"""

from __future__ import annotations

import os
import sys
from typing import Literal

import torch

from aatm.logs import get_logger

logger = get_logger(__name__)

DEVICE_ENV_VAR = "AATM_DEVICE"
"""Environment variable that overrides automatic device selection."""

Device = Literal["cuda", "mps", "cpu"]
SUPPORTED_DEVICES: tuple[Device, ...] = ("cuda", "mps", "cpu")

PYTORCH_INSTALL_URL = "https://pytorch.org/get-started/locally/"

_logged_device: Device | None = None


def is_device_available(device: Device) -> bool:
    """Return whether PyTorch can use the given device on this machine.

    Args:
        device: Device name to check.

    Returns:
        ``True`` if the device is usable.
    """
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return True


def detect_device() -> Device:
    """Return the best available device, ignoring ``AATM_DEVICE``.

    Returns:
        ``"cuda"`` if an NVIDIA GPU is usable, else ``"mps"`` if an Apple GPU is
        usable, else ``"cpu"``.
    """
    for device in SUPPORTED_DEVICES:
        if is_device_available(device):
            return device
    return "cpu"


def describe_device(device: Device) -> str:
    """Return a human-readable description of a device.

    Args:
        device: Device name.

    Returns:
        For CUDA, the GPU name; otherwise the device name.
    """
    if device == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "mps (Apple GPU)"
    return "cpu"


def cpu_fallback_hint() -> str:
    """Explain why local models run on CPU and how to enable a GPU.

    Returns:
        A short hint suitable for a log message.
    """
    if sys.platform == "darwin":
        return "No Apple GPU (MPS) is available to PyTorch."
    if torch.version.cuda is None:
        return (
            "The installed PyTorch build has no CUDA support. If you have an NVIDIA "
            f"GPU, install a CUDA build of PyTorch: {PYTORCH_INSTALL_URL}"
        )
    return (
        f"PyTorch was built with CUDA {torch.version.cuda} but cannot see a usable "
        "NVIDIA GPU. Check that a GPU is present and its driver supports this CUDA "
        "version (`nvidia-smi`)."
    )


def _log_selection(device: Device, source: str) -> None:
    """Log the selected device once per process, and again if it changes."""
    global _logged_device
    if device == _logged_device:
        return
    _logged_device = device

    message = f"Running local models on {describe_device(device)} ({source})."
    if device == "cpu" and source == "auto-detected":
        message += " " + cpu_fallback_hint()
    logger.info(message)


def get_device() -> Device:
    """Return the device local models should run on.

    The ``AATM_DEVICE`` environment variable takes precedence over automatic
    detection.

    Returns:
        The selected device name.

    Raises:
        ValueError: If ``AATM_DEVICE`` is set to an unsupported value, or names a
            device that is not available on this machine.
    """
    requested = os.environ.get(DEVICE_ENV_VAR, "").strip().lower()

    if not requested:
        device = detect_device()
        _log_selection(device, "auto-detected")
        return device

    if requested not in SUPPORTED_DEVICES:
        raise ValueError(
            f"{DEVICE_ENV_VAR}={requested!r} is not supported; "
            f"use one of {', '.join(SUPPORTED_DEVICES)}."
        )

    device = requested  # type: ignore[assignment]
    if not is_device_available(device):
        raise ValueError(
            f"{DEVICE_ENV_VAR}={requested!r} was requested, but PyTorch cannot use "
            f"that device on this machine."
        )
    _log_selection(device, f"set by {DEVICE_ENV_VAR}")
    return device
