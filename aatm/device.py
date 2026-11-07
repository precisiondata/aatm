"""
Select the PyTorch device used to run local models.

AATM does not require a GPU. Local models (embedding functions and rerankers) run on
the best accelerator PyTorch can see, in this order:

1. ``cuda``: an NVIDIA GPU, when PyTorch was installed with CUDA support.
2. ``mps``: Apple Silicon GPUs, through Metal Performance Shaders.
3. ``cpu``: always available.

Set the ``AATM_DEVICE`` environment variable to ``cuda``, ``mps`` or ``cpu`` to
override the automatic choice, for example to force CPU on a machine with a GPU.

```python
from aatm.device import get_device

device = get_device()  # "cuda", "mps" or "cpu"
```
"""

from __future__ import annotations

import os
from typing import Literal

import torch

DEVICE_ENV_VAR = "AATM_DEVICE"
"""Environment variable that overrides automatic device selection."""

Device = Literal["cuda", "mps", "cpu"]
SUPPORTED_DEVICES: tuple[Device, ...] = ("cuda", "mps", "cpu")


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
        return detect_device()

    if requested not in SUPPORTED_DEVICES:
        raise ValueError(
            f"{DEVICE_ENV_VAR}={requested!r} is not supported; "
            f"use one of {', '.join(SUPPORTED_DEVICES)}."
        )

    device: Device = requested  # type: ignore[assignment]
    if not is_device_available(device):
        raise ValueError(
            f"{DEVICE_ENV_VAR}={requested!r} was requested, but PyTorch cannot use "
            f"that device on this machine."
        )
    return device
