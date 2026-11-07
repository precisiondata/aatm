"""
Diagnose GPU support and suggest how to fix it.

This module powers the ``aatm doctor`` command. It compares the NVIDIA driver on the
machine (via ``nvidia-smi``) with the installed PyTorch build and explains how to get
GPU acceleration working. It never changes the environment: it only returns the
commands the user should run.

```python
from aatm.doctor import collect_driver_info, collect_torch_info, diagnose

diagnosis = diagnose(collect_driver_info(), collect_torch_info())
print(diagnosis.summary)
for command in diagnosis.commands:
    print(command)
```
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl"
PYTORCH_INSTALL_URL = "https://pytorch.org/get-started/locally/"

CUDA_INDEXES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("cu130", (13, 0)),
    ("cu128", (12, 8)),
    ("cu126", (12, 6)),
)
"""PyTorch CUDA wheel indexes, newest first, with the CUDA version each needs.

Update this list when PyTorch adds or drops CUDA builds.
"""

_CUDA_VERSION_RE = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")

Status = Literal["ok", "info", "action"]
CudaVersion = tuple[int, int]
Runner = Callable[[list[str]], Optional[str]]


@dataclass(frozen=True)
class DriverInfo:
    """NVIDIA driver details reported by ``nvidia-smi``.

    Attributes:
        gpu_name: Name of the first GPU, if reported.
        driver_version: Installed driver version, if reported.
        max_cuda: Highest CUDA version the driver supports, if reported.
    """

    gpu_name: Optional[str]
    driver_version: Optional[str]
    max_cuda: Optional[CudaVersion]


@dataclass(frozen=True)
class TorchInfo:
    """Details of the installed PyTorch build.

    Attributes:
        version: PyTorch version string, e.g. ``"2.11.0+cu128"``.
        cuda: CUDA version PyTorch was built with, or ``None`` for CPU-only builds.
        cuda_available: Whether PyTorch can use a CUDA GPU right now.
        mps_available: Whether PyTorch can use an Apple GPU right now.
    """

    version: str
    cuda: Optional[str]
    cuda_available: bool
    mps_available: bool


@dataclass(frozen=True)
class Diagnosis:
    """Result of ``diagnose``.

    Attributes:
        status: ``"ok"`` if a GPU is in use, ``"info"`` if running on CPU is
            expected, ``"action"`` if the user can enable a GPU.
        summary: One-line conclusion.
        details: Extra explanation, one item per line.
        commands: Shell commands that fix the problem, in order.
    """

    status: Status
    summary: str
    details: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)


def parse_cuda_version(text: str) -> Optional[CudaVersion]:
    """Parse a ``major.minor`` CUDA version.

    Args:
        text: Version string such as ``"12.8"``.

    Returns:
        ``(major, minor)``, or ``None`` if it cannot be parsed.
    """
    match = re.match(r"\s*(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_driver_max_cuda(nvidia_smi_output: str) -> Optional[CudaVersion]:
    """Extract the highest supported CUDA version from ``nvidia-smi`` output.

    Handles both the ``CUDA Version: 12.8`` and ``CUDA UMD Version: 13.3`` headers
    printed by different driver releases.

    Args:
        nvidia_smi_output: Output of plain ``nvidia-smi``.

    Returns:
        ``(major, minor)``, or ``None`` if not found.
    """
    match = _CUDA_VERSION_RE.search(nvidia_smi_output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _run(command: list[str]) -> Optional[str]:
    """Run a command and return its stdout, or ``None`` if it fails."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def collect_driver_info(run: Runner = _run) -> Optional[DriverInfo]:
    """Query the NVIDIA driver with ``nvidia-smi``.

    Args:
        run: Command runner, replaceable in tests.

    Returns:
        Driver details, or ``None`` if ``nvidia-smi`` is missing or fails (which
        usually means there is no NVIDIA GPU or driver).
    """
    if run is _run and shutil.which("nvidia-smi") is None:
        return None

    summary = run(["nvidia-smi"])
    if summary is None:
        return None

    gpu_name = driver_version = None
    query = run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    if query and query.strip():
        first_gpu = query.strip().splitlines()[0]
        name, _, version = first_gpu.rpartition(",")
        gpu_name, driver_version = name.strip() or None, version.strip() or None

    return DriverInfo(
        gpu_name=gpu_name,
        driver_version=driver_version,
        max_cuda=parse_driver_max_cuda(summary),
    )


def collect_torch_info() -> TorchInfo:
    """Describe the installed PyTorch build.

    Returns:
        PyTorch build details.
    """
    return TorchInfo(
        version=torch.__version__,
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
    )


def choose_cuda_index(max_cuda: Optional[CudaVersion]) -> Optional[str]:
    """Pick the newest PyTorch CUDA index the driver supports.

    Args:
        max_cuda: Highest CUDA version supported by the driver. ``None`` means
            unknown, in which case the oldest known index is the safest choice.

    Returns:
        An index tag such as ``"cu128"``, or ``None`` if the driver is older than
        every known index.
    """
    if max_cuda is None:
        return CUDA_INDEXES[-1][0]
    for tag, required in CUDA_INDEXES:
        if max_cuda >= required:
            return tag
    return None


def install_commands(
    index_tag: str, python: str = sys.executable, has_pip: Optional[bool] = None
) -> list[str]:
    """Return commands that replace the installed torch with a given build.

    Commands target the interpreter running AATM, so they change the right
    environment. Environments created by uv usually have no pip, so uv commands are
    returned there.

    Args:
        index_tag: PyTorch index tag, e.g. ``"cu128"`` or ``"cpu"``.
        python: Interpreter whose environment should be changed.
        has_pip: Whether pip is installed; detected when ``None``.

    Returns:
        Shell commands, in order.
    """
    if has_pip is None:
        has_pip = importlib.util.find_spec("pip") is not None
    index_url = f"{PYTORCH_INDEX_URL}/{index_tag}"

    if has_pip:
        return [
            f'"{python}" -m pip uninstall -y torch',
            f'"{python}" -m pip install torch --index-url {index_url}',
        ]
    return [
        f'uv pip install --python "{python}" --reinstall-package torch '
        f"torch --index-url {index_url}"
    ]


def _format_cuda(version: CudaVersion) -> str:
    return f"{version[0]}.{version[1]}"


def diagnose(
    driver: Optional[DriverInfo],
    torch_info: TorchInfo,
    platform: str = sys.platform,
    python: str = sys.executable,
    has_pip: Optional[bool] = None,
) -> Diagnosis:
    """Work out whether local models can use a GPU, and how to fix it if not.

    Args:
        driver: NVIDIA driver details, or ``None`` if no driver was found.
        torch_info: Installed PyTorch build.
        platform: ``sys.platform`` value.
        python: Interpreter used in suggested commands.
        has_pip: Whether pip is installed; detected when ``None``.

    Returns:
        The diagnosis.
    """
    if torch_info.cuda_available:
        gpu = (driver.gpu_name if driver else None) or "an NVIDIA GPU"
        return Diagnosis(
            "ok", f"PyTorch is using CUDA {torch_info.cuda} on {gpu}. Nothing to do."
        )

    if platform == "darwin":
        if torch_info.mps_available:
            return Diagnosis(
                "ok", "PyTorch is using the Apple GPU (MPS). Nothing to do."
            )
        return Diagnosis(
            "info",
            "No Apple GPU (MPS) is available, so local models run on CPU.",
            [
                "CUDA is not supported on macOS. MPS needs Apple Silicon and macOS 12.3+."
            ],
        )

    if driver is None:
        details = [
            "nvidia-smi was not found or failed, so there seems to be no NVIDIA GPU "
            "or driver. Local models will run on CPU.",
            "If this machine has an NVIDIA GPU, install its driver and run "
            "`aatm doctor` again.",
        ]
        if torch_info.cuda is not None and platform.startswith("linux"):
            details.append(
                "Your PyTorch build includes CUDA libraries you cannot use. To save "
                "several GB of disk space, you can switch to the CPU-only build:"
            )
            return Diagnosis(
                "info",
                "No NVIDIA GPU detected; running on CPU.",
                details,
                install_commands("cpu", python, has_pip),
            )
        return Diagnosis("info", "No NVIDIA GPU detected; running on CPU.", details)

    gpu = driver.gpu_name or "an NVIDIA GPU"
    max_cuda = driver.max_cuda
    supports = f"CUDA up to {_format_cuda(max_cuda)}" if max_cuda else "an unknown CUDA"
    index_tag = choose_cuda_index(max_cuda)

    if index_tag is None:
        return Diagnosis(
            "action",
            f"Found {gpu}, but its driver is too old for current PyTorch builds.",
            [
                f"The driver supports {supports}; PyTorch needs at least CUDA "
                f"{_format_cuda(CUDA_INDEXES[-1][1])}.",
                "Update the NVIDIA driver, then run `aatm doctor` again.",
            ],
        )

    if torch_info.cuda is None:
        return Diagnosis(
            "action",
            f"Found {gpu}, but the installed PyTorch ({torch_info.version}) is "
            "CPU-only.",
            [
                f"The driver supports {supports}, so the {index_tag} build of PyTorch "
                "will work. Run these commands in the environment where AATM is "
                "installed:",
            ],
            install_commands(index_tag, python, has_pip),
        )

    torch_cuda = parse_cuda_version(torch_info.cuda)
    if max_cuda and torch_cuda and torch_cuda > max_cuda:
        return Diagnosis(
            "action",
            f"PyTorch was built for CUDA {torch_info.cuda}, but the driver for "
            f"{gpu} only supports {supports}.",
            [
                "Either update the NVIDIA driver, or install a PyTorch build that "
                "matches it:",
            ],
            install_commands(index_tag, python, has_pip),
        )

    details = [
        f"PyTorch {torch_info.version} (CUDA {torch_info.cuda}) and the driver "
        f"({supports}) look compatible, but PyTorch cannot use the GPU.",
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        details.append(
            f"CUDA_VISIBLE_DEVICES is set to {visible!r}, which may hide the GPU."
        )
    details.append(
        "Try restarting, or reinstalling the NVIDIA driver. If it persists, pick a "
        f"build from {PYTORCH_INSTALL_URL}."
    )
    return Diagnosis("action", f"Found {gpu}, but CUDA is not usable.", details)
