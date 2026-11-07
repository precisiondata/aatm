import logging

import pytest
from typer.testing import CliRunner

from aatm import doctor, main
from aatm.logs import LOG_LEVEL_ENV_VAR
from aatm.doctor import (
    DriverInfo,
    TorchInfo,
    choose_cuda_index,
    collect_driver_info,
    diagnose,
    install_commands,
    parse_driver_max_cuda,
)

PY = "/env/bin/python"
CPU_TORCH = TorchInfo("2.15.0+cpu", None, False, False)
CUDA_TORCH_UNUSABLE = TorchInfo("2.15.0+cu130", "13.0", False, False)
RTX = DriverInfo("NVIDIA GeForce RTX 4090", "610.88", (13, 3))


@pytest.fixture
def cli(monkeypatch):
    """Invoke the CLI without leaking its logging setup into other tests.

    The CLI callback attaches a handler bound to CliRunner's temporary stderr and
    exports AATM_LOG_LEVEL; both must be undone once the test ends.
    """
    logger = logging.getLogger("aatm")
    handlers, level = list(logger.handlers), logger.level
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "INFO")
    # Keep get_device() from probing real hardware in the table row.
    monkeypatch.setenv("AATM_DEVICE", "cpu")
    yield lambda *args: CliRunner().invoke(main.app, list(args))
    logger.handlers[:] = handlers
    logger.setLevel(level)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "| NVIDIA-SMI 570.86   Driver Version: 570.86   CUDA Version: 12.8 |",
            (12, 8),
        ),
        (
            "| NVIDIA-SMI 610.88   KMD Version: 610.88   CUDA UMD Version: 13.3 |",
            (13, 3),
        ),
        ("no version here", None),
    ],
)
def test_parse_driver_max_cuda_handles_old_and_new_headers(output, expected):
    assert parse_driver_max_cuda(output) == expected


@pytest.mark.parametrize(
    ("max_cuda", "expected"),
    [((13, 3), "cu130"), ((12, 9), "cu128"), ((12, 6), "cu126"), ((11, 8), None)],
)
def test_choose_cuda_index_picks_newest_supported(max_cuda, expected):
    assert choose_cuda_index(max_cuda) == expected


def test_choose_cuda_index_falls_back_to_oldest_when_unknown():
    assert choose_cuda_index(None) == doctor.CUDA_INDEXES[-1][0]


def test_install_commands_use_pip_when_available():
    assert install_commands("cu128", PY, has_pip=True) == [
        f'"{PY}" -m pip uninstall -y torch',
        f'"{PY}" -m pip install torch --index-url https://download.pytorch.org/whl/cu128',
    ]


def test_install_commands_use_uv_without_pip():
    (command,) = install_commands("cu128", PY, has_pip=False)
    assert command.startswith(f'uv pip install --python "{PY}" --reinstall-package')
    assert command.endswith("--index-url https://download.pytorch.org/whl/cu128")


def test_collect_driver_info_parses_nvidia_smi():
    outputs = {
        "nvidia-smi": "| NVIDIA-SMI 570.86   CUDA Version: 12.8 |",
        "--query-gpu=name,driver_version": "NVIDIA RTX A6000, 570.86\n",
    }

    def run(command):
        return outputs[command[1] if len(command) > 1 else command[0]]

    assert collect_driver_info(run) == DriverInfo("NVIDIA RTX A6000", "570.86", (12, 8))


def test_collect_driver_info_returns_none_when_nvidia_smi_fails():
    assert collect_driver_info(lambda command: None) is None


def test_gpu_in_use_is_ok():
    torch_info = TorchInfo("2.11.0+cu128", "12.8", True, False)
    result = diagnose(RTX, torch_info, platform="linux")
    assert result.status == "ok"
    assert "RTX 4090" in result.summary


def test_cpu_only_torch_with_gpu_suggests_matching_build():
    result = diagnose(RTX, CPU_TORCH, platform="win32", python=PY, has_pip=True)
    assert result.status == "action"
    assert "CPU-only" in result.summary
    assert result.commands == install_commands("cu130", PY, has_pip=True)


def test_torch_newer_than_driver_suggests_older_build():
    old_driver = DriverInfo("GPU", "550.0", (12, 6))
    result = diagnose(old_driver, CUDA_TORCH_UNUSABLE, platform="linux", python=PY)
    assert result.status == "action"
    assert "only supports CUDA up to 12.6" in result.summary
    assert result.commands[-1].endswith("/cu126")


def test_driver_too_old_asks_for_driver_update():
    result = diagnose(DriverInfo("GPU", "450.0", (11, 0)), CPU_TORCH, platform="linux")
    assert result.status == "action"
    assert result.commands == []
    assert any("Update the NVIDIA driver" in line for line in result.details)


def test_compatible_but_unusable_mentions_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = diagnose(RTX, CUDA_TORCH_UNUSABLE, platform="linux")
    assert result.status == "action"
    assert result.commands == []
    assert any("CUDA_VISIBLE_DEVICES" in line for line in result.details)


def test_no_gpu_on_linux_offers_cpu_build_to_save_space():
    result = diagnose(None, CUDA_TORCH_UNUSABLE, platform="linux", python=PY)
    assert result.status == "info"
    assert result.commands[-1].endswith("/cpu")


def test_no_gpu_with_cpu_torch_needs_nothing():
    result = diagnose(None, CPU_TORCH, platform="win32")
    assert result.status == "info"
    assert result.commands == []


@pytest.mark.parametrize(("mps", "status"), [(True, "ok"), (False, "info")])
def test_macos_reports_mps(mps, status):
    result = diagnose(None, TorchInfo("2.15.0", None, False, mps), platform="darwin")
    assert result.status == status


def test_doctor_command_prints_fix_and_exits_1(cli, monkeypatch):
    monkeypatch.setattr(doctor, "collect_driver_info", lambda: RTX)
    monkeypatch.setattr(doctor, "collect_torch_info", lambda: CPU_TORCH)

    result = cli("doctor")

    assert result.exit_code == 1, result.output
    assert "CPU-only build" in result.output
    assert "https://download.pytorch.org/whl/cu130" in result.output


def test_doctor_command_exits_0_when_gpu_works(cli, monkeypatch):
    monkeypatch.setattr(doctor, "collect_driver_info", lambda: RTX)
    monkeypatch.setattr(
        doctor, "collect_torch_info", lambda: TorchInfo("2.11", "12.8", True, False)
    )

    result = cli("doctor")

    assert result.exit_code == 0, result.output
    assert "Nothing to do" in result.output
