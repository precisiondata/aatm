import pytest

from aatm import device as device_module
from aatm.device import DEVICE_ENV_VAR, detect_device, get_device


@pytest.fixture
def available(monkeypatch):
    """Control which accelerators look available, without real hardware."""

    def set_available(*devices: str) -> None:
        monkeypatch.setattr(
            device_module.torch.cuda, "is_available", lambda: "cuda" in devices
        )
        monkeypatch.setattr(
            device_module.torch.backends.mps, "is_available", lambda: "mps" in devices
        )

    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    monkeypatch.setattr(device_module, "_logged_device", None)
    monkeypatch.setattr(device_module.torch.cuda, "get_device_name", lambda _: "GPU X")
    return set_available


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        (("cuda", "mps"), "cuda"),
        (("mps",), "mps"),
        ((), "cpu"),
    ],
)
def test_detect_device_prefers_cuda_then_mps_then_cpu(available, devices, expected):
    available(*devices)
    assert detect_device() == expected
    assert get_device() == expected


def test_env_var_overrides_detection(available, monkeypatch):
    available("cuda")
    monkeypatch.setenv(DEVICE_ENV_VAR, " CPU ")
    assert get_device() == "cpu"


def test_env_var_rejects_unknown_device(available, monkeypatch):
    available()
    monkeypatch.setenv(DEVICE_ENV_VAR, "tpu")
    with pytest.raises(ValueError, match="not supported"):
        get_device()


def test_env_var_rejects_unavailable_device(available, monkeypatch):
    available()
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")
    with pytest.raises(ValueError, match="cannot use"):
        get_device()


def test_logs_selected_device_once(available, caplog):
    available("cuda")
    with caplog.at_level("INFO", logger="aatm.device"):
        get_device()
        get_device()
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["Running local models on cuda (GPU X) (auto-detected)."]


def test_cpu_fallback_logs_install_hint(available, caplog, monkeypatch):
    available()
    monkeypatch.setattr(device_module.sys, "platform", "linux")
    monkeypatch.setattr(device_module.torch.version, "cuda", None)
    with caplog.at_level("INFO", logger="aatm.device"):
        get_device()
    (message,) = [r.getMessage() for r in caplog.records]
    assert "no CUDA support" in message
    assert device_module.PYTORCH_INSTALL_URL in message


def test_cpu_with_cuda_build_suggests_driver_check(available, caplog, monkeypatch):
    available()
    monkeypatch.setattr(device_module.sys, "platform", "linux")
    monkeypatch.setattr(device_module.torch.version, "cuda", "12.8")
    with caplog.at_level("INFO", logger="aatm.device"):
        get_device()
    (message,) = [r.getMessage() for r in caplog.records]
    assert "nvidia-smi" in message


def test_forced_cpu_has_no_hint(available, caplog, monkeypatch):
    available("cuda")
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    with caplog.at_level("INFO", logger="aatm.device"):
        get_device()
    (message,) = [r.getMessage() for r in caplog.records]
    assert message == f"Running local models on cpu (set by {DEVICE_ENV_VAR})."
