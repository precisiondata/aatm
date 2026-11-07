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
