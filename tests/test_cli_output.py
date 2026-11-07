import io
import os
import subprocess
import sys

from aatm import main as cli


def run_cli(*args: str, encoding: str) -> subprocess.CompletedProcess[bytes]:
    """Run the CLI with stdout/stderr redirected to pipes using `encoding`.

    Forcing a legacy encoding through PYTHONIOENCODING reproduces, on any OS, what
    Windows does when output is redirected (cp1252).
    """
    env = {**os.environ, "PYTHONIOENCODING": encoding}
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, "-c", "import sys; from aatm.main import app; app()", *args],
        capture_output=True,
        env=env,
        timeout=300,
    )


def test_logo_prints_with_legacy_encoding_on_redirected_output():
    result = run_cli(encoding="cp1252")

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in result.stderr
    assert "█████╗" in result.stdout.decode("utf-8")


def test_ensure_utf8_output_reconfigures_legacy_streams(monkeypatch):
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    cli.ensure_utf8_output()

    assert stdout.encoding == "utf-8"
    assert stderr.encoding == "utf-8"
    assert os.environ["PYTHONIOENCODING"] == "utf-8"
    print("╚═╝", file=stdout)  # would raise UnicodeEncodeError with cp1252


def test_ensure_utf8_output_keeps_user_pythonioencoding(monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8:replace")
    cli.ensure_utf8_output()
    assert os.environ["PYTHONIOENCODING"] == "utf-8:replace"


def test_ensure_utf8_output_ignores_streams_without_reconfigure(monkeypatch):
    class PlainStream:
        encoding = "cp1252"

        def write(self, text: str) -> int:
            return len(text)

    monkeypatch.setattr(sys, "stdout", PlainStream())
    cli.ensure_utf8_output()  # must not raise


def test_search_ui_disables_streamlit_file_watcher_by_default():
    command = cli.build_search_ui_command(cli.Path("app.py"), env={})
    assert command[1:5] == ["-m", "streamlit", "run", "app.py"]
    assert command[-2:] == ["--server.fileWatcherType", "none"]


def test_search_ui_respects_user_file_watcher_setting():
    command = cli.build_search_ui_command(
        cli.Path("app.py"), env={"STREAMLIT_SERVER_FILE_WATCHER_TYPE": "auto"}
    )
    assert "--server.fileWatcherType" not in command
