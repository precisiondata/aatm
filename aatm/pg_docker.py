"""Start a local PostgreSQL with pgvector in Docker, and save its connection string.

`aatm init` offers this when PostgreSQL is chosen but cannot be reached (see
`aatm.main.offer_local_pgvector`). `start_local_pgvector` starts, or reuses, one
container:

- named ``aatm-pgvector``, from the ``pgvector/pgvector:pg17`` image;
- reachable from this machine only (``127.0.0.1``), without a password
  (``POSTGRES_HOST_AUTH_METHOD=trust``), on port 5432 or the next free one;
- with its data in a Docker volume of the same name, so the index survives the
  container being removed;
- with 1 GB of shared memory, which the HNSW index build needs.

A stopped container of that name is started again; a running one is reused.
`save_env` writes ``AATM_PG_DSN`` (and optionally ``AATM_VECTOR_DB``) to
``.env``, which every ``aatm`` command loads.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from aatm.logs import get_logger

logger = get_logger(__name__)

CONTAINER_NAME = "aatm-pgvector"
IMAGE = "pgvector/pgvector:pg17"
DATABASE = "aatm"
DEFAULT_PORT = 5432
HOST = "127.0.0.1"

READY_TIMEOUT = 120.0
"""Seconds to wait for a new server: the first start initialises the database."""

Runner = Callable[..., subprocess.CompletedProcess]


class LocalPgVectorError(RuntimeError):
    """Raised when the local PostgreSQL container cannot be started or reached."""


def _run(args: Sequence[str], capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        capture_output=capture,
        text=True,
        check=False,
        timeout=None if not capture else 60,
    )


def docker_problem(run: Runner = _run) -> str | None:
    """Say why Docker cannot be used, or ``None`` when it can.

    Returns:
        ``None`` when the ``docker`` command exists and its daemon answers, and
        otherwise what to install or start.
    """
    if run is _run and shutil.which("docker") is None:
        return "Docker is not installed (https://docs.docker.com/get-docker/)"
    try:
        result = run(["docker", "info", "--format", "{{.ServerVersion}}"])
    except (OSError, subprocess.SubprocessError) as e:
        return f"Docker could not be run: {e}"
    if result.returncode != 0:
        return "Docker is not running; start Docker Desktop (or the Docker daemon)"
    return None


@dataclass(frozen=True)
class ContainerState:
    """An existing container: whether it runs, and the host port of PostgreSQL."""

    running: bool
    port: int | None


def container_state(
    name: str = CONTAINER_NAME, run: Runner = _run
) -> ContainerState | None:
    """Return the state of the container ``name``, or ``None`` if there is none."""
    result = run(["docker", "inspect", "--format", "{{.State.Running}}", name])
    if result.returncode != 0:
        return None
    running = result.stdout.strip().lower() == "true"
    return ContainerState(running=running, port=_published_port(name, run))


def _published_port(name: str, run: Runner) -> int | None:
    # The configured binding, which a stopped container keeps (`docker port`
    # only reports the ports of a running one).
    result = run(
        ["docker", "inspect", "--format", "{{json .HostConfig.PortBindings}}", name]
    )
    if result.returncode != 0:
        return None
    try:
        bindings = json.loads(result.stdout.strip() or "null") or {}
    except json.JSONDecodeError:
        return None
    for binding in bindings.get("5432/tcp") or []:
        port = str(binding.get("HostPort") or "")
        if port.isdigit():
            return int(port)
    return None


def free_port(start: int = DEFAULT_PORT, host: str = HOST, tries: int = 20) -> int:
    """Return ``start``, or the first port after it that nothing listens on.

    Raises:
        LocalPgVectorError: If none of ``tries`` ports is free.
    """
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise LocalPgVectorError(f"No free port between {start} and {start + tries - 1}.")


def dsn_for(port: int) -> str:
    """The connection string of the local container on ``port``."""
    return f"postgresql://postgres@{HOST}:{port}/{DATABASE}"


def run_command(port: int, name: str = CONTAINER_NAME) -> list[str]:
    """The ``docker run`` command that creates the container."""
    return [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--shm-size=1g",
        "-e",
        "POSTGRES_HOST_AUTH_METHOD=trust",
        "-e",
        f"POSTGRES_DB={DATABASE}",
        "-p",
        f"{HOST}:{port}:5432",
        "-v",
        f"{name}:/var/lib/postgresql/data",
        IMAGE,
    ]


def _check(result: subprocess.CompletedProcess, what: str) -> None:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise LocalPgVectorError(f"Could not {what}: {detail or 'docker failed'}")


def start_local_pgvector(
    name: str = CONTAINER_NAME,
    run: Runner = _run,
    wait: Callable[[str, float], None] | None = None,
    log: Callable[[str], None] = logger.info,
) -> str:
    """Start (or reuse) the local PostgreSQL container and wait until it answers.

    Args:
        name: Container name.
        run: Runs a command; replaceable in tests.
        wait: Waits until a connection string answers, or raises. Defaults to
            `wait_until_ready`.
        log: Reports each step.

    Returns:
        The connection string, e.g. ``postgresql://postgres@127.0.0.1:5432/aatm``.

    Raises:
        LocalPgVectorError: If Docker is unavailable, the container cannot be
            started, or the server does not answer in time.
    """
    if problem := docker_problem(run):
        raise LocalPgVectorError(problem)

    state = container_state(name, run)
    if state is not None and state.port is None:
        raise LocalPgVectorError(
            f"A container named '{name}' exists but does not publish PostgreSQL's "
            f"port. Remove it (docker rm -f {name}) and try again."
        )
    if state is None:
        port = free_port()
        log(f"Starting PostgreSQL with pgvector in Docker ({IMAGE}) on port {port}...")
        # Not captured: the first run downloads the image, and shows its progress.
        _check(run(run_command(port, name), capture=False), "start the container")
    elif not state.running:
        port = state.port  # type: ignore[assignment]
        log(f"Starting the existing container '{name}' on port {port}...")
        _check(run(["docker", "start", name]), f"start the container '{name}'")
    else:
        port = state.port  # type: ignore[assignment]
        log(f"Using the running container '{name}' on port {port}.")

    dsn = dsn_for(port)
    (wait or wait_until_ready)(dsn, READY_TIMEOUT)
    return dsn


def wait_until_ready(
    dsn: str, timeout: float = READY_TIMEOUT, interval: float = 1.0
) -> None:
    """Wait until PostgreSQL accepts connections at ``dsn``.

    The image initialises the database on its first start, on a server that
    does not listen on TCP yet, so connecting succeeds only once it is ready.

    Raises:
        LocalPgVectorError: If it does not answer within ``timeout`` seconds.
    """
    from aatm.pgvector_store import import_psycopg

    psycopg = import_psycopg()
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=2) as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError as e:
            last = e
            time.sleep(interval)
    raise LocalPgVectorError(
        f"PostgreSQL did not answer at {dsn} within {timeout:.0f} s"
        + (f": {' '.join(str(last).split())}" if last else "")
        + f". See its log with: docker logs {CONTAINER_NAME}"
    )


def save_env(values: Mapping[str, str], path: Path = Path(".env")) -> None:
    """Write settings to a ``.env`` file and to this process's environment.

    Existing lines with the same names are replaced; other lines are kept.
    """
    import dotenv

    path.touch(exist_ok=True)
    for key, value in values.items():
        dotenv.set_key(str(path), key, value, quote_mode="never")
        os.environ[key] = value
