"""Start a local PostgreSQL with pgvector in Docker, and save its connection string.

`aatm init` offers this when PostgreSQL is chosen but cannot be reached (see
`aatm.main.offer_local_pgvector`). `start_local_pgvector` starts, or reuses, one
container:

- named ``aatm-pgvector``, from the ``pgvector/pgvector:pg17`` image;
- reachable from this machine only (``127.0.0.1``), without a password
  (``POSTGRES_HOST_AUTH_METHOD=trust``), on port 5432 or the next free one;
- with its data in a Docker volume of the same name, so the index survives the
  container being removed;
- with shared memory sized for the HNSW index build (see `build_settings`).

A stopped container of that name is started again; a running one is reused.
`start_local_pgvector` also returns the build memory and workers that suit the
container and Docker's resources: PostgreSQL's defaults (64 MB, 2 workers)
leave a full index building on disk for over half an hour. `save_env` writes
``AATM_PG_DSN``, those settings (and optionally ``AATM_VECTOR_DB``) to ``.env``,
which every ``aatm`` command loads.
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


MiB = 2**20
GiB = 2**30

MAX_BUILD_MEMORY = 16 * GiB
"""Most memory `build_settings` gives an index build.

Enough for the full index of the largest models: about 9 GB for 2.2 million
768-dimension embeddings, 14 GB for 2,560 dimensions in half precision.
"""

BUILD_MEMORY_SHARE = 0.3
"""Share of Docker's memory `build_settings` gives an index build."""

SHM_HEADROOM = 1 * GiB
"""Shared memory a new container gets on top of its build memory."""

MAX_BUILD_WORKERS = 7
"""Most parallel build workers: the server's default ``max_worker_processes`` is 8."""


@dataclass(frozen=True)
class BuildSettings:
    """Index build settings that suit a container, and its shared memory."""

    maintenance_work_mem: str
    """For ``AATM_PG_MAINTENANCE_WORK_MEM``, e.g. ``"8192MB"``."""

    workers: int
    """For ``AATM_PG_BUILD_WORKERS``."""

    shm_bytes: int
    """``--shm-size`` of a new container."""


def build_settings(
    memory: int | None, cpus: int | None, shm_bytes: int | None = None
) -> BuildSettings:
    """Choose the index build memory and workers for a local container.

    A parallel HNSW build keeps its graph in shared memory, so the build memory
    must fit in the container's ``--shm-size``.

    Args:
        memory: Docker's memory in bytes (``docker info``), or ``None``.
        cpus: Docker's CPUs, or ``None``.
        shm_bytes: The shared memory of an existing container. ``None`` for a
            new one, which gets the build memory plus `SHM_HEADROOM`.

    Returns:
        30% of Docker's memory, at most 16 GB (1 GB if unknown), and within an
        existing container's shared memory; one worker less than the CPUs, at
        most 7.
    """
    build = int(min(MAX_BUILD_MEMORY, memory * BUILD_MEMORY_SHARE)) if memory else GiB
    if shm_bytes is not None:
        build = min(build, shm_bytes - 256 * MiB)
    build = max(64 * MiB, build // MiB * MiB)
    workers = max(1, min(MAX_BUILD_WORKERS, (cpus or 3) - 1))
    shm = shm_bytes if shm_bytes is not None else build + SHM_HEADROOM
    return BuildSettings(f"{build // MiB}MB", workers, shm)


def docker_resources(run: Runner = _run) -> tuple[int | None, int | None]:
    """Docker's memory in bytes and its CPUs, or ``None`` for what it does not say."""
    result = run(["docker", "info", "--format", "{{.MemTotal}} {{.NCPU}}"])
    parts = result.stdout.split() if result.returncode == 0 else []
    values = [int(part) if part.isdigit() else None for part in parts[:2]]
    memory, cpus = (values + [None, None])[:2]
    return memory, cpus


@dataclass(frozen=True)
class ContainerState:
    """An existing container: whether it runs, PostgreSQL's host port, its shared memory."""

    running: bool
    port: int | None
    shm_bytes: int | None = None


def container_state(
    name: str = CONTAINER_NAME, run: Runner = _run
) -> ContainerState | None:
    """Return the state of the container ``name``, or ``None`` if there is none."""
    result = run(
        [
            "docker",
            "container",
            "inspect",
            "--format",
            "{{.State.Running}} {{.HostConfig.ShmSize}}",
            name,
        ]
    )
    if result.returncode != 0:
        return None
    running, _, shm = result.stdout.strip().partition(" ")
    return ContainerState(
        running=running.lower() == "true",
        port=_published_port(name, run),
        # Docker reports 0 for its default of 64 MB.
        shm_bytes=(int(shm) or 64 * MiB) if shm.isdigit() else None,
    )


def _published_port(name: str, run: Runner) -> int | None:
    # The configured binding, which a stopped container keeps (`docker port`
    # only reports the ports of a running one).
    result = run(
        [
            "docker",
            "container",
            "inspect",
            "--format",
            "{{json .HostConfig.PortBindings}}",
            name,
        ]
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


def run_command(
    port: int, name: str = CONTAINER_NAME, shm_bytes: int = 2 * GiB
) -> list[str]:
    """The ``docker run`` command that creates the container."""
    return [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        f"--shm-size={shm_bytes // MiB}m",
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


@dataclass(frozen=True)
class LocalServer:
    """A local PostgreSQL container that answers, and the settings that suit it."""

    dsn: str
    """For ``AATM_PG_DSN``, e.g. ``postgresql://postgres@127.0.0.1:5432/aatm``."""

    build: BuildSettings
    """Index build memory and workers."""

    def env(self) -> dict[str, str]:
        """The settings to save to ``.env``."""
        from aatm.vector_db import (
            PG_BUILD_WORKERS_ENV_VAR,
            PG_DSN_ENV_VAR,
            PG_MAINTENANCE_WORK_MEM_ENV_VAR,
        )

        return {
            PG_DSN_ENV_VAR: self.dsn,
            PG_MAINTENANCE_WORK_MEM_ENV_VAR: self.build.maintenance_work_mem,
            PG_BUILD_WORKERS_ENV_VAR: str(self.build.workers),
        }


def start_local_pgvector(
    name: str = CONTAINER_NAME,
    run: Runner = _run,
    wait: Callable[[str, float], None] | None = None,
    log: Callable[[str], None] = logger.info,
) -> LocalServer:
    """Start (or reuse) the local PostgreSQL container and wait until it answers.

    Args:
        name: Container name.
        run: Runs a command; replaceable in tests.
        wait: Waits until a connection string answers, or raises. Defaults to
            `wait_until_ready`.
        log: Reports each step.

    Returns:
        Its connection string, and the index build settings that fit its shared
        memory and Docker's resources (see `build_settings`).

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
    memory, cpus = docker_resources(run)
    if state is None:
        build = build_settings(memory, cpus)
        port = free_port()
        log(
            f"Starting PostgreSQL with pgvector in Docker ({IMAGE}) on port {port}, "
            f"with {build.maintenance_work_mem} and {build.workers} workers for "
            "index builds..."
        )
        # Not captured: the first run downloads the image, and shows its progress.
        _check(
            run(run_command(port, name, build.shm_bytes), capture=False),
            "start the container",
        )
    else:
        build = build_settings(memory, cpus, shm_bytes=state.shm_bytes)
        port = state.port  # type: ignore[assignment]
        if not state.running:
            log(f"Starting the existing container '{name}' on port {port}...")
            _check(run(["docker", "start", name]), f"start the container '{name}'")
        else:
            log(f"Using the running container '{name}' on port {port}.")
        _hint_if_shm_limits(name, state.shm_bytes, build_settings(memory, cpus), log)

    dsn = dsn_for(port)
    (wait or wait_until_ready)(dsn, READY_TIMEOUT)
    return LocalServer(dsn, build)


def _hint_if_shm_limits(
    name: str, shm_bytes: int | None, ideal: BuildSettings, log: Callable[[str], None]
) -> None:
    """Say how to give an existing container's index builds more memory."""
    if shm_bytes is None or shm_bytes >= ideal.shm_bytes:
        return
    log(
        f"The container '{name}' has {shm_bytes // MiB} MB of shared memory, which "
        f"limits index builds; they could use {ideal.maintenance_work_mem}. To "
        f"recreate it with more (its data stays in the volume '{name}'): "
        f"docker rm -f {name}, then run aatm init again."
    )


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
