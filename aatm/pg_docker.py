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
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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


DockerState = Literal["ready", "missing", "stopped", "denied", "error"]


@dataclass(frozen=True)
class DockerStatus:
    """Whether Docker can be used, and if not, why."""

    state: DockerState
    """``ready``, ``missing`` (not installed), ``stopped`` (its daemon does not
    answer), ``denied`` (no permission to use it) or ``error``."""

    detail: str = ""
    """What ``docker`` said, when it failed."""

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def reason(self) -> str:
        """One line on what is wrong, for a disabled choice."""
        return {
            "ready": "Docker is ready",
            "missing": "Docker is not installed",
            "stopped": "Docker is not running",
            "denied": "no permission to use Docker",
            "error": f"Docker failed: {self.detail}",
        }[self.state]


def docker_status(run: Runner = _run) -> DockerStatus:
    """Check whether the ``docker`` command exists and its daemon answers."""
    if run is _run and shutil.which("docker") is None:
        return DockerStatus("missing")
    try:
        result = run(["docker", "info", "--format", "{{.ServerVersion}}"])
    except (OSError, subprocess.SubprocessError) as e:
        return DockerStatus("error", str(e))
    if result.returncode == 0:
        return DockerStatus("ready")
    detail = " ".join((result.stderr or result.stdout or "").split())
    if "permission denied" in detail.lower():
        return DockerStatus("denied", detail)
    return DockerStatus("stopped", detail)


def docker_problem(run: Runner = _run) -> str | None:
    """Say why Docker cannot be used, or ``None`` when it can."""
    status = docker_status(run)
    return None if status.ready else status.reason()


DOCKER_INSTALL_URLS = {
    "win32": "https://docs.docker.com/desktop/setup/install/windows-install/",
    "darwin": "https://docs.docker.com/desktop/setup/install/mac-install/",
    "linux": "https://docs.docker.com/engine/install/",
}


def _platform(platform: str | None) -> str:
    platform = platform or sys.platform
    return "linux" if platform.startswith("linux") else platform


def docker_guidance(status: DockerStatus, platform: str | None = None) -> str:
    """Explain how to install, start or get access to Docker on this platform."""
    platform = _platform(platform)
    if status.state == "missing":
        url = DOCKER_INSTALL_URLS.get(platform, "https://docs.docker.com/get-docker/")
        if platform == "win32":
            return (
                f"Docker is not installed. Install Docker Desktop: {url}\n"
                "It needs WSL 2 (the installer offers to set it up). Then start "
                "Docker Desktop and wait until it says 'Engine running'."
            )
        if platform == "darwin":
            return (
                f"Docker is not installed. Install Docker Desktop: {url}\n"
                "(or: brew install --cask docker). Then open Docker Desktop and "
                "wait until it says 'Engine running'."
            )
        return (
            f"Docker is not installed. Install Docker Engine: {url}\n"
            "Then start it (sudo systemctl start docker) and let your user run it "
            "(sudo usermod -aG docker $USER, then log out and in)."
        )
    if status.state == "stopped":
        if platform in ("win32", "darwin"):
            where = "the Start menu" if platform == "win32" else "Applications"
            return (
                f"Docker is installed but not running. Start Docker Desktop (from "
                f"{where}) and wait until it says 'Engine running'."
            )
        return (
            "Docker is installed but its daemon is not running. Start it: "
            "sudo systemctl start docker"
        )
    if status.state == "denied":
        return (
            "Docker is running, but your user may not use it. Run: "
            "sudo usermod -aG docker $USER, then log out and in.\n"
            f"Docker said: {status.detail}"
        )
    return f"Docker could not be used: {status.detail or status.reason()}"


def docker_desktop_command(platform: str | None = None) -> list[str] | None:
    """The command that opens Docker Desktop, or ``None`` if it is not found here."""
    platform = _platform(platform)
    if platform == "win32":
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files")]
        for root in roots:
            app = Path(root) / "Docker" / "Docker" / "Docker Desktop.exe"
            if app.exists():
                return [str(app)]
        return None
    if platform == "darwin":
        app = Path("/Applications/Docker.app")
        return ["open", "-a", "Docker"] if app.exists() else None
    return None


def launch_docker_desktop(
    platform: str | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
) -> bool:
    """Open Docker Desktop in the background. Returns whether it was found."""
    command = docker_desktop_command(platform)
    if command is None:
        return False
    popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def wait_for_docker(
    timeout: float = 180.0,
    interval: float = 3.0,
    run: Runner = _run,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> DockerStatus:
    """Wait until Docker answers, e.g. after Docker Desktop was opened."""
    deadline = clock() + timeout
    status = docker_status(run)
    while not status.ready and clock() < deadline:
        sleep(interval)
        status = docker_status(run)
    return status


MiB = 2**20
GiB = 2**30

MAX_BUILD_MEMORY = 20 * GiB
"""Most memory `build_settings` gives an index build.

Enough for the full index (2.46 million expressions) of every model with AATM's HNSW
settings (``m=24``; see `aatm.pgvector_store.hnsw_memory_estimate`): about 10.5 GB for
768 dimensions, 15 GB for 2,560 and 17.5 GB for 3,072 in half precision.
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
        30% of Docker's memory, at most 20 GB (1 GB if unknown), and within an
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
class Resources:
    """The memory and CPUs a local container can use.

    Docker Desktop runs containers in a VM with its own memory and CPU limits
    (``docker info``); the machine's free memory bounds them further.
    """

    docker_memory: int | None = None
    docker_cpus: int | None = None
    host_memory: int | None = None
    host_available: int | None = None
    host_cpus: int | None = None

    @property
    def memory_limit(self) -> int | None:
        """The most memory a container can count on: Docker's, within what is free."""
        known = [m for m in (self.docker_memory, self.host_available) if m]
        return min(known) if known else None

    @property
    def cpus(self) -> int | None:
        return self.docker_cpus or self.host_cpus

    def describe(self) -> str:
        """One line, e.g. ``Docker can use 62.6 GB and 32 CPUs; 90.8 GB of 127.7 GB is free``."""
        parts = []
        if self.docker_memory or self.docker_cpus:
            docker = []
            if self.docker_memory:
                docker.append(_gb(self.docker_memory))
            if self.docker_cpus:
                docker.append(f"{self.docker_cpus} CPUs")
            parts.append("Docker can use " + " and ".join(docker))
        if self.host_available and self.host_memory:
            parts.append(
                f"{_gb(self.host_available)} of this machine's "
                f"{_gb(self.host_memory)} is free"
            )
        return "; ".join(parts) or "Docker's resources are unknown"


def _gb(size: int) -> str:
    return f"{size / GiB:.1f} GB" if size >= GiB else f"{size // MiB} MB"


def detect_resources(run: Runner = _run) -> Resources:
    """Read Docker's limits and, with ``psutil``, this machine's free memory."""
    docker_memory, docker_cpus = docker_resources(run)
    host_memory, host_available = host_memory_reading()
    return Resources(
        docker_memory=docker_memory,
        docker_cpus=docker_cpus,
        host_memory=host_memory,
        host_available=host_available,
        host_cpus=os.cpu_count(),
    )


def host_memory_reading() -> tuple[int | None, int | None]:
    """This machine's total and free memory in bytes, with ``psutil``, if it can tell."""
    try:
        import psutil  # type: ignore[import-untyped]

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:  # psutil missing, or no reading on this platform
        return None, None


def suggest_build(resources: Resources, shm_bytes: int | None = None) -> BuildSettings:
    """`build_settings` for these resources: what ``aatm init`` suggests."""
    return build_settings(resources.memory_limit, resources.cpus, shm_bytes)


_MEMORY = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*$", re.IGNORECASE)
_UNITS = {"": MiB, "k": 2**10, "m": MiB, "g": GiB, "t": 2**40}


def parse_memory(text: str) -> int:
    """Read a memory size such as ``8GB``, ``8g``, ``8192MB`` or ``8192`` (MB), in bytes.

    Raises:
        ValueError: If it is not a size.
    """
    match = _MEMORY.match(text or "")
    if not match:
        raise ValueError(f"Not a memory size: {text!r}. Use e.g. 8GB or 4096MB.")
    number, unit = match.groups()
    return int(float(number) * _UNITS[unit.lower()])


def custom_build(
    memory: int, workers: int, shm_bytes: int | None = None
) -> BuildSettings:
    """Build settings the user chose; a new container gets shared memory to match."""
    memory = max(64 * MiB, memory // MiB * MiB)
    shm = shm_bytes if shm_bytes is not None else memory + SHM_HEADROOM
    return BuildSettings(f"{memory // MiB}MB", int(workers), shm)


def check_build(memory: int, workers: int, resources: Resources) -> str | None:
    """Say why build settings will not work here, or ``None`` when they can."""
    if memory < 64 * MiB:
        return "Use at least 64 MB."
    limit = resources.memory_limit
    if limit and memory > limit * 0.8:
        return (
            f"{_gb(memory)} is more than 80% of the {_gb(limit)} a container can "
            "use here; the server and the rest of the build need memory too."
        )
    if not 1 <= workers <= MAX_BUILD_WORKERS:
        return f"Use 1 to {MAX_BUILD_WORKERS} workers (PostgreSQL's limit is 8)."
    return None


def build_memory_bytes(build: BuildSettings) -> int:
    """The build memory of ``build``, in bytes."""
    return parse_memory(build.maintenance_work_mem)


def fits(build: BuildSettings, shm_bytes: int | None) -> bool:
    """Whether a parallel build with ``build`` fits in ``shm_bytes`` of shared memory."""
    return shm_bytes is None or build_memory_bytes(build) <= shm_bytes - 256 * MiB


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
    build: BuildSettings | None = None,
    recreate: bool = False,
    resources: Resources | None = None,
) -> LocalServer:
    """Start (or reuse) the local PostgreSQL container and wait until it answers.

    Args:
        name: Container name.
        run: Runs a command; replaceable in tests.
        wait: Waits until a connection string answers, or raises. Defaults to
            `wait_until_ready`.
        log: Reports each step.
        build: Index build settings. A new container gets their shared
            memory. Defaults to `suggest_build` for ``resources``.
        recreate: Remove an existing container first, e.g. to give it more
            shared memory. Its data stays in the volume of the same name.
        resources: What the container can use. Defaults to `detect_resources`.

    Returns:
        Its connection string, and the index build settings, within an
        existing container's shared memory.

    Raises:
        LocalPgVectorError: If Docker is unavailable, the container cannot be
            started, or the server does not answer in time.
    """
    if problem := docker_problem(run):
        raise LocalPgVectorError(problem)

    state = container_state(name, run)
    if state is not None and recreate:
        log(
            f"Removing the container '{name}' (its data stays in the volume '{name}')..."
        )
        _check(run(["docker", "rm", "-f", name]), f"remove the container '{name}'")
        state = None
    if state is not None and state.port is None:
        raise LocalPgVectorError(
            f"A container named '{name}' exists but does not publish PostgreSQL's "
            f"port. Remove it (docker rm -f {name}) and try again."
        )
    resources = resources or detect_resources(run)
    if state is None:
        build = build or suggest_build(resources)
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
        ideal = build or suggest_build(resources)
        if fits(ideal, state.shm_bytes):
            build = custom_build(
                build_memory_bytes(ideal), ideal.workers, state.shm_bytes
            )
        else:
            build = suggest_build(resources, shm_bytes=state.shm_bytes)
            build = custom_build(
                build_memory_bytes(build), ideal.workers, state.shm_bytes
            )
            _hint_if_shm_limits(name, state.shm_bytes, ideal, log)
        port = state.port  # type: ignore[assignment]
        if not state.running:
            log(f"Starting the existing container '{name}' on port {port}...")
            _check(run(["docker", "start", name]), f"start the container '{name}'")
        else:
            log(f"Using the running container '{name}' on port {port}.")

    dsn = dsn_for(port)
    (wait or wait_until_ready)(dsn, READY_TIMEOUT)
    return LocalServer(dsn, build)


def _hint_if_shm_limits(
    name: str, shm_bytes: int | None, ideal: BuildSettings, log: Callable[[str], None]
) -> None:
    """Say how to give an existing container's index builds more memory."""
    if fits(ideal, shm_bytes):
        return
    log(
        f"The container '{name}' has {shm_bytes // MiB} MB of shared memory, which "  # type: ignore[operator]
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
