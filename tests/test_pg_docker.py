"""Starting a local PostgreSQL with pgvector in Docker, against a fake ``docker`` command."""

import json
import subprocess

import pytest

from aatm import main, pg_docker

REAL_FREE_PORT = pg_docker.free_port


class FakeDocker:
    """Answers ``docker`` commands from a dict of containers, and records them."""

    def __init__(self, running=True, containers=None, memory=64 * 2**30, cpus=8):
        self.daemon_running = running
        self.memory = memory
        self.cpus = cpus
        self.containers = containers if containers is not None else {}
        self.calls = []

    def __call__(self, args, capture=True):
        self.calls.append(list(args))
        if args[1] == "container":
            # `docker container inspect` only looks at containers: a volume of
            # the same name must not count.
            assert args[2] == "inspect", args
            args = [args[0], *args[2:]]
        elif args[1] == "inspect":
            raise AssertionError("use `docker container inspect`, not `docker inspect`")
        command = args[1]
        if command == "info":
            if not self.daemon_running:
                return self._result(1)
            if "MemTotal" in args[3]:
                return self._result(0, f"{self.memory} {self.cpus}")
            return self._result(0, "28.5.2")
        if command == "inspect":
            name = args[-1]
            if name not in self.containers:
                return self._result(1, stderr=f"No such object: {name}")
            container = self.containers[name]
            if "PortBindings" in args[3]:
                port = container["port"]
                bindings = (
                    {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]}
                    if port
                    else {}
                )
                return self._result(0, json.dumps(bindings))
            running = "true" if container["running"] else "false"
            return self._result(0, f"{running} {container.get('shm', 2**30)}")
        if command == "run":
            name = args[args.index("--name") + 1]
            port = int(args[args.index("-p") + 1].split(":")[1])
            shm = next(a for a in args if a.startswith("--shm-size="))
            shm_bytes = int(shm.split("=")[1].rstrip("m")) * 2**20
            self.containers[name] = {"running": True, "port": port, "shm": shm_bytes}
            return self._result(0, "abc123")
        if command == "start":
            self.containers[args[2]]["running"] = True
            return self._result(0, args[2])
        raise AssertionError(f"unexpected docker command: {args}")

    @staticmethod
    def _result(code, stdout="", stderr=""):
        return subprocess.CompletedProcess([], code, stdout, stderr)

    def commands(self):
        return [call[1] for call in self.calls]


def no_wait(dsn, timeout):
    pass


@pytest.fixture(autouse=True)
def free_5432(monkeypatch):
    monkeypatch.setattr(pg_docker, "free_port", lambda: 5432)


def test_docker_must_be_running():
    assert pg_docker.docker_problem(FakeDocker()) is None
    assert "not running" in pg_docker.docker_problem(FakeDocker(running=False))


def test_a_new_container_is_started_on_localhost_only():
    docker = FakeDocker()

    server = pg_docker.start_local_pgvector(
        run=docker, wait=no_wait, log=lambda _: None
    )

    assert server.dsn == "postgresql://postgres@127.0.0.1:5432/aatm"
    (run,) = [call for call in docker.calls if call[1] == "run"]
    assert run == pg_docker.run_command(5432, shm_bytes=server.build.shm_bytes)
    assert "127.0.0.1:5432:5432" in run
    assert "POSTGRES_HOST_AUTH_METHOD=trust" in run
    assert "aatm-pgvector:/var/lib/postgresql/data" in run
    # 30% of 64 GB for index builds (capped at 16 GB), plus 1 GB of headroom.
    assert server.build.maintenance_work_mem == "16384MB"
    assert "--shm-size=17408m" in run
    assert server.build.workers == 7
    assert run[-1] == "pgvector/pgvector:pg17"


def test_a_running_container_is_reused_on_its_port():
    docker = FakeDocker(containers={"aatm-pgvector": {"running": True, "port": 5440}})

    server = pg_docker.start_local_pgvector(
        run=docker, wait=no_wait, log=lambda _: None
    )

    assert server.dsn == "postgresql://postgres@127.0.0.1:5440/aatm"
    assert "run" not in docker.commands()
    assert "start" not in docker.commands()


def test_a_stopped_container_is_started_again():
    docker = FakeDocker(containers={"aatm-pgvector": {"running": False, "port": 5433}})

    server = pg_docker.start_local_pgvector(
        run=docker, wait=no_wait, log=lambda _: None
    )

    assert server.dsn.endswith(":5433/aatm")
    assert ["docker", "start", "aatm-pgvector"] in docker.calls
    assert "run" not in docker.commands()


def test_a_container_without_a_published_port_is_explained():
    docker = FakeDocker(containers={"aatm-pgvector": {"running": True, "port": None}})

    with pytest.raises(
        pg_docker.LocalPgVectorError, match="docker rm -f aatm-pgvector"
    ):
        pg_docker.start_local_pgvector(run=docker, wait=no_wait)


def test_docker_not_running_is_explained():
    with pytest.raises(pg_docker.LocalPgVectorError, match="not running"):
        pg_docker.start_local_pgvector(run=FakeDocker(running=False), wait=no_wait)


def test_it_waits_for_the_server_it_started():
    waited = []

    pg_docker.start_local_pgvector(
        run=FakeDocker(), wait=lambda dsn, timeout: waited.append(dsn), log=print
    )

    assert waited == ["postgresql://postgres@127.0.0.1:5432/aatm"]


def test_free_port_skips_ports_in_use():
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]

        assert REAL_FREE_PORT(port) > port

    with pytest.raises(pg_docker.LocalPgVectorError, match="No free port"):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            REAL_FREE_PORT(busy.getsockname()[1], tries=1)


def test_save_env_replaces_the_setting_and_keeps_other_lines(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=k\nAATM_PG_DSN=postgresql://old\n")
    monkeypatch.delenv("AATM_PG_DSN", raising=False)

    pg_docker.save_env(
        {"AATM_PG_DSN": "postgresql://new", "AATM_VECTOR_DB": "pgvector"}, env
    )

    assert env.read_text().splitlines() == [
        "OPENAI_API_KEY=k",
        "AATM_PG_DSN=postgresql://new",
        "AATM_VECTOR_DB=pgvector",
    ]
    import os

    assert os.environ["AATM_PG_DSN"] == "postgresql://new"


def test_save_env_creates_the_file(tmp_path):
    env = tmp_path / ".env"

    pg_docker.save_env({"AATM_PG_DSN": "postgresql://x"}, env)

    assert env.read_text().strip() == "AATM_PG_DSN=postgresql://x"


# --- The offer in aatm init ------------------------------------------------------


class Answers:
    """Stand-in for questionary prompts: returns queued answers, records questions."""

    def __init__(self, monkeypatch, select=None, confirm=None):
        self.questions = []
        self.choices = []

        def make(answer):
            def prompt(question, **kwargs):
                self.questions.append(question)
                self.choices.append(kwargs.get("choices"))
                return type("Q", (), {"ask": lambda _self: answer})()

            return prompt

        monkeypatch.setattr(main.questionary, "select", make(select))
        monkeypatch.setattr(main.questionary, "confirm", make(confirm))


def server(dsn):
    return pg_docker.LocalServer(dsn, pg_docker.BuildSettings("8192MB", 7, 9 * 2**30))


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AATM_PG_DSN", raising=False)
    return tmp_path


def test_choosing_docker_starts_it_and_saves_the_settings(in_tmp, monkeypatch):
    monkeypatch.setattr(pg_docker, "docker_problem", lambda: None)
    monkeypatch.setattr(
        pg_docker,
        "start_local_pgvector",
        lambda log: server("postgresql://postgres@127.0.0.1:5432/aatm"),
    )
    answers = Answers(monkeypatch, select="docker", confirm=True)

    assert main.offer_local_pgvector(OSError("Could not connect\nmore")) is True

    assert (in_tmp / ".env").read_text().splitlines() == [
        "AATM_PG_DSN=postgresql://postgres@127.0.0.1:5432/aatm",
        "AATM_PG_MAINTENANCE_WORK_MEM=8192MB",
        "AATM_PG_BUILD_WORKERS=7",
        "AATM_VECTOR_DB=pgvector",
    ]
    assert "AATM_VECTOR_DB=pgvector" in answers.questions[1]


def test_the_default_backend_is_left_alone_if_declined(in_tmp, monkeypatch):
    monkeypatch.setattr(pg_docker, "docker_problem", lambda: None)
    monkeypatch.setattr(
        pg_docker, "start_local_pgvector", lambda log: server("postgresql://x")
    )
    Answers(monkeypatch, select="docker", confirm=False)

    main.offer_local_pgvector(OSError("x"))

    assert "AATM_VECTOR_DB" not in (in_tmp / ".env").read_text()


def test_setting_it_myself_changes_nothing(in_tmp, monkeypatch):
    monkeypatch.setattr(pg_docker, "docker_problem", lambda: None)

    def must_not_start(log):
        raise AssertionError("must not start a container")

    monkeypatch.setattr(pg_docker, "start_local_pgvector", must_not_start)
    Answers(monkeypatch, select="manual")

    assert main.offer_local_pgvector(OSError("x")) is False
    assert not (in_tmp / ".env").exists()


def test_without_docker_the_option_is_disabled_with_the_reason(in_tmp, monkeypatch):
    monkeypatch.setattr(pg_docker, "docker_problem", lambda: "Docker is not running")
    answers = Answers(monkeypatch, select="manual")

    main.offer_local_pgvector(OSError("x"))

    docker_choice = answers.choices[0][0]
    assert docker_choice.disabled == "Docker is not running"


def test_a_failed_start_is_reported_and_nothing_is_saved(in_tmp, monkeypatch):
    def fails(log):
        raise pg_docker.LocalPgVectorError("image not found")

    monkeypatch.setattr(pg_docker, "docker_problem", lambda: None)
    monkeypatch.setattr(pg_docker, "start_local_pgvector", fails)
    Answers(monkeypatch, select="docker")

    assert main.offer_local_pgvector(OSError("x")) is False
    assert not (in_tmp / ".env").exists()


# --- Build memory and workers ----------------------------------------------------

GiB = 2**30


@pytest.mark.parametrize(
    "memory, cpus, shm, expected",
    [
        (64 * GiB, 32, None, ("16384MB", 7, 17 * GiB)),  # capped at 16 GB and 7
        (8 * GiB, 4, None, ("2457MB", 3, 2457 * 2**20 + GiB)),  # 30% of 8 GB
        (None, None, None, ("1024MB", 2, 2 * GiB)),  # unknown: 1 GB
        (64 * GiB, 32, 1 * GiB, ("768MB", 7, 1 * GiB)),  # within an existing shm
        (64 * GiB, 1, 64 * 2**20, ("64MB", 1, 64 * 2**20)),  # never below 64 MB
    ],
)
def test_build_settings(memory, cpus, shm, expected):
    settings = pg_docker.build_settings(memory, cpus, shm_bytes=shm)

    assert (settings.maintenance_work_mem, settings.workers, settings.shm_bytes) == (
        expected
    )


def test_a_reused_container_gets_settings_within_its_shared_memory():
    docker = FakeDocker(
        containers={"aatm-pgvector": {"running": True, "port": 5432, "shm": GiB}}
    )
    messages = []

    server = pg_docker.start_local_pgvector(
        run=docker, wait=no_wait, log=messages.append
    )

    assert server.build.maintenance_work_mem == "768MB"
    assert server.env()["AATM_PG_MAINTENANCE_WORK_MEM"] == "768MB"
    # And says how to give it more.
    assert any("docker rm -f aatm-pgvector" in m for m in messages)
    assert any("its data stays in the volume" in m for m in messages)


def test_a_container_with_enough_shared_memory_gets_no_hint():
    docker = FakeDocker(
        containers={"aatm-pgvector": {"running": True, "port": 5432, "shm": 20 * GiB}}
    )
    messages = []

    pg_docker.start_local_pgvector(run=docker, wait=no_wait, log=messages.append)

    assert not any("docker rm" in m for m in messages)
