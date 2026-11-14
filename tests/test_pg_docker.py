"""Starting a local PostgreSQL with pgvector in Docker, against a fake ``docker`` command."""

import json
import subprocess

import pytest

from aatm import main, pg_docker

REAL_FREE_PORT = pg_docker.free_port


class FakeDocker:
    """Answers ``docker`` commands from a dict of containers, and records them."""

    def __init__(self, running=True, containers=None):
        self.daemon_running = running
        self.containers = containers if containers is not None else {}
        self.calls = []

    def __call__(self, args, capture=True):
        self.calls.append(list(args))
        command = args[1]
        if command == "info":
            return self._result(0 if self.daemon_running else 1, "28.5.2")
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
            return self._result(0, "true" if container["running"] else "false")
        if command == "run":
            name = args[args.index("--name") + 1]
            port = int(args[args.index("-p") + 1].split(":")[1])
            self.containers[name] = {"running": True, "port": port}
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

    dsn = pg_docker.start_local_pgvector(run=docker, wait=no_wait, log=lambda _: None)

    assert dsn == "postgresql://postgres@127.0.0.1:5432/aatm"
    (run,) = [call for call in docker.calls if call[1] == "run"]
    assert run == pg_docker.run_command(5432)
    assert "127.0.0.1:5432:5432" in run
    assert "POSTGRES_HOST_AUTH_METHOD=trust" in run
    assert "aatm-pgvector:/var/lib/postgresql/data" in run
    assert "--shm-size=1g" in run
    assert run[-1] == "pgvector/pgvector:pg17"


def test_a_running_container_is_reused_on_its_port():
    docker = FakeDocker(containers={"aatm-pgvector": {"running": True, "port": 5440}})

    dsn = pg_docker.start_local_pgvector(run=docker, wait=no_wait, log=lambda _: None)

    assert dsn == "postgresql://postgres@127.0.0.1:5440/aatm"
    assert "run" not in docker.commands()
    assert "start" not in docker.commands()


def test_a_stopped_container_is_started_again():
    docker = FakeDocker(containers={"aatm-pgvector": {"running": False, "port": 5433}})

    dsn = pg_docker.start_local_pgvector(run=docker, wait=no_wait, log=lambda _: None)

    assert dsn.endswith(":5433/aatm")
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
        lambda log: "postgresql://postgres@127.0.0.1:5432/aatm",
    )
    answers = Answers(monkeypatch, select="docker", confirm=True)

    assert main.offer_local_pgvector(OSError("Could not connect\nmore")) is True

    assert (in_tmp / ".env").read_text().splitlines() == [
        "AATM_PG_DSN=postgresql://postgres@127.0.0.1:5432/aatm",
        "AATM_VECTOR_DB=pgvector",
    ]
    assert "AATM_VECTOR_DB=pgvector" in answers.questions[1]


def test_the_default_backend_is_left_alone_if_declined(in_tmp, monkeypatch):
    monkeypatch.setattr(pg_docker, "docker_problem", lambda: None)
    monkeypatch.setattr(pg_docker, "start_local_pgvector", lambda log: "postgresql://x")
    Answers(monkeypatch, select="docker", confirm=False)

    main.offer_local_pgvector(OSError("x"))

    assert (in_tmp / ".env").read_text().strip() == "AATM_PG_DSN=postgresql://x"


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
