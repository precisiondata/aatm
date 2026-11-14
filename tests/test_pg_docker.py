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
        if command == "rm":
            self.containers.pop(args[-1], None)
            return self._result(0, args[-1])
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
    # Size from the fake Docker only, not from this machine's free memory.
    monkeypatch.setattr(pg_docker, "host_memory_reading", lambda: (None, None))


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
    """Stand-in for questionary prompts: answers in order, and records the questions."""

    def __init__(self, monkeypatch, *answers):
        self.answers = list(answers)
        self.questions = []
        self.choices = []
        self.defaults = []

        def prompt(question, **kwargs):
            self.questions.append(question)
            self.choices.append(kwargs.get("choices"))
            self.defaults.append(kwargs.get("default"))
            validate = kwargs.get("validate")
            answer = self.answers.pop(0)
            if validate is not None and answer is not None:
                assert validate(answer) is True, (answer, validate(answer))
            return type("Q", (), {"ask": lambda _self: answer})()

        for kind in ("select", "confirm", "text"):
            monkeypatch.setattr(main.questionary, kind, prompt)


READY = pg_docker.DockerStatus("ready")
RESOURCES = pg_docker.Resources(
    docker_memory=64 * 2**30, docker_cpus=8, host_memory=128 * 2**30
)


def server(dsn, build=None):
    return pg_docker.LocalServer(
        dsn, build or pg_docker.BuildSettings("8192MB", 7, 9 * 2**30)
    )


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AATM_PG_DSN", raising=False)
    monkeypatch.delenv("AATM_VECTOR_DB", raising=False)
    monkeypatch.setattr(pg_docker, "docker_status", lambda run=None: READY)
    monkeypatch.setattr(pg_docker, "detect_resources", lambda run=None: RESOURCES)
    monkeypatch.setattr(pg_docker, "container_state", lambda name=None, run=None: None)
    return tmp_path


@pytest.fixture
def started(monkeypatch):
    """Record how `start_local_pgvector` is called, and pretend it worked."""
    calls = []

    def start(log, build=None, recreate=False, resources=None):
        calls.append({"build": build, "recreate": recreate})
        return server("postgresql://postgres@127.0.0.1:5432/aatm", build)

    monkeypatch.setattr(pg_docker, "start_local_pgvector", start)
    return calls


def test_choosing_docker_starts_it_and_saves_the_settings(in_tmp, monkeypatch, started):
    answers = Answers(monkeypatch, "docker", "use", True)

    assert main.offer_local_pgvector(OSError("Could not connect\nmore")) is True

    suggested = pg_docker.suggest_build(RESOURCES)
    assert started == [{"build": suggested, "recreate": False}]
    assert (in_tmp / ".env").read_text().splitlines() == [
        "AATM_PG_DSN=postgresql://postgres@127.0.0.1:5432/aatm",
        f"AATM_PG_MAINTENANCE_WORK_MEM={suggested.maintenance_work_mem}",
        f"AATM_PG_BUILD_WORKERS={suggested.workers}",
        "AATM_VECTOR_DB=pgvector",
    ]
    assert "AATM_VECTOR_DB=pgvector" in answers.questions[-1]


def test_the_suggestion_can_be_changed(in_tmp, monkeypatch, started):
    Answers(monkeypatch, "docker", "change", "4GB", "3", False)

    main.offer_local_pgvector(OSError("x"))

    (call,) = started
    assert call["build"].maintenance_work_mem == "4096MB"
    assert call["build"].workers == 3
    assert call["build"].shm_bytes == 5 * 2**30  # a new container gets 1 GB more
    env = (in_tmp / ".env").read_text()
    assert "AATM_PG_MAINTENANCE_WORK_MEM=4096MB" in env
    assert "AATM_PG_BUILD_WORKERS=3" in env
    assert "AATM_VECTOR_DB" not in env


def test_changed_settings_are_checked_against_the_resources(in_tmp, monkeypatch):
    validators = {}

    def text(question, default=None, validate=None):
        validators[question.split(" ")[0]] = validate
        return type("Q", (), {"ask": lambda _self: None})()

    Answers(monkeypatch, "docker", "change")
    monkeypatch.setattr(main.questionary, "text", text)

    assert main.offer_local_pgvector(OSError("x")) is False

    memory = validators["Memory"]
    assert memory("8GB") is True
    assert "Not a memory size" in memory("lots")
    assert "more than 80%" in memory("60GB")  # Docker can use 64 GB
    assert "at least 64 MB" in memory("10MB")


def test_too_many_workers_are_refused(in_tmp, monkeypatch):
    seen = {}

    def text(question, default=None, validate=None):
        seen[question.split(" ")[0]] = validate
        answer = "8GB" if question.startswith("Memory") else None
        return type("Q", (), {"ask": lambda _self: answer})()

    Answers(monkeypatch, "docker", "change")
    monkeypatch.setattr(main.questionary, "text", text)

    main.offer_local_pgvector(OSError("x"))

    workers = seen["Parallel"]
    assert workers("7") is True
    assert "1 to 7 workers" in workers("12")
    assert "whole number" in workers("two")


def test_an_undersized_container_can_be_recreated(in_tmp, monkeypatch, started):
    existing = pg_docker.ContainerState(running=True, port=5432, shm_bytes=2**30)
    monkeypatch.setattr(
        pg_docker, "container_state", lambda name=None, run=None: existing
    )
    answers = Answers(monkeypatch, "docker", "use", "recreate", False)

    main.offer_local_pgvector(OSError("x"))

    assert "1024 MB of shared memory" in answers.questions[2]
    assert "data stays in the volume" in answers.choices[2][0].title
    assert started[0]["recreate"] is True


def test_an_undersized_container_can_be_kept(in_tmp, monkeypatch, started):
    existing = pg_docker.ContainerState(running=True, port=5432, shm_bytes=2**30)
    monkeypatch.setattr(
        pg_docker, "container_state", lambda name=None, run=None: existing
    )
    Answers(monkeypatch, "docker", "use", "keep", False)

    main.offer_local_pgvector(OSError("x"))

    assert started[0]["recreate"] is False


def test_a_container_with_enough_shared_memory_is_not_questioned(
    in_tmp, monkeypatch, started
):
    existing = pg_docker.ContainerState(running=True, port=5432, shm_bytes=40 * 2**30)
    monkeypatch.setattr(
        pg_docker, "container_state", lambda name=None, run=None: existing
    )
    answers = Answers(monkeypatch, "docker", "use", False)

    main.offer_local_pgvector(OSError("x"))

    assert len(answers.questions) == 3  # no recreate question
    assert started[0]["recreate"] is False


def test_setting_it_myself_changes_nothing(in_tmp, monkeypatch):
    def must_not_start(**kwargs):
        raise AssertionError("must not start a container")

    monkeypatch.setattr(pg_docker, "start_local_pgvector", must_not_start)
    Answers(monkeypatch, "manual")

    assert main.offer_local_pgvector(OSError("x")) is False
    assert not (in_tmp / ".env").exists()


def test_a_failed_start_is_reported_and_nothing_is_saved(in_tmp, monkeypatch):
    def fails(**kwargs):
        raise pg_docker.LocalPgVectorError("image not found")

    monkeypatch.setattr(pg_docker, "start_local_pgvector", fails)
    Answers(monkeypatch, "docker", "use")

    assert main.offer_local_pgvector(OSError("x")) is False
    assert not (in_tmp / ".env").exists()


# --- Without Docker --------------------------------------------------------------


def test_the_docker_choice_says_what_is_missing(in_tmp, monkeypatch):
    status = pg_docker.DockerStatus("missing")
    monkeypatch.setattr(pg_docker, "docker_status", lambda run=None: status)
    answers = Answers(monkeypatch, "manual")

    main.offer_local_pgvector(OSError("x"))

    docker_choice = answers.choices[0][0]
    assert docker_choice.disabled is None  # chosen, it explains what to do
    assert "Docker is not installed: AATM shows how to fix that" in docker_choice.title


def test_missing_docker_is_explained_and_checked_again(in_tmp, monkeypatch, started):
    statuses = iter([pg_docker.DockerStatus("missing"), READY])
    monkeypatch.setattr(pg_docker, "docker_status", lambda run=None: next(statuses))
    shown = []
    monkeypatch.setattr(
        main.console, "print", lambda *a, **k: shown.append(" ".join(map(str, a)))
    )
    answers = Answers(monkeypatch, "docker", "retry", "use", False)

    assert main.offer_local_pgvector(OSError("x")) is True

    assert any("Install Docker" in line for line in shown)
    values = [c.value for c in answers.choices[1]]
    assert values == ["retry", "stop"]  # nothing to launch when it is not installed
    assert started


def test_stopped_docker_desktop_can_be_started_for_the_user(
    in_tmp, monkeypatch, started
):
    monkeypatch.setattr(
        pg_docker, "docker_status", lambda run=None: pg_docker.DockerStatus("stopped")
    )
    monkeypatch.setattr(pg_docker, "docker_desktop_command", lambda: ["Docker.exe"])
    launched = []
    monkeypatch.setattr(pg_docker, "launch_docker_desktop", lambda: launched.append(1))
    monkeypatch.setattr(pg_docker, "wait_for_docker", lambda: READY)
    answers = Answers(monkeypatch, "docker", "launch", "use", False)

    assert main.offer_local_pgvector(OSError("x")) is True

    assert launched == [1]
    assert answers.choices[1][0].value == "launch"
    assert answers.defaults[1] == "launch"


def test_giving_up_on_docker_stops_the_offer(in_tmp, monkeypatch):
    monkeypatch.setattr(
        pg_docker, "docker_status", lambda run=None: pg_docker.DockerStatus("stopped")
    )
    monkeypatch.setattr(pg_docker, "docker_desktop_command", lambda: None)
    Answers(monkeypatch, "docker", "stop")

    assert main.offer_local_pgvector(OSError("x")) is False
    assert not (in_tmp / ".env").exists()


@pytest.mark.parametrize(
    "state, platform, expected",
    [
        (
            "missing",
            "win32",
            "Install Docker Desktop: https://docs.docker.com/desktop/setup/install/windows-install/",
        ),
        ("missing", "darwin", "brew install --cask docker"),
        (
            "missing",
            "linux",
            "Install Docker Engine: https://docs.docker.com/engine/install/",
        ),
        ("stopped", "win32", "Start Docker Desktop (from the Start menu)"),
        ("stopped", "darwin", "Start Docker Desktop (from Applications)"),
        ("stopped", "linux", "sudo systemctl start docker"),
        ("denied", "linux", "sudo usermod -aG docker $USER"),
    ],
)
def test_guidance_fits_the_platform(state, platform, expected):
    guidance = pg_docker.docker_guidance(pg_docker.DockerStatus(state), platform)

    assert expected in guidance


def test_docker_status_tells_stopped_from_denied():
    class Docker(FakeDocker):
        def __init__(self, stderr):
            super().__init__()
            self.stderr = stderr

        def __call__(self, args, capture=True):
            return self._result(1, stderr=self.stderr)

    stopped = pg_docker.docker_status(Docker("Cannot connect to the Docker daemon"))
    denied = pg_docker.docker_status(
        Docker("permission denied while trying to connect")
    )

    assert stopped.state == "stopped"
    assert denied.state == "denied"


def test_waiting_for_docker_stops_when_it_answers():
    answers = iter([1, 1, 0])

    class Docker(FakeDocker):
        def __call__(self, args, capture=True):
            return self._result(next(answers), "28.5.2")

    naps = []
    status = pg_docker.wait_for_docker(run=Docker(), sleep=naps.append, clock=lambda: 0)

    assert status.ready
    assert len(naps) == 2


def test_waiting_for_docker_gives_up():
    class Docker(FakeDocker):
        def __call__(self, args, capture=True):
            return self._result(1)

    times = iter(range(0, 1000, 60))
    status = pg_docker.wait_for_docker(
        timeout=180, run=Docker(), sleep=lambda _: None, clock=lambda: next(times)
    )

    assert not status.ready


def test_launching_docker_desktop_runs_it_in_the_background(tmp_path, monkeypatch):
    app = tmp_path / "Docker" / "Docker" / "Docker Desktop.exe"
    app.parent.mkdir(parents=True)
    app.touch()
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    started_commands = []

    assert pg_docker.launch_docker_desktop(
        "win32", popen=lambda command, **kw: started_commands.append(command)
    )
    assert started_commands == [[str(app)]]
    assert pg_docker.docker_desktop_command("linux") is None


# --- Resources and memory sizes --------------------------------------------------


def test_the_memory_limit_is_dockers_within_what_is_free():
    assert pg_docker.Resources(docker_memory=64, host_available=32).memory_limit == 32
    assert pg_docker.Resources(docker_memory=16, host_available=32).memory_limit == 16
    assert pg_docker.Resources().memory_limit is None


def test_the_suggestion_follows_free_memory():
    tight = pg_docker.Resources(docker_memory=64 * 2**30, host_available=8 * 2**30)

    assert pg_docker.suggest_build(tight).maintenance_work_mem == "2457MB"


def test_resources_are_described():
    resources = pg_docker.Resources(
        docker_memory=int(62.6 * 2**30),
        docker_cpus=32,
        host_memory=int(127.7 * 2**30),
        host_available=int(90.5 * 2**30),
    )

    assert resources.describe() == (
        "Docker can use 62.6 GB and 32 CPUs; 90.5 GB of this machine's 127.7 GB is free"
    )


@pytest.mark.parametrize(
    "text, size",
    [
        ("8GB", 8 * 2**30),
        ("8g", 8 * 2**30),
        ("8 GiB", 8 * 2**30),
        ("8192MB", 8 * 2**30),
        ("8192", 8 * 2**30),
        ("1.5GB", int(1.5 * 2**30)),
        ("512mb", 512 * 2**20),
    ],
)
def test_memory_sizes_are_read(text, size):
    assert pg_docker.parse_memory(text) == size


@pytest.mark.parametrize("text", ["", "lots", "8 GB please", "-1GB"])
def test_other_text_is_not_a_memory_size(text):
    with pytest.raises(ValueError):
        pg_docker.parse_memory(text)


def test_chosen_settings_fit_a_container_or_not():
    build = pg_docker.custom_build(8 * 2**30, 4)

    assert pg_docker.fits(build, 9 * 2**30)
    assert not pg_docker.fits(build, 2**30)
    assert pg_docker.fits(build, None)


def test_settings_chosen_for_an_existing_container_stay_within_its_shm():
    docker = FakeDocker(
        containers={"aatm-pgvector": {"running": True, "port": 5432, "shm": 2**30}}
    )

    server = pg_docker.start_local_pgvector(
        run=docker,
        wait=no_wait,
        log=lambda _: None,
        build=pg_docker.custom_build(8 * 2**30, 3),
    )

    assert server.build.maintenance_work_mem == "768MB"
    assert server.build.workers == 3  # the chosen workers are kept


def test_recreating_removes_the_container_and_starts_a_bigger_one():
    docker = FakeDocker(
        containers={"aatm-pgvector": {"running": True, "port": 5432, "shm": 2**30}}
    )

    server = pg_docker.start_local_pgvector(
        run=docker,
        wait=no_wait,
        log=lambda _: None,
        build=pg_docker.custom_build(8 * 2**30, 3),
        recreate=True,
    )

    assert ["docker", "rm", "-f", "aatm-pgvector"] in docker.calls
    (run,) = [call for call in docker.calls if call[1] == "run"]
    assert "--shm-size=9216m" in run
    assert "aatm-pgvector:/var/lib/postgresql/data" in run  # the same volume
    assert server.build.maintenance_work_mem == "8192MB"


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
