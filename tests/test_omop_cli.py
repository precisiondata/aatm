"""`aatm omop run / rebuild / status / failures`, with a fake extractor and vocabulary."""

import json
import logging
import sqlite3

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from aatm import main
from aatm.logs import LOG_LEVEL_ENV_VAR
from aatm.omop import OmopRun
from aatm.omop import cli as omop_cli
from aatm.omop.shards import shard_dir

from test_omop_mapping import FakeLookup, ecog_definitions
from test_omop_runner import FakeExtractor, notes


class InterruptedExtractor(FakeExtractor):
    """Stops as if the user pressed Ctrl+C."""

    async def __call__(self, texts):
        raise KeyboardInterrupt


class Extractors:
    """The fake extractors the CLI made, and the texts that fail."""

    def __init__(self):
        self.made = []
        self.fail = set()
        self.interrupt = False

    @property
    def calls(self):
        return sum(e.calls for e in self.made)


@pytest.fixture
def extractors(monkeypatch):
    made = Extractors()

    def make_extractor(options, definitions):
        if options.model is None:
            raise omop_cli.OptionsError("Give the extraction model with --model")
        kind = InterruptedExtractor if made.interrupt else FakeExtractor
        extractor = kind(definitions, made.fail)
        extractor.model_id = options.model
        made.made.append(extractor)
        return extractor

    monkeypatch.setattr(omop_cli, "make_extractor", make_extractor)
    monkeypatch.setattr(omop_cli, "make_vocabulary", lambda options: FakeLookup())
    return made


@pytest.fixture
def cli(monkeypatch):
    """Invoke the CLI without leaking its logging setup into other tests."""
    logger = logging.getLogger("aatm")
    handlers, level = list(logger.handlers), logger.level
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "WARNING")
    yield lambda *args: CliRunner().invoke(main.app, ["omop", *map(str, args)])
    logger.handlers[:] = handlers
    logger.setLevel(level)


def write_definitions(path, **fields):
    data = ecog_definitions(**fields).model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    path.write_text(yaml.safe_dump(data["entities"], sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path):
    """A Parquet file of 6 notes and an ECOG definitions file."""
    source = tmp_path / "notes.parquet"
    frame = pd.DataFrame(notes(6)).rename(columns={"text": "note_text"})
    frame.to_parquet(source)
    definitions = write_definitions(tmp_path / "ecog.yaml")
    return source, definitions, tmp_path / "out"


def run_args(source, definitions, out, *extra):
    return (
        "run",
        source,
        out,
        "--definitions",
        definitions,
        "--model",
        "fake/model",
        "--columns",
        "note_text=text",
        "--chunk-size",
        "2",
        *extra,
    )


def test_run_and_resume(cli, extractors, setup):
    source, definitions, out = setup

    result = cli(*run_args(source, definitions, out))

    assert result.exit_code == 0, result.output
    summary_path = result.stdout.strip()
    assert summary_path.endswith("-run.json")
    assert (out / "runs").is_dir()
    assert len(OmopRun(out).table("measurement")) == 9
    assert extractors.calls == 6
    recorded = yaml.safe_load((out / omop_cli.OPTIONS_FILE).read_text())
    assert recorded["model"] == "fake/model"
    assert recorded["columns"] == {"note_text": "text"}

    # The same command again resumes: every note is already built.
    result = cli(*run_args(source, definitions, out, "--json"))

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert (summary["skipped"], summary["built"]) == (6, 0)
    assert summary["summary_path"].endswith("-run.json")
    assert summary["summary_path"] != summary_path
    assert extractors.calls == 6


def test_failed_notes_exit_with_2_and_are_listed(cli, extractors, setup):
    source, definitions, out = setup
    extractors.fail.add(notes(6)[2]["text"])

    result = cli(*run_args(source, definitions, out))

    assert result.exit_code == 2, result.output
    assert "1 notes failed" in result.stderr

    result = cli("failures", out)
    assert result.exit_code == 0, result.output
    assert "API down" in result.stdout
    assert "1 failed notes" in result.stdout

    failures = json.loads(cli("failures", out, "--json").stdout)
    assert [f["note_id"] for f in failures] == [3]

    result = cli("status", out)
    assert result.exit_code == 0, result.output
    assert "5 built" in result.stdout and "1 failed" in result.stdout
    assert "measurement" in result.stdout

    # Once the API is back, running again builds the note: exit 0.
    extractors.fail.clear()
    result = cli(*run_args(source, definitions, out))
    assert result.exit_code == 0, result.output
    assert OmopRun(out).status() == {"built": 6}
    assert "No failed notes" in cli("failures", out).stdout


def test_rebuild_uses_the_recorded_options(cli, extractors, setup):
    source, definitions, out = setup
    assert cli(*run_args(source, definitions, out)).exit_code == 0

    # A new omop block: the value source is gone.
    write_definitions(definitions, value_source_value={"value": "x"})
    result = cli("rebuild", out, "--definitions", definitions, "--json")

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["kind"] == "rebuild"
    assert summary["built"] == 6
    assert extractors.calls == 6  # no extraction call
    # The model was taken from options.yaml.
    assert extractors.made[-1].model_id == "fake/model"
    assert set(OmopRun(out).table("measurement")["value_source_value"]) == {"x"}


def test_rebuild_needs_a_run(cli, extractors, tmp_path):
    result = cli("rebuild", tmp_path / "nothing", "--definitions", "x")
    assert result.exit_code == 1
    assert "No such folder" in result.stderr


def test_a_shard_goes_to_its_folder(cli, extractors, setup):
    source, definitions, out = setup

    for index in range(2):
        result = cli(*run_args(source, definitions, out, "--shard", f"{index}/2"))
        assert result.exit_code == 0, result.output
        assert shard_dir(out, index, 2).name in result.stdout

    run = OmopRun(out)
    assert run.sharded
    assert run.status() == {"built": 6}
    status = json.loads(cli("status", out, "--json").stdout)
    assert status["notes"] == {"built": 6}
    assert status["last_summary"]["notes"] == 6

    result = cli("rebuild", out)
    assert result.exit_code == 0, result.output


def test_limit_runs_the_first_notes(cli, extractors, setup):
    source, definitions, out = setup

    result = cli(*run_args(source, definitions, out, "--limit", "3", "--json"))

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["built"] == 3
    assert OmopRun(out).status() == {"built": 3}


def test_a_config_file_with_flags_over_it(cli, extractors, setup, tmp_path):
    source, definitions, out = setup
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "source": str(source),
                "output": str(out),
                "definitions": str(definitions),
                "model": "fake/model",
                "columns": {"note_text": "text"},
                "chunk-size": 4,
                "limit": 2,
                "prices": {"fake/model": [1.0, 2.0]},
            }
        ),
        encoding="utf-8",
    )

    result = cli("run", "--config", config, "--limit", "5", "--json")

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["built"] == 5
    assert summary["settings"]["chunk_size"] == 4


def test_a_misspelt_config_key_fails(cli, extractors, setup, tmp_path):
    source, definitions, out = setup
    config = tmp_path / "run.yaml"
    config.write_text("chunk_sise: 4\n", encoding="utf-8")

    result = cli(*run_args(source, definitions, out, "--config", config))

    assert result.exit_code == 1
    assert "chunk_sise" in result.stderr


def test_notes_from_a_query(cli, extractors, setup, tmp_path):
    _, definitions, out = setup
    db = tmp_path / "cdm.db"
    con = sqlite3.connect(db)
    pd.DataFrame(notes(4)).rename(columns={"text": "note_text"}).to_sql("note", con)
    con.close()

    result = cli(
        "run",
        out,
        "--sql",
        "SELECT note_id, note_text, person_id, note_date FROM note",
        "--db",
        f"sqlite:///{db}",
        "--definitions",
        definitions,
        "--model",
        "fake/model",
        "--columns",
        "note_text=text",
        "--json",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["built"] == 4


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--price", "fake/model=0.25"), "--price"),
        (("--shard", "3"), "--shard"),
        (("--shard", "8/8"), "outside"),
        (("--coder", "magic"), "coder"),
        (("--selector", "x"), "--coder mapper"),
        (("--min-probability", "0.9"), "--labeller"),
    ],
)
def test_bad_options_exit_with_1(cli, extractors, setup, extra, message):
    source, definitions, out = setup

    result = cli(*run_args(source, definitions, out, *extra))

    assert result.exit_code == 1, result.output
    assert message in result.stderr
    assert extractors.calls == 0


def test_options_that_are_needed(cli, extractors, setup):
    source, definitions, out = setup

    result = cli("run", source, out, "--model", "fake/model")
    assert result.exit_code == 1
    assert "--definitions" in result.stderr

    result = cli("run", source, out, "--definitions", definitions)
    assert result.exit_code == 1
    assert "--model" in result.stderr

    result = cli("run", out / "missing.parquet", out, "-d", definitions, "-m", "x")
    assert result.exit_code == 1
    assert "No such file" in result.stderr


def test_ctrl_c_stops_with_130(cli, extractors, setup):
    source, definitions, out = setup
    extractors.interrupt = True

    result = cli(*run_args(source, definitions, out))

    assert result.exit_code == 130, result.output
    assert "run the same command again" in result.stderr.lower()


def test_status_of_a_folder_that_is_not_a_run(cli, tmp_path):
    result = cli("status", tmp_path)
    assert result.exit_code == 1
    assert "not the output of a run" in result.stderr


def test_parsing():
    assert omop_cli.parse_prices(["openai/gpt-5-mini=0.25,2.00"]) == {
        "openai/gpt-5-mini": (0.25, 2.0)
    }
    assert omop_cli.parse_pairs(["a=b", "c = d"], "--columns") == {"a": "b", "c": "d"}
    assert omop_cli.parse_shard("3/8") == (3, 8)


def test_min_score_applies_to_map_sources_without_one(tmp_path):
    definitions = write_definitions(
        tmp_path / "ecog.yaml",
        value_source_value={"from": "text", "map": {"domain": "Measurement"}},
        measurement_source_value={
            "from": "text",
            "map": {"domain": "Measurement", "min_score": 0.3},
        },
    )
    options = omop_cli.RunOptions(definitions=[str(definitions)], min_score=0.8)

    loaded = omop_cli.load_definitions(options)

    fields = loaded.entities[0].omop.fields
    assert fields["value_source_value"][0].map.min_score == 0.8
    assert fields["measurement_source_value"][0].map.min_score == 0.3


def test_help_shows_every_default(cli):
    result = CliRunner().invoke(
        main.app, ["omop", "run", "--help"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0
    for option in ("--chunk-size", "--max-attempts", "--coder", "--poll-every"):
        assert option in result.output
    assert "[default: (100)]" in result.output
