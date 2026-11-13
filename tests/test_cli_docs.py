"""The command line reference page matches the CLI."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_cli_docs.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_cli_docs", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_reference_is_up_to_date():
    gen = load_generator()
    page = gen.render()
    assert gen.PAGE.read_text(encoding="utf-8") == page, (
        "docs/reference/cli.md is stale: run uv run python scripts/gen_cli_docs.py"
    )


def test_cli_reference_lists_every_visible_command():
    page = load_generator().render()
    for command in (
        "aatm init",
        "aatm doctor",
        "aatm map",
        "aatm search-ui",
        "aatm serve",
        "aatm schema",
        "aatm vector-db check",
        "aatm vector-db upload",
        "aatm cache info",
        "aatm cache clear",
        "aatm omop run",
        "aatm omop rebuild",
        "aatm omop status",
        "aatm omop failures",
        "aatm omop pilot log",
        "aatm omop pilot compare",
    ):
        assert f"`{command}`" in page
    assert "aatm amap" not in page  # hidden: not implemented
    assert "\\" not in page  # no Windows paths in defaults
