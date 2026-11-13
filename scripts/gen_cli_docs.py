"""Write the command line reference, docs/reference/cli.md, from the CLI itself.

    uv run python scripts/gen_cli_docs.py           # write the page
    uv run python scripts/gen_cli_docs.py --check   # exit 1 if the page is stale

Every command, argument and option comes from the Typer app in `aatm.main`, so the
page lists what `aatm <command> --help` shows. `tests/test_cli_docs.py` fails when
the page and the CLI differ: run this script after changing a command.
"""

import sys
from pathlib import Path
from typing import Any

import typer
from typer.cli import MARKUP_MODE_KEY, get_docs_for_click

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reference" / "cli.md"

INTRO = """\
# Command line reference

Every `aatm` command, with its arguments and options. This page is generated from the
CLI itself (`scripts/gen_cli_docs.py`), so it matches `aatm <command> --help` for this
version. For how to use the commands together, see the
[high-level guide](../getting-started/high_level.md).

| Workflow | Commands |
|---|---|
| Setup | [`aatm init`](#aatm-init), [`aatm doctor`](#aatm-doctor), [`aatm vector-db check`](#aatm-vector-db-check), [`aatm vector-db upload`](#aatm-vector-db-upload) |
| Clinical notes to OMOP tables | [`aatm omop run`](#aatm-omop-run), [`aatm omop status`](#aatm-omop-status), [`aatm omop failures`](#aatm-omop-failures), [`aatm omop rebuild`](#aatm-omop-rebuild), [`aatm omop pilot log`](#aatm-omop-pilot-log), [`aatm omop pilot compare`](#aatm-omop-pilot-compare), [`aatm cache info`](#aatm-cache-info), [`aatm cache clear`](#aatm-cache-clear) |
| Entity definitions | [`aatm schema`](#aatm-schema) |
| Terminology mapping | [`aatm map`](#aatm-map), [`aatm search-ui`](#aatm-search-ui), [`aatm serve`](#aatm-serve) |

Global options, such as `--log-level`, go before the command name:
`aatm --log-level DEBUG omop run ...`.

`aatm omop run` exits with `0` when every note is built, `2` when the run completed
but some notes failed, `1` on an error and `130` when stopped with Ctrl+C, so
schedulers can act on it.

"""


def _drop_hidden(command: Any) -> None:
    """Remove hidden commands, which `--help` does not list either."""
    commands = getattr(command, "commands", None)  # a group
    if commands:
        for name, sub in list(commands.items()):
            if sub.hidden:
                del commands[name]
            else:
                _drop_hidden(sub)


def render() -> str:
    """The page, as the CLI describes itself."""
    from aatm.main import app

    command = typer.main.get_command(app)
    _drop_hidden(command)
    ctx = typer.Context(command)
    ctx.ensure_object(dict)
    ctx.obj[MARKUP_MODE_KEY] = app.rich_markup_mode
    body = get_docs_for_click(obj=command, ctx=ctx, name="aatm", indent=1)
    return INTRO + body.strip() + "\n"


def main(argv: list[str]) -> int:
    page = render()
    if "--check" in argv:
        if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != page:
            print(
                f"{PAGE} is stale: run python scripts/gen_cli_docs.py", file=sys.stderr
            )
            return 1
        return 0
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f"Wrote {PAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
