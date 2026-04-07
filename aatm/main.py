import typer
from rich.console import Console
from pathlib import Path
import questionary
from questionary import Choice
import dotenv


from .local_database_utils import (
    build_local_sqlite_vocab_database,
    build_local_vector_database,
    build_mapping_datasets,
)

dotenv.load_dotenv()  # Load environment variables

app = typer.Typer()
console = Console()

LOCAL_HELPER_PATH = Path(".aatm")
DEFAULT_VOCAB_DIR = Path("vocabularies")
DEFAULT_EMBEDDING_MODEL = "embeddinggemma-300M"
SUPPORTED_EMBEDDING_MODELS = [
    "qwen3-06B",
    "qwen3-4B",
    "gemini-embedding-001",
    "embeddinggemma-300M",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

AATM_LOGO = r"""
   █████╗  █████╗ ████████╗███╗   ███╗
  ██╔══██╗██╔══██╗╚══██╔══╝████╗ ████║
  ███████║███████║   ██║   ██╔████╔██║
  ██╔══██║██╔══██║   ██║   ██║╚██╔╝██║
  ██║  ██║██║  ██║   ██║   ██║ ╚═╝ ██║
  ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝
"""


def print_logo() -> None:
    console.print(f"[bold bright_blue]{AATM_LOGO}[/bold bright_blue]")
    console.print(
        "[blue]Welcome to the Any-to-Any Terminology Mapping Project![/blue]\n"
    )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        print_logo()
        console.print("[dim]Use 'aatm --help' to see available commands.[/dim]")


@app.command(
    help="Set up your environment to work with OMOP vocabularies and create terminology mappings."
)
def init(
    vocab_dir: str = typer.Option(
        None,
        "--vocab-dir",
        "-vd",
        help="Path to the OMOP vocabularies directory.",
        show_default=False,
    ),
) -> None:
    # Setup local directories
    LOCAL_HELPER_PATH.mkdir(exist_ok=True, parents=True)
    gitignore_path = Path(".gitignore")
    entry = ".aatm"
    if not gitignore_path.exists():
        gitignore_path.write_text(entry + "\n")
    else:
        lines = {line.strip() for line in gitignore_path.read_text().splitlines()}

        if entry not in lines:
            with gitignore_path.open("a") as f:
                f.write(
                    "\n" + entry
                    if not gitignore_path.read_text().endswith("\n")
                    else entry + "\n"
                )

    print_logo()
    console.print("Let's set up your environment!\n")

    if vocab_dir is None:
        vocab_dir: Path = DEFAULT_VOCAB_DIR
    else:
        vocab_dir = Path(vocab_dir)
        if not vocab_dir.exists():
            console.print(
                f"[yellow]OOPS![/yellow] You specified a non-existent directory: `{vocab_dir}`. Please, check the path and try again or download the desired OMOP vocabularies at https://athena.ohdsi.org/ if you haven't yet.\n"
            )
            raise typer.Exit()

    if not vocab_dir.exists():
        console.print(
            "[yellow]OOPS![/yellow] You still don't have a `vocabularies` directory. Download the desired OMOP vocabularies at https://athena.ohdsi.org/ and place them in the `./vocabularies` directory or use the `--vocab-dir` option to specify a different directory.\n"
        )
        raise typer.Exit()

    console.print("[blue]1) Building local OMOP vocabulary database[/blue]")
    console.print(f"Using vocabulary files from: '{vocab_dir}'")

    build_local_sqlite_vocab_database(vocab_dir)
    console.print("[green]Done![/green]\n")

    console.print("[blue]2) Building local vector database[/blue]")

    supported_embedding_models_choices = [
        Choice(f"{model_name} (default)", value=model_name)
        if model_name == DEFAULT_EMBEDDING_MODEL
        else Choice(model_name, value=model_name)
        for model_name in SUPPORTED_EMBEDDING_MODELS
    ]
    selected_embedding_model = questionary.select(
        "Select one of the supported embedding models for building the vector database:",
        choices=supported_embedding_models_choices,
        default=DEFAULT_EMBEDDING_MODEL,
    ).ask()

    selected_standard_vocabularies = questionary.checkbox(
        "Select one or more standard vocabularies to use for mapping:",
        choices=[
            Choice("LOINC", value="LOINC", checked=True),
            Choice("RxNorm", value="RxNorm", checked=True),
            Choice("SNOMED", value="SNOMED", checked=True),
        ],
    ).ask()

    build_mapping_datasets(selected_standard_vocabularies)

    build_local_vector_database(selected_embedding_model)
    console.print("[green]Done![/green]\n")
