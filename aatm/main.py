import typer
from rich.console import Console
from pathlib import Path
import questionary
from questionary import Choice

from .local_database_utils import build_local_sqlite_vocab_database
from .retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY

app = typer.Typer()
console = Console()

DEFAULT_VOCAB_DIR = Path("vocabularies")
DEFAULT_EMBEDDING_MODEL = "embeddinggemma-300M"

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
    console.print("[blue]Welcome to the Any-to-Any Terminology Mapping Project![/blue]\n")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        print_logo()
        console.print("[dim]Use 'aatm --help' to see available commands.[/dim]")

@app.command()
def init(
    vocab_dir: str = typer.Option(None, "--vocab-dir", "-vd", help="Path to the OMOP vocabularies directory.", show_default=False),
    ) -> None:
    print_logo()
    console.print("Let's set up your environment!\n")

    if vocab_dir is None:
        vocab_dir: Path = DEFAULT_VOCAB_DIR
    else:
        vocab_dir = Path(vocab_dir)
        if not vocab_dir.exists():
            console.print(f"[yellow]OOPS![/yellow] You specified a non-existent directory: `{vocab_dir}`. Please, check the path and try again or download the desired OMOP vocabularies at https://athena.ohdsi.org/ if you haven't yet.\n")
            raise typer.Exit()

    if not vocab_dir.exists():
        console.print("[yellow]OOPS![/yellow] You still don't have a `vocabularies` directory. Download the desired OMOP vocabularies at https://athena.ohdsi.org/ and place them in the `./vocabularies` directory or use the `--vocab-dir` option to specify a different directory.\n")
        raise typer.Exit()

    console.print("[blue]1) Building local OMOP vocabulary database[/blue]")
    console.print(f"Using vocabulary files from: '{vocab_dir}'")

    #build_local_sqlite_vocab_database(vocab_dir)
    console.print("[green]Done![/green]\n")

    console.print("[blue]2) Building local vector database[/blue]")
    supported_embedding_models = list(CHROMADB_RETRIEVER_MODEL_REGISTRY.keys())
    supported_embedding_models_choices = [
        Choice(f"{model_name} (default)", value=model_name) if model_name == DEFAULT_EMBEDDING_MODEL else Choice(model_name, value=model_name) for model_name in supported_embedding_models 
    ]
    selected_embedding_model = questionary.select(
    "Select one of the supported embedding models for building the vector database:",
    choices=supported_embedding_models_choices,
    default=DEFAULT_EMBEDDING_MODEL).ask()