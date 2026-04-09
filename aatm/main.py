import logging
import subprocess
import sys
from typing import Annotated, List, Optional
from rich.json import JSON

import typer
from rich.console import Console
from pathlib import Path
import questionary
from questionary import Choice
import dotenv

from aatm.terminology_mapper import TerminologyMapper

from .data_models import TerminologyMappingTask
from .logs import get_logger
from .local_database_utils import (
    build_local_sqlite_vocab_database,
    build_local_vector_database,
    build_mapping_datasets,
)

dotenv.load_dotenv()  # Load environment variables
logger = get_logger(__name__, level=logging.INFO)

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
STANDARD_VOCABULARIES = ["LOINC", "SNOMED", "RxNorm"]
DEFAULT_STANDARD_VOCABULARIES = ["LOINC", "SNOMED", "RxNorm"]

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
    vocab_dir: Annotated[
        str,
        typer.Option(
            "--vocab-dir",
            "-vd",
            help="Path to the OMOP vocabularies directory.",
            show_default=False,
        ),
    ] = None,
    embedding_model: Annotated[
        Optional[str],
        typer.Option(
            "--embedding-model",
            "-e",
            help="Name of the embedding model to use.",
            show_default=False,
        ),
    ] = None,
    standard_vocabs: Annotated[
        Optional[List[str]],
        typer.Option(
            "--standard-vocabs",
            "-sv",
            help=f"List of standard vocabularies to use. Available options: {STANDARD_VOCABULARIES}.",
        ),
    ] = None,
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

    # Validate vocab directory
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

    # Validate embedding model
    if embedding_model is not None:
        if embedding_model not in SUPPORTED_EMBEDDING_MODELS:
            console.print(
                f"[yellow]OOPS![/yellow] You specified an unsupported embedding model: `{embedding_model}`. Please, check the model name and try again.\nThe supported embedding models are: {SUPPORTED_EMBEDDING_MODELS}\n"
            )
            raise typer.Exit()

    # Validate provided standard vocabs
    if standard_vocabs is not None:
        if not set(standard_vocabs).issubset(STANDARD_VOCABULARIES):
            console.print(
                f"[yellow]OOPS![/yellow] You specified standard vocabularies that are not available. Available options: {STANDARD_VOCABULARIES}. You provided: {standard_vocabs}\n"
            )
            raise typer.Exit()

    console.print("[blue]1) Building local OMOP vocabulary database[/blue]")
    console.print(f"Using vocabulary files from: '{vocab_dir}'")

    build_local_sqlite_vocab_database(vocab_dir)
    console.print("[green]Done![/green]\n")

    console.print("[blue]2) Building local vector database[/blue]")

    if embedding_model is None:
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
    else:
        console.print(f"Using embedding model: {embedding_model}")
        selected_embedding_model = embedding_model

    if standard_vocabs is None:
        selected_standard_vocabularies = questionary.checkbox(
            "Select one or more standard vocabularies to use for mapping:",
            choices=[
                Choice(
                    vocab, value=vocab, checked=vocab in DEFAULT_STANDARD_VOCABULARIES
                )
                for vocab in STANDARD_VOCABULARIES
            ],
        ).ask()
    else:
        console.print(f"Using standard vocabularies: {standard_vocabs}")
        selected_standard_vocabularies = standard_vocabs

    build_mapping_datasets(selected_standard_vocabularies)

    build_local_vector_database(selected_embedding_model)
    console.print("[green]Done![/green]\n")


@app.command("search-ui")
def search_ui() -> None:
    """Launch the Streamlit search UI."""
    streamlit_app = Path(__file__).resolve().parent / "search_ui.py"

    if not streamlit_app.exists():
        raise typer.BadParameter(f"Streamlit app not found: {streamlit_app}")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(streamlit_app)],
        check=True,
    )


@app.command("map")
def map(
    task_config_path: Annotated[
        Optional[str],
        typer.Option(
            "--task-config-path",
            "-t",
            help="Path to the task config file",
        ),
    ] = None,
    input_file: Annotated[
        Optional[str],
        typer.Option(
            "--input-file",
            "-i",
            help="Path to the input file containing source concepts",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            "--output-dir",
            "-o",
            help="Path to the output directory",
        ),
    ] = None,
    translator_id: Annotated[
        Optional[str],
        typer.Option(
            "--translator-id",
            "-tr",
            help="ID of the translator to use",
        ),
    ] = None,
    retriever_id: Annotated[
        Optional[str],
        typer.Option(
            "--retriever-id",
            "-r",
            help="ID of the retriever to use",
        ),
    ] = None,
    selector_id: Annotated[
        Optional[str],
        typer.Option(
            "--selector-id",
            "-s",
            help="ID of the selector to use",
        ),
    ] = None,
    reranker_id: Annotated[
        Optional[str],
        typer.Option(
            "--reranker-id",
            "-rr",
            help="ID of the reranker to use",
        ),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size to use when mapping source concepts",
        ),
    ] = None,
    rate_limit: Annotated[
        Optional[int],
        typer.Option(
            "--rate-limit",
            "-rl",
            help="Rate limit to use when mapping source concepts",
        ),
    ] = None,
) -> None:
    if task_config_path is not None:
        task_config_path = Path(task_config_path)
        if not task_config_path.exists():
            console.print(
                f"[yellow]OOPS![/yellow] You specified a non-existent task config file: `{task_config_path}`. Please, check the path and try again.\n"
            )
            raise typer.Exit()

        task_config = TerminologyMappingTask.from_config_file(task_config_path)
    else:
        task_config = TerminologyMappingTask(
            input_file=Path(input_file) if input_file else None,
            output_dir=Path(output_dir) if output_dir else None,
            translator_id=translator_id,
            retriever_id=retriever_id,
            selector_id=selector_id,
            reranker_id=reranker_id,
            batch_size=batch_size,
            rate_limit=rate_limit,
        )

    print_logo()

    logger.debug(f"Loaded task config: {task_config} {task_config.model_dump()}")

    console.print("[blue]Loaded mapping task config:[/blue]")
    console.print(JSON(task_config.model_dump_json()), "\n")

    console.print("[blue]Started mapping...[/blue]")
    terminology_mapper = TerminologyMapper.from_task_config(task_config)
    terminology_mapper.map()

    console.print("[green]Done![/green]\n")


@app.command("amap")
def amap() -> None:
    pass
