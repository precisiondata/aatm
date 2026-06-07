"""
This module defines the main Typer application for the Any-to-Any
Terminology Mapping (AATM) package. It provides commands for
initializing the local project environment, launching the search user
interface, and running terminology mapping tasks from either a config
file or command-line options.

The CLI is responsible for orchestrating high-level workflows such as:

- setting up local helper directories and Git ignore rules
- validating vocabulary and embedding model inputs
- building local SQLite and vector databases for terminology mapping
- collecting interactive user input for setup choices
- loading and displaying mapping task configurations
- starting synchronous terminology mapping runs
- launching the Streamlit-based search interface

**Commands**:

- `init`:
    Set up the local environment for OMOP vocabulary processing and
    vector database creation.

- `search-ui`:
    Launch the Streamlit-based search interface.

- `map`:
    Run a terminology mapping task from a configuration file or from
    explicit CLI arguments.

- `amap`:
    Placeholder for future asynchronous mapping support.

- `serve`:
    Serves a minimal FastAPI application with AATM's functionality.

Attributes:
    LOCAL_HELPER_PATH (Path): Path to the local helper directory used by AATM
        to store generated artifacts and local resources.
    DEFAULT_VOCAB_DIR (Path): Default directory containing OMOP vocabulary
        files.
    DEFAULT_EMBEDDING_MODEL (str): Default embedding model used when building
        the local vector database.
    SUPPORTED_EMBEDDING_MODELS (List[str]): List of embedding model identifiers
        supported by the CLI setup workflow.
    STANDARD_VOCABULARIES (List[str]): List of supported standard vocabularies
        available for terminology mapping.
    DEFAULT_STANDARD_VOCABULARIES (List[str]): Default subset of standard
        vocabularies selected during initialization.
    AATM_LOGO (str): Multiline ASCII logo displayed in the CLI welcome
        screen.        

Examples:
    Initialize the local environment with defaults:

        $ aatm init

    Initialize with a custom vocabulary directory:

        $ aatm init --vocab-dir ./vocabularies

    Run a mapping task from a config file:

        $ aatm map --task-config-path config.yaml

    Run a mapping task from explicit options:

        $ aatm map --input-file concepts.csv --output-dir outputs \\
            --translator-id my_translator --retriever-id my_retriever \\
            --selector-id my_selector

    Launch the search UI:

        $ aatm search-ui

Note:
    This module is intended to serve as the main CLI entrypoint for the
    package. It coordinates user interaction and delegates implementation
    details to lower-level utilities and domain-specific components.
"""

import logging
import dotenv
import typer
from rich.console import Console
from pathlib import Path
from typing import Annotated, List, Literal, Optional
import questionary
from questionary import Choice
import subprocess
import sys
from rich.json import JSON

from aatm.logs import get_logger


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
    """Initialize the local AATM environment.

    This command prepares the local project environment required for
    terminology mapping with OMOP vocabularies. It creates the helper
    directory used by AATM, ensures `.aatm` is listed in `.gitignore`,
    validates user inputs, builds the local SQLite vocabulary database,
    prepares mapping datasets for the selected standard vocabularies,
    and creates the local vector database used for retrieval.

    When `embedding_model` or `standard_vocabs` are not provided, the
    command interactively prompts the user to choose among the supported
    options.

    Args:
        vocab_dir: Path to the directory containing the OMOP vocabulary
            files. If not provided, the default vocabulary directory is
            used.
        embedding_model: Name of the embedding model to use for building
            the local vector database. If not provided, the user is
            prompted to select one interactively.
        standard_vocabs: List of standard vocabularies to include in the
            mapping datasets. If not provided, the user is prompted to
            select one or more interactively.

    Raises:
        typer.Exit: If the vocabulary directory does not exist, if the
            embedding model is not supported, or if any provided
            standard vocabulary is invalid.

    Examples:
        Initialize using the default vocabulary directory:

            $ aatm init

        Initialize with a custom vocabulary directory:

            $ aatm init --vocab-dir ./vocabularies

        Initialize with an explicit embedding model:

            $ aatm init --embedding-model qwen3-4B

        Initialize with specific standard vocabularies:

            $ aatm init --standard-vocabs LOINC --standard-vocabs SNOMED
    """
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

    # Lazy import
    console.print("Preparing local database builder...")
    from aatm.local_database_utils import build_local_sqlite_vocab_database

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

    # Lazy import
    console.print("Preparing mapping datasets...")
    from aatm.local_database_utils import build_mapping_datasets

    build_mapping_datasets(selected_standard_vocabularies)

    # Lazy import
    console.print("Preparing local vector database builder...")
    from aatm.local_database_utils import build_local_vector_database

    build_local_vector_database(selected_embedding_model)
    console.print("[green]Done![/green]\n")


@app.command("search-ui")
def search_ui() -> None:
    """Launch the Streamlit-based search interface.

    This command locates the `search_ui.py` application in the same
    package directory as this module and starts it using the current
    Python interpreter with Streamlit.

    Raises:
        typer.BadParameter: If the Streamlit application file cannot be
            found.
        subprocess.CalledProcessError: If the Streamlit process exits
            with a non-zero status code.
    """
    streamlit_app = Path(__file__).resolve().parent / "search_ui.py"

    if not streamlit_app.exists():
        raise typer.BadParameter(f"Streamlit app not found: {streamlit_app}")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(streamlit_app)],
        check=True,
    )


@app.command("map", help="Run a terminology mapping task")
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
    limit_to: Annotated[
        Optional[int],
        typer.Option(
            "--limit-to",
            "-l",
            help="Limit the number of source concepts to map. Useful for testing and debugging.",
        ),
    ] = None,
) -> None:
    """Run a terminology mapping task.

    This command executes a terminology mapping workflow using either a
    configuration file or parameters provided directly through the command
    line. When a task configuration file is given, it is loaded and used to
    build the mapping task. Otherwise, a `TerminologyMappingTask` is created
    from the individual command options.

    The command prints the loaded task configuration, instantiates a
    `TerminologyMapper` from it, and starts the mapping process.

    Args:
        task_config_path (Optional[str]): Path to a task configuration file.
            When provided, the mapping task is loaded from this file.
        input_file (Optional[str]): Path to the input file containing source
            concepts to be mapped. Used when no task configuration file is
            provided.
        output_dir (Optional[str]): Path to the output directory where
            mapping results will be written. Used when no task configuration
            file is provided.
        translator_id (Optional[str]): Identifier of the translator to use
            in the mapping pipeline.
        retriever_id (Optional[str]): Identifier of the retriever to use in
            the mapping pipeline.
        selector_id (Optional[str]): Identifier of the selector to use in
            the mapping pipeline.
        reranker_id (Optional[str]): Identifier of the reranker to use in
            the mapping pipeline.
        batch_size (Optional[int]): Batch size to use during the mapping
            process.
        rate_limit (Optional[int]): Rate limit to apply when processing
            source concepts.
        limit_to (Optional[int]): Maximum number of source concepts to map.
            Useful for testing and debugging.

    Raises:
        typer.Exit: If `task_config_path` is provided but does not exist.

    Examples:
        Run a mapping task from a config file:

            $ aatm map --task-config-path config.yaml

        Run a mapping task from explicit options:

            $ aatm map --input-file concepts.csv --output-dir outputs \\
                --translator-id my_translator --retriever-id my_retriever \\
                --selector-id my_selector

        Run a small test mapping job:

            $ aatm map --input-file concepts.csv --output-dir outputs \\
                --limit-to 20
    """

    # Lazy import
    from aatm.terminology_mapper import TerminologyMapper
    from .data_models import TerminologyMappingTask

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
            limit_to=limit_to,
        )

    print_logo()

    logger.debug(f"Loaded task config: {task_config} {task_config.model_dump()}")

    console.print("[blue]Loaded mapping task config:[/blue]")
    console.print(JSON(task_config.model_dump_json()), "\n")

    console.print("[blue]Started mapping...[/blue]")
    terminology_mapper = TerminologyMapper.from_task_config(task_config)
    terminology_mapper.map()

    console.print("[green]Done![/green]\n")


@app.command("amap", help="Run a terminology mapping task with asynchronous methods")
def amap() -> None:
    raise NotImplementedError


@app.command("serve", help="Serve a FastAPI application with AATM's functionality")
def serve(
    host: Annotated[
        Optional[str],
        typer.Option(
            "--host",
            "-h",
            help="Host",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        str,
        typer.Option(
            "--port",
            "-p",
            help="Port",
        ),
    ] = "8000",
    mode: Annotated[
        Literal["prod", "dev"],
        typer.Option("--mode", "-m", help="Serving mode: 'prod' or 'dev'"),
    ] = "dev",
    reload: Annotated[
        bool | None,
        typer.Option(
            "--reload", help="Enables hot reload. Compatible with 'dev' mode only."
        ),
    ] = False,
    workers: Annotated[
        Optional[str],
        typer.Option(
            "--workers",
            "-w",
            help="Number of workers",
        ),
    ] = None,
    rate_limit: Annotated[
        Optional[int],
        typer.Option(
            "--rate-limit",
            "-r",
            help="Maximum number of documents allowed per minute",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size",
        ),
    ] = 100,
) -> None:
    """Serve a FastAPI application exposing AATM functionality.

    This command validates the serving configuration, persists the API settings to
    disk, and launches the FastAPI application using the appropriate runtime mode.
    It supports development and production execution, optional hot reload, worker
    configuration, and request-processing parameters such as rate limits and batch
    size.

    Args:
        host: Host interface on which the FastAPI application will listen.
        port: Port on which the FastAPI application will listen.
        mode: Serving mode. Use `"dev"` for development and `"prod"` for
            production.
        reload: Whether to enable hot reload. This option is only effective in
            development mode.
        workers: Number of worker processes to use. This option is only effective
            in production mode.
        rate_limit: Maximum number of documents allowed per minute.
        batch_size: Batch size used by the API processing pipeline.

    Returns:
        None.

    Raises:
        typer.BadParameter: If the FastAPI application entrypoint cannot be found.
        subprocess.CalledProcessError: If the FastAPI process exits with a
            non-zero status.
    """
    api_main_file = Path(__file__).resolve().parent / Path("api/main.py")

    if not api_main_file.exists():
        raise typer.BadParameter(
            f"FastAPI application implementation was not found: {api_main_file}"
        )

    if reload and mode != "dev":
        console.print(
            f"[yellow]Attention:[/yellow] You provided the flag '--reload' and the mode '{mode}'. Hot reload is only enabled in 'dev' mode, so it will be ignored.\n"
        )

    print_logo()

    # Lazy import
    from aatm.api.config import APIConfig

    # Save API configuration to disk
    api_config = APIConfig(
        host=host,
        port=port,
        rate_limit=rate_limit,
        batch_size=batch_size,
        workers=workers,
    )
    api_config.save_to_disk()

    command = [
        sys.executable,
        "-m",
        "fastapi",
        "run" if mode == "prod" else "dev",
        # default entrypoint is defined at pyproject.toml
        "--host",
        host,
        "--port",
        port,
    ]

    if workers and mode != "prod":
        console.print(
            f"[yellow]Attention:[/yellow] You provided the flag '--workers' and the mode '{mode}'. Workers are only enabled in 'prod' mode, so it will be ignored.\n"
        )
    elif workers:
        command.append("--workers")
        command.append(workers)

    if reload and mode == "dev":
        command.append("--reload")

    subprocess.run(
        command,
        check=True,
    )
