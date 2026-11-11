"""Download, unpack and check Athena vocabulary bundles.

Athena (https://athena.ohdsi.org) has no API for requesting a vocabulary bundle:
a logged-in user picks the vocabularies, accepts their licences and Athena emails
a link to a zip file. This module starts from that link, or from the zip once
downloaded, and prepares the directory of CSVs that `aatm init` builds the local
databases from:

- `download_bundle` streams the zip to disk, resuming after network errors, and
  checks that the result is a zip file rather than a login page.
- `unpack_bundle` extracts it without overwriting files unless asked to.
- `missing_required_files` and `read_bundle_vocabularies` check what it contains.
- `has_cpt4_tooling` and `run_cpt4` add the CPT4 concept names, which Athena
  leaves out for licensing reasons, using the user's UMLS API key.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from aatm.logs import get_logger

logger = get_logger(__name__)

ATHENA_URL = "https://athena.ohdsi.org/vocabulary/list"
UMLS_API_KEY_ENV_VAR = "UMLS_API_KEY"

# Tables `aatm init` needs to build the vocabulary database and mapping datasets.
REQUIRED_VOCAB_FILES = (
    "CONCEPT.csv",
    "CONCEPT_RELATIONSHIP.csv",
    "CONCEPT_SYNONYM.csv",
    "VOCABULARY.csv",
    "DOMAIN.csv",
)
CPT4_JAR = "cpt4.jar"
CPT4_FILES = (CPT4_JAR, "cpt.sh", "cpt.bat")

_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")
_LOGIN_HINT = (
    "The link may need you to be logged in to Athena, or it may have expired. "
    f"Request the bundle again at {ATHENA_URL} and use the new link, or download "
    "the zip in your browser and pass its path instead."
)


class VocabBundleError(Exception):
    """A vocabulary bundle could not be downloaded, unpacked or prepared."""


class BundleDownloadError(VocabBundleError):
    """The bundle could not be downloaded, or the download is not a zip file."""


class VocabFilesExistError(VocabBundleError):
    """Unpacking would overwrite files that are already in the target directory."""

    def __init__(self, directory: Path, conflicts: list[str]) -> None:
        self.directory = directory
        self.conflicts = conflicts
        shown = ", ".join(conflicts[:5]) + (" …" if len(conflicts) > 5 else "")
        super().__init__(
            f"{len(conflicts)} file(s) already exist in `{directory}`: {shown}"
        )


class Cpt4Error(VocabBundleError):
    """The CPT4 reconstitution step could not run or failed."""


@dataclass(frozen=True)
class BundleVocabulary:
    """A vocabulary listed in a bundle's `VOCABULARY.csv`."""

    vocabulary_id: str
    name: str
    version: str


def is_url(value: str) -> bool:
    """Whether `value` is an http(s) URL rather than a local path."""
    return urlparse(value).scheme.lower() in ("http", "https")


def bundle_filename(url: str) -> str:
    """A file name for the zip downloaded from `url`.

    Athena links end in the bundle id (`.../vocabularies/zip/<id>`), so `.zip` is
    added when the last part of the path has no such extension.
    """
    name = PurePosixPath(urlparse(url).path).name or "vocabulary_bundle"
    return name if name.lower().endswith(".zip") else f"{name}.zip"


ProgressCallback = Callable[[int, Optional[int]], None]


def download_bundle(
    url: str,
    dest_dir: Path,
    *,
    retries: int = 5,
    timeout: float = 60.0,
    backoff: float = 1.0,
    chunk_size: int = 1 << 20,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download a vocabulary bundle to `dest_dir` and return the zip's path.

    The file is written to `<name>.part` and renamed once complete, so an
    interrupted download is resumed (with an HTTP Range request) by the next
    attempt or the next run. Network errors and server errors (5xx) are retried
    with exponential backoff.

    Args:
        url: The bundle link Athena emailed.
        dest_dir: Directory to keep the zip in. Created if missing.
        retries: How many times to retry after a network or server error.
        timeout: Socket timeout in seconds.
        backoff: Seconds to wait before the first retry; doubled each time.
        chunk_size: Bytes read per iteration.
        progress: Called with `(bytes_done, total_bytes_or_None)` as data
            arrives.

    Returns:
        Path of the downloaded zip.

    Raises:
        BundleDownloadError: If the link needs a login, has expired, keeps
            failing, or does not return a zip file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / bundle_filename(url)
    if dest.exists() and zipfile.is_zipfile(dest):
        logger.info("Using the bundle downloaded earlier: %s", dest)
        return dest
    part = dest.with_name(dest.name + ".part")

    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        try:
            _download_once(url, part, offset, timeout, chunk_size, progress)
            break
        except BundleDownloadError:
            raise
        except HTTPError as e:
            if e.code == 416 and offset:
                break  # The part file already holds the whole bundle.
            if e.code in (401, 403):
                raise BundleDownloadError(
                    f"Athena refused the download (HTTP {e.code}). {_LOGIN_HINT}"
                ) from e
            if e.code in (404, 410):
                raise BundleDownloadError(
                    f"The bundle was not found (HTTP {e.code}). {_LOGIN_HINT}"
                ) from e
            if e.code < 500 or attempt == retries:
                raise BundleDownloadError(
                    f"Downloading the bundle failed (HTTP {e.code} {e.reason})."
                ) from e
            error: Exception = e
        except (URLError, HTTPException, OSError) as e:
            if attempt == retries:
                raise BundleDownloadError(
                    f"Downloading the bundle failed after {retries + 1} attempts: {e}"
                ) from e
            error = e
        wait = backoff * 2**attempt
        logger.warning(
            "Download interrupted (%s); retrying in %.0f s (%d/%d)",
            error,
            wait,
            attempt + 1,
            retries,
        )
        time.sleep(wait)

    _check_zip(part)
    part.replace(dest)
    return dest


def _download_once(
    url: str,
    part: Path,
    offset: int,
    timeout: float,
    chunk_size: int,
    progress: Optional[ProgressCallback],
) -> None:
    headers = {"User-Agent": "aatm"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            raise BundleDownloadError(
                f"The link returned a web page instead of a zip file. {_LOGIN_HINT}"
            )
        if offset and response.status != 206:
            offset = 0  # The server ignored the Range header; start over.
        length = response.headers.get("Content-Length")
        total = offset + int(length) if length is not None else None
        done = offset
        if progress:
            progress(done, total)
        with part.open("ab" if offset else "wb") as f:
            while chunk := response.read(chunk_size):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if total is not None and done < total:
        raise HTTPException(f"connection closed after {done} of {total} bytes")


def _check_zip(path: Path) -> None:
    with path.open("rb") as f:
        head = f.read(512)
    if head[:4] in _ZIP_MAGIC and zipfile.is_zipfile(path):
        return
    path.unlink(missing_ok=True)
    if b"<html" in head.lower() or b"<!doctype" in head.lower():
        raise BundleDownloadError(
            f"The link returned a web page instead of a zip file. {_LOGIN_HINT}"
        )
    raise BundleDownloadError("The download is not a zip file. " + _LOGIN_HINT)


def unpack_bundle(
    zip_path: Path, vocab_dir: Path, *, overwrite: bool = False
) -> list[str]:
    """Extract the bundle's files into `vocab_dir` and return their names.

    Files are placed directly in `vocab_dir`, also when the zip keeps them in a
    folder, because `aatm init` reads the CSVs from the top of the directory.

    Args:
        zip_path: Path of the bundle zip.
        vocab_dir: Directory to extract into. Created if missing.
        overwrite: Replace files that already exist in `vocab_dir`.

    Returns:
        Names of the extracted files.

    Raises:
        VocabBundleError: If `zip_path` is missing or not a zip file.
        VocabFilesExistError: If files would be overwritten and `overwrite` is
            false. Nothing is extracted in that case.
    """
    if not zip_path.is_file():
        raise VocabBundleError(f"The bundle `{zip_path}` does not exist.")
    if not zipfile.is_zipfile(zip_path):
        raise VocabBundleError(f"`{zip_path}` is not a zip file.")

    with zipfile.ZipFile(zip_path) as archive:
        members = {}
        for info in archive.infolist():
            name = PurePosixPath(info.filename.replace("\\", "/")).name
            if info.is_dir() or not name or name.startswith("."):
                continue
            members[name] = info

        conflicts = sorted(n for n in members if (vocab_dir / n).exists())
        if conflicts and not overwrite:
            raise VocabFilesExistError(vocab_dir, conflicts)

        vocab_dir.mkdir(parents=True, exist_ok=True)
        for name, info in members.items():
            with archive.open(info) as src, (vocab_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    return sorted(members)


def missing_required_files(vocab_dir: Path) -> list[str]:
    """The required vocabulary CSVs that are not in `vocab_dir`."""
    return [n for n in REQUIRED_VOCAB_FILES if not (vocab_dir / n).is_file()]


def read_bundle_vocabularies(vocab_dir: Path) -> list[BundleVocabulary]:
    """The vocabularies listed in `VOCABULARY.csv`, with their versions."""
    path = vocab_dir / "VOCABULARY.csv"
    if not path.is_file():
        return []
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [
            BundleVocabulary(
                vocabulary_id=row.get("vocabulary_id") or "",
                name=row.get("vocabulary_name") or "",
                version=row.get("vocabulary_version") or "",
            )
            for row in reader
            if row.get("vocabulary_id")
        ]


def has_cpt4_tooling(vocab_dir: Path) -> bool:
    """Whether the bundle includes the CPT4 reconstitution tool."""
    return (vocab_dir / CPT4_JAR).is_file()


def run_cpt4(
    vocab_dir: Path,
    umls_api_key: str,
    *,
    java: Optional[str] = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Add the CPT4 concept names to the bundle with the user's UMLS API key.

    Runs `java -Dumls-apikey=<key> -jar cpt4.jar 5` in `vocab_dir`, as the
    bundle's `readme.txt` describes. The key is passed only to Java and kept out
    of logs and error messages.

    Args:
        vocab_dir: The unpacked bundle.
        umls_api_key: The user's UMLS API key, from
            https://uts.nlm.nih.gov/uts/profile.
        java: Path to the Java executable. Found on `PATH` when not given.
        runner: Runs the command; replaceable in tests.

    Raises:
        Cpt4Error: If the tool is missing, Java is not installed, or the tool
            fails.
    """
    if not has_cpt4_tooling(vocab_dir):
        raise Cpt4Error(f"`{CPT4_JAR}` is not in `{vocab_dir}`.")
    java = java or shutil.which("java")
    if java is None:
        raise Cpt4Error(
            "Adding the CPT4 names needs Java, which was not found on PATH. "
            "Install a Java runtime (11 or newer, e.g. from https://adoptium.net) "
            "and run `aatm init` again."
        )
    command = [java, f"-Dumls-apikey={umls_api_key}", "-jar", CPT4_JAR, "5"]
    try:
        result = runner(command, cwd=vocab_dir, capture_output=True, text=True)
    except OSError as e:
        raise Cpt4Error(f"Could not start Java (`{java}`): {e}") from e
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").replace(umls_api_key, "***")
        tail = "\n".join(output.strip().splitlines()[-10:])
        raise Cpt4Error(
            f"The CPT4 tool exited with code {result.returncode}. Check that your "
            f"UMLS API key is valid.\n{tail}"
        )
