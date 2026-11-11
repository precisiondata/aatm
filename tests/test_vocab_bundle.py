import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest
import typer

from aatm import main as cli
from aatm import vocab_bundle as vb

VOCABULARY_CSV = (
    "vocabulary_id\tvocabulary_name\tvocabulary_reference\tvocabulary_version\tvocabulary_concept_id\n"
    "LOINC\tLogical Observation Identifiers\thttp://loinc.org\tLOINC 2.80\t44819102\n"
    "SNOMED\tSystematic Nomenclature of Medicine\thttp://snomed.org\tSnomed v20250301\t44819097\n"
)


def make_bundle(path: Path, *, folder: str = "", cpt4: bool = False, skip=()) -> Path:
    files = {name: "header\n" for name in vb.REQUIRED_VOCAB_FILES}
    files["VOCABULARY.csv"] = VOCABULARY_CSV
    files["readme.txt"] = "Athena bundle"
    if cpt4:
        files.update({"cpt4.jar": "jar", "cpt.sh": "sh", "cpt.bat": "bat"})
    with zipfile.ZipFile(path, "w") as z:
        for name, content in files.items():
            if name not in skip:
                z.writestr(folder + name, content)
    return path


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return make_bundle(tmp_path / "bundle.zip")


# --- unpacking and validation ---------------------------------------------------


def test_unpack_extracts_files_and_reads_vocabularies(bundle: Path, tmp_path: Path):
    target = tmp_path / "vocabularies"
    names = vb.unpack_bundle(bundle, target)

    assert set(vb.REQUIRED_VOCAB_FILES) <= set(names)
    assert vb.missing_required_files(target) == []
    assert vb.read_bundle_vocabularies(target) == [
        vb.BundleVocabulary("LOINC", "Logical Observation Identifiers", "LOINC 2.80"),
        vb.BundleVocabulary(
            "SNOMED", "Systematic Nomenclature of Medicine", "Snomed v20250301"
        ),
    ]


def test_unpack_flattens_a_folder_inside_the_zip(tmp_path: Path):
    bundle = make_bundle(tmp_path / "b.zip", folder="vocabulary_download_v5/")
    target = tmp_path / "v"
    vb.unpack_bundle(bundle, target)
    assert (target / "CONCEPT.csv").is_file()


def test_unpack_ignores_paths_that_escape_the_target(tmp_path: Path):
    bundle = tmp_path / "evil.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        z.writestr("../../escaped.csv", "x")
    target = tmp_path / "a" / "b"
    vb.unpack_bundle(bundle, target)
    assert (target / "escaped.csv").is_file()
    assert not (tmp_path / "escaped.csv").exists()


def test_unpack_refuses_to_overwrite_without_permission(bundle: Path, tmp_path: Path):
    target = tmp_path / "v"
    target.mkdir()
    (target / "CONCEPT.csv").write_text("mine")

    with pytest.raises(vb.VocabFilesExistError) as e:
        vb.unpack_bundle(bundle, target)

    assert e.value.conflicts == ["CONCEPT.csv"]
    assert (target / "CONCEPT.csv").read_text() == "mine"
    assert not (target / "DOMAIN.csv").exists()  # nothing extracted

    vb.unpack_bundle(bundle, target, overwrite=True)
    assert (target / "CONCEPT.csv").read_text() == "header\n"


def test_missing_required_files_are_reported(tmp_path: Path):
    bundle = make_bundle(tmp_path / "b.zip", skip={"CONCEPT_SYNONYM.csv"})
    target = tmp_path / "v"
    vb.unpack_bundle(bundle, target)
    assert vb.missing_required_files(target) == ["CONCEPT_SYNONYM.csv"]


def test_unpack_rejects_a_file_that_is_not_a_zip(tmp_path: Path):
    fake = tmp_path / "fake.zip"
    fake.write_text("<html>login</html>")
    with pytest.raises(vb.VocabBundleError, match="not a zip"):
        vb.unpack_bundle(fake, tmp_path / "v")
    with pytest.raises(vb.VocabBundleError, match="does not exist"):
        vb.unpack_bundle(tmp_path / "missing.zip", tmp_path / "v")


@pytest.mark.parametrize(
    "url, name",
    [
        ("https://athena.ohdsi.org/api/v1/vocabularies/zip/abc-123", "abc-123.zip"),
        ("https://example.org/files/bundle.zip?token=x", "bundle.zip"),
        ("https://example.org/", "vocabulary_bundle.zip"),
    ],
)
def test_bundle_filename(url: str, name: str):
    assert vb.bundle_filename(url) == name


def test_is_url():
    assert vb.is_url("https://athena.ohdsi.org/x")
    assert not vb.is_url("C:\\Users\\me\\bundle.zip")
    assert not vb.is_url("~/Downloads/bundle.zip")


# --- CPT4 ----------------------------------------------------------------------


def unpacked_with_cpt4(tmp_path: Path) -> Path:
    target = tmp_path / "v"
    vb.unpack_bundle(make_bundle(tmp_path / "b.zip", cpt4=True), target)
    return target


def test_run_cpt4_calls_java_in_the_bundle_directory(tmp_path: Path):
    target = unpacked_with_cpt4(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    vb.run_cpt4(target, "secret-key", java="java", runner=runner)

    command, kwargs = calls[0]
    assert command == ["java", "-Dumls-apikey=secret-key", "-jar", "cpt4.jar", "5"]
    assert kwargs["cwd"] == target


def test_run_cpt4_without_java_explains_what_is_missing(tmp_path, monkeypatch):
    target = unpacked_with_cpt4(tmp_path)
    monkeypatch.setattr(vb.shutil, "which", lambda name: None)
    with pytest.raises(vb.Cpt4Error, match="needs Java"):
        vb.run_cpt4(target, "key")


def test_run_cpt4_failure_hides_the_key(tmp_path: Path):
    target = unpacked_with_cpt4(tmp_path)

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "bad key secret-key")

    with pytest.raises(vb.Cpt4Error) as e:
        vb.run_cpt4(target, "secret-key", java="java", runner=runner)
    assert "secret-key" not in str(e.value)
    assert "***" in str(e.value)


# --- downloads -----------------------------------------------------------------


class Server:
    """A local HTTP server whose behaviour each test sets."""

    def __init__(self) -> None:
        self.body = b""
        self.status = 200
        self.content_type = "application/zip"
        self.truncate_first = False  # cut the first response short
        self.requests: list[dict[str, str]] = []

    def url(self, path: str = "/api/v1/vocabularies/zip/abc") -> str:
        return f"http://127.0.0.1:{self.port}{path}"


@pytest.fixture
def server() -> Iterator[Server]:
    state = Server()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.requests.append(dict(self.headers))
            if state.status != 200:
                self.send_error(state.status)
                return
            body, status = state.body, 200
            start = 0
            if rng := self.headers.get("Range"):
                start = int(rng.removeprefix("bytes=").split("-")[0])
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", state.content_type)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            payload = body[start:]
            if state.truncate_first:
                state.truncate_first = False
                payload = payload[: len(payload) // 2]
                self.wfile.write(payload)
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.port = httpd.server_address[1]  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield state
    httpd.shutdown()
    httpd.server_close()


def test_download_saves_the_zip(server: Server, bundle: Path, tmp_path: Path):
    server.body = bundle.read_bytes()
    seen = []

    path = vb.download_bundle(
        server.url(), tmp_path / "dl", progress=lambda d, t: seen.append((d, t))
    )

    assert path == tmp_path / "dl" / "abc.zip"
    assert path.read_bytes() == server.body
    assert seen[-1] == (len(server.body), len(server.body))
    assert not (tmp_path / "dl" / "abc.zip.part").exists()


def test_download_resumes_after_the_connection_drops(
    server: Server, bundle: Path, tmp_path: Path
):
    server.body = bundle.read_bytes()
    server.truncate_first = True

    path = vb.download_bundle(server.url(), tmp_path, backoff=0)

    assert path.read_bytes() == server.body
    assert len(server.requests) == 2
    assert server.requests[1]["Range"] == f"bytes={len(server.body) // 2}-"


def test_download_reuses_a_finished_download(server, bundle, tmp_path):
    server.body = bundle.read_bytes()
    vb.download_bundle(server.url(), tmp_path)
    vb.download_bundle(server.url(), tmp_path)
    assert len(server.requests) == 1


def test_download_of_a_login_page_is_rejected(server: Server, tmp_path: Path):
    server.body = b"<!DOCTYPE html><html>Please log in</html>"
    server.content_type = "text/html; charset=utf-8"
    with pytest.raises(vb.BundleDownloadError, match="logged in"):
        vb.download_bundle(server.url(), tmp_path)


def test_download_that_is_not_a_zip_is_rejected(server: Server, tmp_path: Path):
    server.body = b"definitely not a zip"
    server.content_type = "application/octet-stream"
    with pytest.raises(vb.BundleDownloadError, match="not a zip"):
        vb.download_bundle(server.url(), tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status, message", [(403, "refused"), (404, "not found")])
def test_expired_or_protected_links_give_a_clear_error(
    server: Server, tmp_path: Path, status: int, message: str
):
    server.status = status
    with pytest.raises(vb.BundleDownloadError, match=message) as e:
        vb.download_bundle(server.url(), tmp_path)
    assert "expired" in str(e.value)
    assert len(server.requests) == 1  # not retried


def test_server_errors_are_retried_then_reported(server: Server, tmp_path: Path):
    server.status = 503
    with pytest.raises(vb.BundleDownloadError, match="503"):
        vb.download_bundle(server.url(), tmp_path, retries=2, backoff=0)
    assert len(server.requests) == 3


# --- CLI -----------------------------------------------------------------------


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_prepare_from_a_url(server: Server, bundle: Path, in_tmp: Path, capsys):
    server.body = bundle.read_bytes()

    vocab_dir = cli.prepare_vocab_bundle(
        server.url(), Path("vocabularies"), standard_vocabs=["LOINC", "RxNorm"]
    )

    assert vocab_dir == Path("vocabularies")
    assert (in_tmp / "vocabularies" / "CONCEPT.csv").is_file()
    assert (in_tmp / ".aatm" / "downloads" / "abc.zip").is_file()
    out = capsys.readouterr().out
    assert "LOINC: LOINC 2.80" in out
    assert "RxNorm was requested" in out


def test_prepare_from_a_path_refuses_to_overwrite_without_force(
    bundle: Path, in_tmp: Path, monkeypatch
):
    (in_tmp / "vocabularies").mkdir()
    (in_tmp / "vocabularies" / "CONCEPT.csv").write_text("mine")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)

    with pytest.raises(typer.Exit):
        cli.prepare_vocab_bundle(str(bundle), Path("vocabularies"))
    assert (in_tmp / "vocabularies" / "CONCEPT.csv").read_text() == "mine"

    cli.prepare_vocab_bundle(str(bundle), Path("vocabularies"), force=True)
    assert (in_tmp / "vocabularies" / "CONCEPT.csv").read_text() == "header\n"


def test_prepare_stops_when_required_files_are_missing(in_tmp: Path, capsys):
    bundle = make_bundle(in_tmp / "b.zip", skip={"CONCEPT.csv"})
    with pytest.raises(typer.Exit):
        cli.prepare_vocab_bundle(str(bundle), Path("v"))
    assert "CONCEPT.csv" in capsys.readouterr().out


def test_prepare_warns_and_continues_without_a_umls_key(in_tmp: Path, capsys):
    bundle = make_bundle(in_tmp / "b.zip", cpt4=True)
    cli.prepare_vocab_bundle(str(bundle), Path("v"), umls_api_key=None)
    assert "no UMLS" in capsys.readouterr().out


def test_prepare_runs_cpt4_with_the_key(in_tmp: Path, monkeypatch):
    bundle = make_bundle(in_tmp / "b.zip", cpt4=True)
    calls = []
    monkeypatch.setattr(vb, "run_cpt4", lambda d, key: calls.append((d, key)))

    cli.prepare_vocab_bundle(str(bundle), Path("v"), umls_api_key="k")

    assert calls == [(Path("v"), "k")]
