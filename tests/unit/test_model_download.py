"""Downloading model weights once: configs/models.yaml and pushlab.models.download.

Two stand-ins replace Hugging Face, so nothing is downloaded: a fake
downloader for the bookkeeping, and a fake Hub server that answers the real
huggingface_hub library's requests inside this process.
"""

import doctest
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest
from huggingface_hub import set_client_factory
from huggingface_hub.utils._http import default_client_factory

from pushlab.models import download
from pushlab.models.download import (
    MARKER,
    ensure_models,
    is_downloaded,
    main,
    parse_model_names,
    remove_unfinished_files,
)
from pushlab.models.registry import REGISTRY, ModelSpec

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "0123456789abcdef0123456789abcdef01234567"
REPO = "Qwen/Qwen3.5-4B"
FILES = {
    "config.json": b'{"model_type": "qwen3_5"}',
    "model-00001-of-00002.safetensors": b"first half of the weights",
    "model-00002-of-00002.safetensors": b"second half of the weights",
}
FAKE_TOKEN = "hf_fake_token_for_tests"


class FakeDownloader:
    """Records each call and writes one file, like a real download would."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, spec: ModelSpec, folder: Path) -> str:
        self.calls.append((spec.name, folder))
        (folder / "config.json").write_text("{}")
        return COMMIT


class FakeHubServer:
    """Answers huggingface_hub's requests for Qwen3.5-4B, inside this process."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.model_info_status = 200  # set 401 to refuse the token
        self.listing_status = 200  # set 400 to make listing the files fail

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        self.requests.append(f"{request.method} {path}")
        assert request.headers.get("authorization") == f"Bearer {FAKE_TOKEN}"
        if path == f"/api/models/{REPO}/revision/main":
            if self.model_info_status != 200:
                return self._error(self.model_info_status, request)
            return httpx2.Response(200, json={"id": REPO, "sha": COMMIT})
        if path == f"/api/models/{REPO}/tree/{COMMIT}":
            if self.listing_status != 200:
                return self._error(self.listing_status, request)
            entries = [
                {"type": "file", "path": name, "size": len(data), "oid": _oid(data)}
                for name, data in FILES.items()
            ]
            return httpx2.Response(200, json=entries)
        name = path.removeprefix(f"/{REPO}/resolve/{COMMIT}/")
        if name in FILES:
            data = FILES[name]
            headers = {
                "X-Repo-Commit": COMMIT,
                "ETag": f'"{_oid(data)}"',
                "Content-Length": str(len(data)),
            }
            body = b"" if request.method == "HEAD" else data
            return httpx2.Response(200, headers=headers, content=body)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    @staticmethod
    def _error(status: int, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": "refused"}, request=request)


def _oid(data: bytes) -> str:
    """The git blob ID that Hugging Face reports for a small file."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHubServer]:
    server = FakeHubServer()
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    transport = httpx2.MockTransport(server.handle)
    set_client_factory(lambda: httpx2.Client(transport=transport))
    yield server
    set_client_factory(default_client_factory)


def _config(tmp_path: Path) -> str:
    config = tmp_path / "models.yaml"
    config.write_text("models: [qwen3.5-4b]\n")
    return str(config)


def test_the_pod_downloads_qwen3_5_4b() -> None:
    text = (ROOT / "configs" / "models.yaml").read_text()
    assert parse_model_names(text) == ["qwen3.5-4b"]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("models: [qwen9-1t]", "not in the model registry"),
        ("models: [tiny]", "built by the tests"),
        ("models: qwen3.5-4b", "expected a list"),
        ("models: []", "expected a list"),
        ("models: [{name: qwen3.5-4b}]", "not in the model registry"),
        ("- qwen3.5-4b", "expected a list"),
        ("", "expected a list"),
        ("models: [qwen3.5-4b", "not valid YAML"),
    ],
)
def test_bad_model_lists_are_refused(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_model_names(text)


def test_a_name_listed_twice_is_downloaded_once() -> None:
    assert parse_model_names("models: [qwen3-8b, qwen3-8b]") == ["qwen3-8b"]


def test_downloads_a_missing_model_and_records_it(tmp_path: Path) -> None:
    fake = FakeDownloader()
    folders = ensure_models(["qwen3.5-4b"], tmp_path, download=fake)
    folder = tmp_path / "qwen3.5-4b"
    assert folders == [folder]
    assert fake.calls == [("qwen3.5-4b", folder)]
    record = json.loads((folder / MARKER).read_text())
    assert record == {"hf_id": "Qwen/Qwen3.5-4B", "revision": "main", "commit": COMMIT}
    assert is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_a_finished_model_is_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ensure_models(["qwen3.5-4b"], tmp_path, download=FakeDownloader())
    fake = FakeDownloader()
    ensure_models(["qwen3.5-4b"], tmp_path, download=fake)
    assert fake.calls == []
    assert "already downloaded" in capsys.readouterr().out


def test_an_interrupted_download_is_retried(tmp_path: Path) -> None:
    def interrupted(spec: ModelSpec, folder: Path) -> str:
        (folder / "model-00001-of-00002.safetensors").write_text("half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ensure_models(["qwen3.5-4b"], tmp_path, download=interrupted)
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], tmp_path / "qwen3.5-4b")
    fake = FakeDownloader()
    ensure_models(["qwen3.5-4b"], tmp_path, download=fake)
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "marker",
    [
        {"hf_id": "Qwen/Qwen3.5-4B", "revision": "an-older-revision"},
        {"hf_id": "Qwen/Qwen3-4B", "revision": "main"},
        ["not", "a", "record"],
    ],
)
def test_a_different_or_broken_record_downloads_again(
    tmp_path: Path, marker: object
) -> None:
    folder = tmp_path / "qwen3.5-4b"
    folder.mkdir()
    (folder / MARKER).write_text(json.dumps(marker))
    fake = FakeDownloader()
    ensure_models(["qwen3.5-4b"], tmp_path, download=fake)
    assert len(fake.calls) == 1


def test_downloads_every_file_from_one_commit(
    tmp_path: Path, hub: FakeHubServer
) -> None:
    models = tmp_path / "models"
    folder = models / "qwen3.5-4b"
    leftover = folder / ".cache" / "huggingface" / "download" / "a.b.c.incomplete"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("part of a file from a killed download")
    assert main([_config(tmp_path), str(models)]) == 0
    assert not leftover.exists()
    for name, data in FILES.items():
        assert (folder / name).read_bytes() == data
    assert hub.requests[:2] == [
        f"GET /api/models/{REPO}/revision/main",
        f"GET /api/models/{REPO}/tree/{COMMIT}",
    ]
    assert all(f"/{COMMIT}" in request for request in hub.requests[1:])
    record = json.loads((folder / MARKER).read_text())
    assert record["commit"] == COMMIT


def test_a_finished_download_needs_no_requests(
    tmp_path: Path, hub: FakeHubServer
) -> None:
    models = tmp_path / "models"
    main([_config(tmp_path), str(models)])
    hub.requests.clear()
    assert main([_config(tmp_path), str(models)]) == 0
    assert hub.requests == []


def test_a_half_full_folder_is_not_recorded_when_listing_fails(
    tmp_path: Path, hub: FakeHubServer, capsys: pytest.CaptureFixture[str]
) -> None:
    # An earlier run was interrupted after one file. Now the Hub cannot list
    # the commit's files, and snapshot_download hands the folder back as is.
    folder = tmp_path / "models" / "qwen3.5-4b"
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(FILES["config.json"])
    hub.listing_status = 400
    assert main([_config(tmp_path), str(tmp_path / "models")]) == 1
    assert "did not list the files" in capsys.readouterr().err
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_a_file_missing_after_the_download_is_not_recorded(
    tmp_path: Path,
    hub: FakeHubServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Hugging Face already listed this commit, but a file went missing and
    # the next snapshot_download returned without fetching it.
    models = tmp_path / "models"
    folder = models / "qwen3.5-4b"
    main([_config(tmp_path), str(models)])
    (folder / MARKER).unlink()
    (folder / "model-00002-of-00002.safetensors").unlink()
    monkeypatch.setattr(download, "snapshot_download", lambda *args, **kwargs: "")
    assert main([_config(tmp_path), str(models)]) == 1
    assert "model-00002-of-00002.safetensors" in capsys.readouterr().err
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_main_explains_a_refused_token(
    tmp_path: Path, hub: FakeHubServer, capsys: pytest.CaptureFixture[str]
) -> None:
    hub.model_info_status = 401
    assert main([_config(tmp_path), str(tmp_path / "models")]) == 1
    err = capsys.readouterr().err
    assert "401" in err
    assert "HF_TOKEN" in err
    assert FAKE_TOKEN not in err
    folder = tmp_path / "models" / "qwen3.5-4b"
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_main_explains_a_bad_config_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "models.yaml"
    config.write_text("models: [qwen9-1t]\n")
    assert main([str(config), str(tmp_path / "models")]) == 1
    assert "qwen9-1t" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.yaml"), str(tmp_path)]) == 1


def test_removes_the_partial_files_a_killed_download_left(tmp_path: Path) -> None:
    downloads = tmp_path / ".cache" / "huggingface" / "download"
    downloads.mkdir(parents=True)
    partial = downloads / "abc.0123.9f8e7d6c.incomplete"
    partial.write_text("half a file")
    finished = tmp_path / "config.json"
    finished.write_text("{}")
    assert remove_unfinished_files(tmp_path) == 1
    assert not partial.exists()
    assert finished.exists()


def test_docstring_examples_run() -> None:
    result = doctest.testmod(download)
    assert result.attempted > 0
    assert result.failed == 0
