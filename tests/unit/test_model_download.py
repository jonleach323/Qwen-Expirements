"""Downloading model weights once: configs/models.yaml and pushlab.models.download.

A fake downloader and a fake Hugging Face client stand in for the real ones,
so nothing is downloaded.
"""

import doctest
import json
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError

from pushlab.models import download
from pushlab.models.download import (
    MARKER,
    ensure_models,
    is_downloaded,
    main,
    parse_model_names,
)
from pushlab.models.registry import REGISTRY, ModelSpec

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


class FakeHub:
    """Stands in for huggingface_hub: records each request, downloads nothing."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object]] = []

    def model_info(self, repo_id: str, revision: str) -> SimpleNamespace:
        self.requests.append(("model_info", repo_id, revision))
        return SimpleNamespace(sha=COMMIT)

    def snapshot_download(self, repo_id: str, *, revision: str, local_dir: Path) -> str:
        self.requests.append(("snapshot_download", repo_id, (revision, local_dir)))
        return str(local_dir)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub()
    monkeypatch.setattr(download, "HfApi", lambda: fake)
    monkeypatch.setattr(download, "snapshot_download", fake.snapshot_download)
    return fake


class FakeDownloader:
    """Records each call and writes one file, like a real download would."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, spec: ModelSpec, folder: Path) -> str:
        self.calls.append((spec.name, folder))
        (folder / "config.json").write_text("{}")
        return COMMIT


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


def test_hub_download_fetches_every_file_from_one_commit(
    tmp_path: Path, hub: FakeHub
) -> None:
    commit = download.hub_download(REGISTRY["qwen3.5-4b"], tmp_path)
    assert commit == COMMIT
    assert hub.requests == [
        ("model_info", "Qwen/Qwen3.5-4B", "main"),
        ("snapshot_download", "Qwen/Qwen3.5-4B", (COMMIT, tmp_path)),
    ]


def test_main_downloads_the_listed_models(tmp_path: Path, hub: FakeHub) -> None:
    config = tmp_path / "models.yaml"
    config.write_text("models: [qwen3.5-4b]\n")
    assert main([str(config), str(tmp_path / "models")]) == 0
    folder = tmp_path / "models" / "qwen3.5-4b"
    assert hub.requests[-1] == (
        "snapshot_download",
        "Qwen/Qwen3.5-4B",
        (COMMIT, folder),
    )
    assert is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_main_explains_a_bad_config_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "models.yaml"
    config.write_text("models: [qwen9-1t]\n")
    assert main([str(config), str(tmp_path / "models")]) == 1
    assert "qwen9-1t" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.yaml"), str(tmp_path)]) == 1


def test_main_explains_a_refused_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = "https://huggingface.co/api/models/Qwen/Qwen3.5-4B/revision/main"
    response = httpx2.Response(401, request=httpx2.Request("GET", url))

    class RefusingApi:
        def model_info(self, repo_id: str, revision: str) -> None:
            raise HfHubHTTPError("401 Client Error: Unauthorized", response=response)

    monkeypatch.setattr(download, "HfApi", RefusingApi)
    config = tmp_path / "models.yaml"
    config.write_text("models: [qwen3.5-4b]\n")
    assert main([str(config), str(tmp_path / "models")]) == 1
    err = capsys.readouterr().err
    assert "401" in err
    assert "HF_TOKEN" in err
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], tmp_path / "models" / "qwen3.5-4b")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(download)
    assert result.attempted > 0
    assert result.failed == 0
