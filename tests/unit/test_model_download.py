"""Downloading model weights once: configs/models.yaml and pushlab.models.download.

A fake downloader stands in for Hugging Face, so nothing is downloaded. The
tests that run the real huggingface_hub library are in
test_model_download_hub.py.
"""

import doctest
import json
from pathlib import Path

import pytest

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
PINNED = REGISTRY["qwen3.5-4b"].revision  # the commit Qwen3.5-4B is pinned to


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
    assert record == {"hf_id": "Qwen/Qwen3.5-4B", "revision": PINNED, "commit": COMMIT}
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
        {"hf_id": "Qwen/Qwen3-4B", "revision": PINNED},
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
