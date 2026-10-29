"""pushlab.models.download with the real huggingface_hub library.

A fake Hub server answers the library's requests inside this process, so
nothing is downloaded and no request leaves the machine.
"""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest
from huggingface_hub import constants, set_client_factory
from huggingface_hub.utils._http import default_client_factory

from pushlab.models.download import MARKER, is_downloaded, main
from pushlab.models.registry import REGISTRY

COMMIT = "0123456789abcdef0123456789abcdef01234567"
NEW_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
REPO = "Qwen/Qwen3.5-4B"
FILES = {
    "config.json": b'{"model_type": "qwen3_5"}',
    "model-00001-of-00002.safetensors": b"first half of the weights",
    "model-00002-of-00002.safetensors": b"second half of the weights",
}
# A later commit that changes only config.json.
NEW_FILES = {**FILES, "config.json": b'{"model_type": "qwen3_5", "fixed": true}'}
FAKE_TOKEN = "hf_fake_token_for_tests"


class FakeHubServer:
    """Answers huggingface_hub's requests for Qwen3.5-4B, inside this process."""

    def __init__(self) -> None:
        self.commits = {COMMIT: FILES, NEW_COMMIT: NEW_FILES}
        self.main = COMMIT  # the commit that the revision "main" points to
        self.requests: list[str] = []
        self.model_info_status = 200  # set 401 to refuse the token
        self.listing_failures = 0  # how many requests to list files fail
        self.unreachable: set[str] = set()  # files whose update check fails

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        self.requests.append(f"{request.method} {path}")
        assert request.headers.get("authorization") == f"Bearer {FAKE_TOKEN}"
        if path == f"/api/models/{REPO}/revision/main":
            if self.model_info_status != 200:
                return self._error(self.model_info_status, request)
            return httpx2.Response(200, json={"id": REPO, "sha": self.main})
        for commit, files in self.commits.items():
            if path == f"/api/models/{REPO}/tree/{commit}":
                if self.listing_failures:
                    self.listing_failures -= 1
                    return self._error(400, request)
                entries = [
                    {"type": "file", "path": name, "size": len(data), "oid": _oid(data)}
                    for name, data in files.items()
                ]
                return httpx2.Response(200, json=entries)
            prefix = f"/{REPO}/resolve/{commit}/"
            if path.startswith(prefix) and path[len(prefix) :] in files:
                name = path[len(prefix) :]
                if request.method == "HEAD" and name in self.unreachable:
                    raise httpx2.ConnectError("the Hub could not be reached")
                data = files[name]
                headers = {
                    "X-Repo-Commit": commit,
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
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHubServer]:
    # huggingface_hub 2.x sends every request through an httpx2 client. Its
    # default client factory is private, but it is the only way to put the
    # library back as it was. Its shared cache moves to the test's folder.
    server = FakeHubServer()
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    transport = httpx2.MockTransport(server.handle)
    set_client_factory(lambda: httpx2.Client(transport=transport))
    yield server
    set_client_factory(default_client_factory)


def _config(tmp_path: Path) -> str:
    config = tmp_path / "models.yaml"
    config.write_text("models: [qwen3.5-4b]\n")
    return str(config)


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
    # Without the record, Hugging Face's own notes still spare every file.
    (models / "qwen3.5-4b" / MARKER).unlink()
    assert main([_config(tmp_path), str(models)]) == 0
    assert not [request for request in hub.requests if request.startswith("GET /Qwen")]


@pytest.mark.parametrize(
    ("listing_failures", "message"),
    [(1, "missing or not from commit"), (2, "probably a connection problem")],
)
def test_a_half_full_folder_is_not_recorded_when_listing_fails(
    tmp_path: Path,
    hub: FakeHubServer,
    capsys: pytest.CaptureFixture[str],
    listing_failures: int,
    message: str,
) -> None:
    # An earlier run was interrupted after one file. Now the Hub cannot list
    # the commit's files, and snapshot_download hands the folder back as is.
    # The check after it may then get the list (1 failure) or not (2).
    folder = tmp_path / "models" / "qwen3.5-4b"
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(FILES["config.json"])
    hub.listing_failures = listing_failures
    assert main([_config(tmp_path), str(tmp_path / "models")]) == 1
    assert message in capsys.readouterr().err
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_a_file_that_cannot_be_checked_is_called_a_connection_problem(
    tmp_path: Path, hub: FakeHubServer, capsys: pytest.CaptureFixture[str]
) -> None:
    # config.json is there without Hugging Face's notes, and checking it
    # fails every time, which trips an assertion inside huggingface_hub.
    folder = tmp_path / "models" / "qwen3.5-4b"
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(FILES["config.json"])
    hub.unreachable = {"config.json"}
    assert main([_config(tmp_path), str(tmp_path / "models")]) == 1
    assert "probably a connection problem" in capsys.readouterr().err
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_a_copy_in_the_shared_cache_does_not_count(
    tmp_path: Path, hub: FakeHubServer
) -> None:
    # Loading the model by its Hugging Face ID filled the shared cache, but
    # the model folder is still half full.
    for name, data in FILES.items():
        cached = tmp_path / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots"
        (cached / COMMIT).mkdir(parents=True, exist_ok=True)
        (cached / COMMIT / name).write_bytes(data)
    folder = tmp_path / "models" / "qwen3.5-4b"
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(FILES["config.json"])
    hub.listing_failures = 1
    assert main([_config(tmp_path), str(tmp_path / "models")]) == 1
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_files_a_new_commit_dropped_are_removed(
    tmp_path: Path, hub: FakeHubServer, capsys: pytest.CaptureFixture[str]
) -> None:
    # The new commit merges the two weight files into one.
    merged = "a" * 40
    hub.commits[merged] = {
        "config.json": FILES["config.json"],
        "model.safetensors": b"all of the weights",
    }
    models = tmp_path / "models"
    folder = models / "qwen3.5-4b"
    assert main([_config(tmp_path), str(models)]) == 0
    (folder / MARKER).unlink()
    hub.main = merged
    assert main([_config(tmp_path), str(models)]) == 0
    weights = sorted(path.name for path in folder.glob("*.safetensors"))
    assert weights == ["model.safetensors"]
    assert "removed model-00001-of-00002.safetensors" in capsys.readouterr().out
    assert json.loads((folder / MARKER).read_text())["commit"] == merged


def test_your_own_files_in_the_model_folder_are_kept(
    tmp_path: Path, hub: FakeHubServer
) -> None:
    models = tmp_path / "models"
    folder = models / "qwen3.5-4b"
    assert main([_config(tmp_path), str(models)]) == 0
    (folder / "Untitled.ipynb").write_text("{}")
    (folder / "notes").mkdir()
    (folder / "notes" / "todo.md").write_text("try a larger dose")
    (folder / MARKER).unlink()
    hub.main = NEW_COMMIT
    assert main([_config(tmp_path), str(models)]) == 0
    assert (folder / "Untitled.ipynb").exists()
    assert (folder / "notes" / "todo.md").exists()


def test_a_commit_without_weights_is_refused(
    tmp_path: Path, hub: FakeHubServer, capsys: pytest.CaptureFixture[str]
) -> None:
    # A revision pinned by mistake to the repository's first commit.
    first = "c" * 40
    hub.commits[first] = {".gitattributes": b"*.safetensors filter=lfs"}
    models = tmp_path / "models"
    folder = models / "qwen3.5-4b"
    assert main([_config(tmp_path), str(models)]) == 0
    (folder / MARKER).unlink()
    hub.main = first
    assert main([_config(tmp_path), str(models)]) == 1
    assert "registry.py" in capsys.readouterr().err
    assert len(list(folder.glob("*.safetensors"))) == 2
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], folder)


def test_a_file_kept_from_an_older_commit_is_not_recorded(
    tmp_path: Path, hub: FakeHubServer, capsys: pytest.CaptureFixture[str]
) -> None:
    # A first download finished, then "main" moved on and the record is gone
    # (a new revision does the same). Checking config.json for an update
    # fails, so snapshot_download keeps the old copy with only a warning.
    models = tmp_path / "models"
    folder = models / "qwen3.5-4b"
    assert main([_config(tmp_path), str(models)]) == 0
    (folder / MARKER).unlink()
    hub.main = NEW_COMMIT
    hub.unreachable = {"config.json"}
    assert main([_config(tmp_path), str(models)]) == 1
    assert "config.json" in capsys.readouterr().err
    assert not is_downloaded(REGISTRY["qwen3.5-4b"], folder)
    hub.unreachable.clear()
    assert main([_config(tmp_path), str(models)]) == 0
    assert (folder / "config.json").read_bytes() == NEW_FILES["config.json"]
    assert json.loads((folder / MARKER).read_text())["commit"] == NEW_COMMIT


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
