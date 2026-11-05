"""`pushlab report`: run folders' replies side by side, in one Markdown table.

One row per prompt and one column per run folder, each cell the reply that
folder's run gave the prompt. Above the table, a list says what each folder
is: where it lives, how many runs it holds, from which prompt set, and what
was pushed. The prompts' texts come from the folders' manifests when they
hold them, and folders that give one prompt ID two texts are refused.
"""

import html
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pushlab.store.records import MANIFEST_FILE, RUNS_FILE, RunRecord, read_runs

MISSING = "—"  # a prompt the folder has no run of


def _refuse(message: str) -> int:
    print(f"pushlab report: {message}", file=sys.stderr)
    return 1


def cell(text: str) -> str:
    """text made safe for one cell of a Markdown table, and kept readable.

    Every kind of line break (\\n, \\r, U+2028 and the like) becomes <br>,
    so no reply can end its row early.

    Example:
        >>> cell("a | b\\nand <c> & d")
        'a \\\\| b<br>and &lt;c&gt; &amp; d'
    """
    text = html.escape(text, quote=False)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return "<br>".join(text.splitlines())


def describe_window(start: int, end: int | None, include_prompt: bool) -> str:
    """The passes a push is on, in words.

    A push on reply token k is on the pass that reads token k, which chooses
    token k + 1 (decision 0008).

    Example:
        >>> describe_window(0, None, True)
        'the prompt and every reply token'
        >>> describe_window(2, 5, False)
        'reply tokens 2 to 4'
    """
    if end is None:
        reply = "every reply token" if start == 0 else f"reply tokens from {start} on"
    elif end == start + 1:
        reply = f"reply token {start}"
    else:
        reply = f"reply tokens {start} to {end - 1}"
    return f"the prompt and {reply}" if include_prompt else reply


def describe_push(segment: dict[str, Any]) -> str:
    """One segment of a pushlab steer schedule, in words.

    Example:
        >>> print(  # doctest: +NORMALIZE_WHITESPACE
        ...     describe_push(
        ...         {
        ...             "unit": "qwen3.5-4b:L14:N8949",
        ...             "alpha": 1.0,
        ...             "start": 0,
        ...             "end": None,
        ...             "include_prompt": True,
        ...             "amount": 23.4917,
        ...             "control_for": None,
        ...         }
        ...     )
        ... )
        qwen3.5-4b:L14:N8949 at +1.0 (adds +23.4917) on the prompt and every reply
        token
    """
    text = f"{segment.get('unit')} at {segment.get('alpha', 0):+}"
    if (value := segment.get("amount")) is not None:
        text += f" (adds {value:+.4f})"
    window = describe_window(
        segment.get("start", 0), segment.get("end"), bool(segment.get("include_prompt"))
    )
    text += f" on {window}"
    if control_for := segment.get("control_for"):
        text += f", a random control for {control_for}"
    return text


def _plan(segment: dict[str, Any]) -> dict[str, Any]:
    """A segment without touched, which differs from run to run."""
    return {key: value for key, value in segment.items() if key != "touched"}


def _pushes(runs: Sequence[RunRecord]) -> str:
    plans = {
        json.dumps([_plan(s) for s in run.schedule], sort_keys=True) for run in runs
    }
    if len(plans) > 1:
        return "pushes that differ from run to run"
    schedule = runs[0].schedule
    return "; ".join(map(describe_push, schedule)) if schedule else "no push"


def _manifest(folder: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((folder / MANIFEST_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _prompt_texts(manifest: dict[str, Any]) -> dict[str, str]:
    """The prompts' texts a manifest holds: steer's, or baseline's one."""
    texts = manifest.get("prompts")
    found = dict(texts) if isinstance(texts, dict) else {}
    if isinstance(manifest.get("prompt_id"), str) and isinstance(
        manifest.get("prompt"), str
    ):
        found[manifest["prompt_id"]] = manifest["prompt"]
    return {key: value for key, value in found.items() if isinstance(value, str)}


def _prompt_set(manifest: dict[str, Any]) -> str:
    """Which prompt set a folder used, in words, if its manifest says."""
    name, version = manifest.get("prompt_set"), manifest.get("prompt_set_version")
    sha256 = manifest.get("prompt_set_sha256")
    if not isinstance(name, str):
        return ""
    text = f", prompts from {name}"
    if isinstance(version, int):
        text += f" version {version}"
    if isinstance(sha256, str):
        text += f" (SHA-256 {sha256[:12]}…)"
    return text


def _reply(run: RunRecord, stops: list[int] | None) -> str:
    """The cell for one run's reply, with notes where the reply needs them.

    stops are the tokens that end a reply, if the folder's manifest records
    them. Without them, a reply of the limit's length may also have ended
    its turn on its last token.
    """
    text = cell(run.output_text)
    limit = run.decoding.get("max_new_tokens")
    tokens = run.output_tokens
    if isinstance(limit, int) and len(tokens) >= limit:
        if stops is None:
            text += f"<br>(reached the {limit}-token limit)"
        elif not tokens or tokens[-1] not in stops:
            text += f"<br>(cut off at the {limit}-token limit)"
    schedule = run.schedule
    if (
        schedule
        and all("touched" in segment for segment in schedule)
        and not any(segment.get("include_prompt") for segment in schedule)
        and all(segment["touched"] == 0 for segment in schedule)
    ):
        text += "<br>(the reply ended before any push: nothing was pushed)"
    return text


def _labels(folders: Sequence[Path]) -> list[str]:
    """Each folder's column header: its name, numbered if names repeat."""
    labels: list[str] = []
    for folder in folders:
        name = folder.name or folder.resolve().name or str(folder)
        label, number = name, 2
        while cell(label) in labels:
            label, number = f"{name} ({number})", number + 1
        labels.append(cell(label))
    return labels


def side_by_side(folders: Sequence[Path]) -> str:
    """The report on folders, as Markdown text.

    Raises ValueError if a folder has no runs or two runs of one prompt, or
    if two folders give one prompt ID different texts.

    Example:
        >>> side_by_side([Path("/workspace/runs/none")])  # doctest: +SKIP
        '# Replies side by side...'
    """
    if not folders:
        raise ValueError("give at least one run folder")
    columns: list[dict[str, RunRecord]] = []
    manifests: list[dict[str, Any]] = []
    texts: dict[str, tuple[str, Path]] = {}
    order: list[str] = []
    for folder in folders:
        if not (folder / RUNS_FILE).is_file():
            raise ValueError(f"{folder} holds no {RUNS_FILE}")
        runs = read_runs(folder)
        if not runs:
            raise ValueError(f"{folder / RUNS_FILE} holds no runs")
        by_prompt: dict[str, RunRecord] = {}
        for run in runs:
            if run.prompt_id in by_prompt:
                raise ValueError(
                    f"{folder} has more than one run of prompt {run.prompt_id}; "
                    "a report needs one run per prompt and folder"
                )
            by_prompt[run.prompt_id] = run
            if run.prompt_id not in order:
                order.append(run.prompt_id)
        columns.append(by_prompt)
        manifest = _manifest(folder)
        manifests.append(manifest)
        for prompt_id, text in _prompt_texts(manifest).items():
            if prompt_id in texts and texts[prompt_id][0] != text:
                raise ValueError(
                    f"prompt {prompt_id} reads differently in {texts[prompt_id][1]} "
                    f"and in {folder}; a report compares replies to the same prompts"
                )
            texts.setdefault(prompt_id, (text, folder))
    labels = _labels(folders)
    lines = [
        "# Replies side by side",
        "",
        f"Made with `pushlab report` from {len(folders)} run folders.",
        "",
        "## The run folders",
        "",
    ]
    for label, folder, by_prompt, manifest in zip(
        labels, folders, columns, manifests, strict=True
    ):
        runs = list(by_prompt.values())
        decoding = runs[0].decoding
        lines.append(
            f"- **{label}**: {cell(str(folder))}, {len(runs)} runs of "
            f"{cell(runs[0].model)}{cell(_prompt_set(manifest))}, "
            f"{decoding.get('strategy')} up to {decoding.get('max_new_tokens')} "
            f"tokens: {cell(_pushes(runs))}"
        )
    lines += ["", "## Replies", "", "| Prompt | " + " | ".join(labels) + " |"]
    lines.append("| --- " * (len(labels) + 1) + "|")
    stops = []
    for manifest in manifests:
        ids = manifest.get("stop_token_ids")
        valid = isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        stops.append(ids if valid else None)
    for prompt_id in order:
        prompt = f"**{cell(prompt_id)}**"
        if prompt_id in texts:
            prompt += f"<br>{cell(texts[prompt_id][0])}"
        replies = [
            _reply(by_prompt[prompt_id], folder_stops)
            if prompt_id in by_prompt
            else MISSING
            for by_prompt, folder_stops in zip(columns, stops, strict=True)
        ]
        lines.append(f"| {prompt} | " + " | ".join(replies) + " |")
    return "\n".join(lines) + "\n"


def run_report(folders: Sequence[Path], out: Path, overwrite: bool = False) -> int:
    """Write the report on folders to out; return the exit status.

    Returns 0 once written, and 1 with the reason on stderr when it refuses:
    a folder it cannot report on, an out that is a folder or cannot be
    written, or an existing out unless overwrite.

    Example:
        >>> run_report(
        ...     [Path("runs/none"), Path("runs/dose1")], Path("r.md")
        ... )  # doctest: +SKIP
        0
    """
    if out.is_dir():
        return _refuse(f"--out must be a file, but {out} is a folder")
    if out.exists() and not overwrite:
        return _refuse(f"{out} already exists; pass --overwrite to replace it")
    try:
        text = side_by_side(folders)
    except (OSError, ValueError) as error:
        return _refuse(str(error))
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    except OSError as error:
        return _refuse(f"cannot write {out}: {error}")
    print(f"Saved the report on {len(folders)} run folders: {out}")
    return 0
