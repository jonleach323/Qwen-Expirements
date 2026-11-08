"""`pushlab report SPEC`: an experiment's replies, with markers where its
pushes switch.

It reads the experiment file and its runner folder (pushlab.runs.folder):
the saved runs of the folder's version, their steps (pushlab.runs.shards)
and each run's condition label, from experiment.json. Under a section per
prompt come the replies of the conditions asked for, in their order, each
condition's own run first and then its controls, with a short marker before
the first token chosen with each change in the units pushed
(pushlab.analysis.markers): [A on], [A off, B on], or [A/B x8] for a long
alternation. One line says which unit each letter is; a control's random
neuron takes the letter of the unit it stands for, and its label names it.
The model's tokenizer finds each marker's place in the text. The report is
one Markdown file with no tables, so it reads on a phone. It loads no model
and writes nothing but the report.
"""

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pushlab.analysis.markers import (
    active_sets,
    legend,
    letters,
    markdown_text,
    marks,
    places,
    stand_ins,
    with_markers,
)
from pushlab.prompts import read_prompt_set
from pushlab.runs import folder, shards
from pushlab.runs.spec import ExperimentSpec, read_spec
from pushlab.store.records import RunRecord
from pushlab.store.steps import StepRecord

# What a marker means, the line under the legend.
MEANING = (
    "A marker sits before the first token chosen with that change: a push on "
    "reply token k first changes token k + 1 (decision 0008). [**A/B x8**] is "
    "8 turns of A and B."
)
FAILED = "*failed, so it has no reply; pushlab status shows why*"


def _refuse(message: str) -> int:
    print(f"pushlab report: {message}", file=sys.stderr)
    return 1


def _and(items: Sequence[str]) -> str:
    """Items in words: "a", "a and b", "a, b and c"."""
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _quoted(labels: Sequence[str]) -> str:
    """Labels in quotes, as --conditions takes them."""
    return _and([f'"{label}"' for label in labels])


def runner_folder(spec: ExperimentSpec, runs_dir: Path) -> tuple[Path, dict[str, Any]]:
    """The experiment's runner folder, runs_dir/<name>, and its experiment.json.

    ValueError says why there is none to report on: the folder is missing,
    is not a runner folder, is another experiment's, or does not name its
    runs' conditions.

    Example:
        >>> runner_folder(spec, Path("/workspace/runs"))[0]  # doctest: +SKIP
        PosixPath('/workspace/runs/s2-pulse-lengths')
    """
    path = runs_dir / spec.name
    if not path.exists():
        raise ValueError(
            f"{path} does not exist: no run of {spec.name} is saved there yet; "
            "pushlab run makes it"
        )
    if not path.is_dir():
        raise ValueError(f"{path} is not a folder")
    identity = folder.read_identity(path)
    if identity is None:
        raise ValueError(
            f"{path} is not a runner folder: it has no {folder.IDENTITY_FILE}"
        )
    if identity["experiment"] != spec.name:
        raise ValueError(
            f"{path} belongs to the experiment {identity['experiment']!r}, "
            f"not {spec.name!r}"
        )
    labels = identity.get("conditions")
    if (
        not isinstance(labels, dict)
        or not labels
        or not all(isinstance(label, str) for label in labels.values())
    ):
        raise ValueError(
            f"{path / folder.IDENTITY_FILE} does not name each run's condition"
        )
    return path, identity


def chosen_conditions(
    name: str, labels: dict[str, str], asked: Sequence[str] | None
) -> list[str]:
    """The conditions to show, in order: those asked for, each once, or else
    (asked None or empty) every one the experiment has, in the order its
    runs first show them.

    labels is experiment.json's: each run ID's condition. ValueError names
    a label the experiment does not have, and lists those it has.

    Example:
        >>> labels = {"x-000001": "none", "x-000002": "pulse"}
        >>> chosen_conditions("x", labels, None)
        ['none', 'pulse']
        >>> chosen_conditions("x", labels, ["pulsed"])
        Traceback (most recent call last):
        ...
        ValueError: x has no condition "pulsed"; it has "none" and "pulse"
    """
    known = list(dict.fromkeys(labels[run_id] for run_id in sorted(labels)))
    if not asked:
        return known
    if unknown := list(dict.fromkeys(a for a in asked if a not in known)):
        which = "condition" if len(unknown) == 1 else "conditions"
        raise ValueError(
            f"{name} has no {which} {_quoted(unknown)}; it has {_quoted(known)}"
        )
    return list(dict.fromkeys(asked))


def _counter(run_id: str) -> int:
    return int(run_id.rsplit("-", 1)[1])


def _groups(
    runs: list[RunRecord], labels: dict[str, str], chosen: list[str]
) -> dict[str, list[RunRecord]]:
    """The runs to show, by prompt in run order, each prompt's in the chosen
    conditions' order, and each own run followed by its controls."""
    rank = {label: i for i, label in enumerate(chosen)}
    by_prompt: dict[str, list[RunRecord]] = {}
    for run in runs:
        if labels.get(run.run_id) in rank:
            by_prompt.setdefault(run.prompt_id, []).append(run)
    return {
        prompt_id: sorted(
            shown,
            key=lambda run: (
                rank[labels[run.run_id]],
                run.control_of or run.run_id,  # a control goes with its run
                bool(run.control_of),
                run.run_id,
            ),
        )
        for prompt_id, shown in by_prompt.items()
    }


def _letters(
    groups: dict[str, list[RunRecord]], steps: dict[str, list[StepRecord]]
) -> dict[str, str]:
    """A letter for each unit the runs shown push, in the order they show
    them; a control's random neuron stands for its run's unit."""
    units: list[str] = []
    for shown in groups.values():
        for run in shown:
            stands_for = stand_ins(run.schedule)
            units += stands_for.values()
            for step in steps.get(run.run_id, []):
                units += (stands_for.get(unit, unit) for unit in step.active_units)
    return letters(units)


def _label(
    run: RunRecord, condition: str, letter_of: dict[str, str], seeds: bool
) -> str:
    """A run's label: its condition, its seed when there are several, and for
    a control its number and the neuron that stands in for each letter."""
    label = condition + (f", seed {run.seed}" if seeds else "")
    if not run.control_of:
        return label
    # expand numbers a run's controls right after it: control k is run + k.
    label += f", control {_counter(run.run_id) - _counter(run.control_of)}"
    swaps = sorted(
        (
            (letter_of[real], unit)
            for unit, real in stand_ins(run.schedule).items()
            if unit != real
        ),
        key=lambda swap: (len(swap[0]), swap[0]),  # A to Z, then AA
    )
    if swaps:
        label += f" ({', '.join(f'{letter} is {unit}' for letter, unit in swaps)})"
    return label


def _reply(
    run: RunRecord,
    run_steps: list[StepRecord],
    letter_of: dict[str, str],
    decode: Callable[[Sequence[int]], str],
) -> str:
    """A run's reply as Markdown, with its markers and short notes."""
    if run.status == "failed":
        return FAILED
    tokens, text = run.output_tokens, run.output_text
    notes = []
    if [s.step for s in run_steps] == list(range(len(tokens))) and [
        s.token_id for s in run_steps
    ] == tokens:
        units = [step.active_units for step in run_steps]
        sets = active_sets(units, letter_of, stand_ins(run.schedule))
        reply = with_markers(text, places(text, tokens, marks(sets), decode))
        if run.schedule and not any(sets):
            notes.append("the reply ended before any push")
    else:
        reply = markdown_text(text)
        notes.append(
            "its steps do not match its tokens, so it has no markers; "
            "pushlab verify says more"
        )
    limit = run.decoding.get("max_new_tokens")
    if isinstance(limit, int) and len(tokens) >= limit:
        notes.append(f"reached the {limit}-token limit")
    if run.status != "ok":
        notes.append(f"status {run.status}")
    return (reply or "*(no text)*") + "".join(f" *({note})*" for note in notes)


def experiment_markdown(
    spec: ExperimentSpec,
    path: Path,
    identity: dict[str, Any],
    chosen: list[str],
    decode: Callable[[Sequence[int]], str],
    tokenizer_model: str,
) -> tuple[str, int]:
    """The report on the chosen conditions as Markdown, and how many replies
    it shows.

    decode is the tokenizer's, without special tokens, and tokenizer_model
    the download it is from (<model>@<revision>): a note says when the runs
    were made with another. Notes also say when runs of the conditions are
    not saved yet, and when the prompt set changed since the version was
    made. ValueError says when no run of the conditions is saved yet, or the
    saved runs or the prompt set cannot be read.

    Example:
        >>> experiment_markdown(
        ...     spec, path, identity, ["pulse"], decode, "qwen3.5-4b@851bf6e"
        ... )[1]  # doctest: +SKIP
        5
    """
    labels, version = identity["conditions"], identity["version"]
    try:
        runs, steps = shards.read_results(path), shards.read_result_steps(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the runs saved in {path}: {error}") from None
    groups = _groups(runs, labels, chosen)
    if not groups:
        raise ValueError(
            f"no run of {_quoted(chosen)} is saved in {path} yet; pushlab status "
            "says how far the run has got"
        )
    letter_of = _letters(groups, steps)
    try:
        prompt_set = read_prompt_set(spec.prompts_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {spec.prompts}: {error}") from None
    texts = dict(zip(prompt_set.prompt_ids, prompt_set.prompts, strict=True))
    shown = [run for group in groups.values() for run in group]
    lines = [
        f"# Replies of {markdown_text(spec.name)}, with push markers",
        "",
        markdown_text(
            f"Version {version} of {spec.name}, in {path}, with the conditions "
            f"{_and(chosen)}."
        ),
        "",
        markdown_text(legend(letter_of)) or "Nothing is pushed, so no markers.",
        "",
        MEANING,
    ]
    notes = []
    planned = sum(label in chosen for label in labels.values())
    if pending := planned - len(shown):
        count = "1 run" if pending == 1 else f"{pending} runs"
        notes.append(
            f"Not saved yet: {count} of these conditions; pushlab status says how "
            "far the run has got."
        )
    if prompt_set.sha256 != identity.get("prompts_sha256"):
        notes.append(
            f"{spec.prompts} changed since version {version} was made, so the "
            "prompts' texts below are the file's now."
        )
    if identity.get("model") != tokenizer_model:
        notes.append(
            f"The runs were made with {identity.get('model')}, and the markers "
            f"placed with the tokenizer of {tokenizer_model}."
        )
    lines += [line for note in notes for line in ("", f"Note: {markdown_text(note)}")]
    seeds = len(spec.seeds) > 1 or len({run.seed for run in runs}) > 1
    for prompt_id, group in groups.items():
        text = texts.get(prompt_id)
        prompt = f"not in {spec.prompts} now" if text is None else text
        lines += ["", f"## {markdown_text(prompt_id)}", ""]
        lines.append(f"Prompt: {markdown_text(prompt)}")
        for run in group:
            label = _label(run, labels[run.run_id], letter_of, seeds)
            reply = _reply(run, steps.get(run.run_id, []), letter_of, decode)
            lines += ["", f"**{markdown_text(label)}**: {reply}"]
    replies = sum(run.status != "failed" for run in shown)
    return "\n".join(lines) + "\n", replies


def run_experiment_report(
    spec_path: Path,
    conditions: Sequence[str] | None,
    out: Path,
    runs_dir: Path | None = None,
    overwrite: bool = False,
) -> int:
    """Write the report on the experiment at spec_path to out; return the
    exit status.

    conditions are the labels to show, in order (None: every one). runs_dir
    holds the experiment's runner folder (None: folder.default_runs_dir()).
    Returns 0 once written, and 1 with the reason on stderr when it refuses:
    an experiment file it cannot read, a runner folder that is missing or
    not this experiment's, an unknown condition (it lists the experiment's),
    an out that is a folder, exists unless overwrite, or lies inside the
    runner folder, a tokenizer that is not downloaded, and saved runs it
    cannot read.

    Example:
        >>> run_experiment_report(
        ...     Path("experiments/p0/s2_pulse_lengths.yaml"), None, Path("r.md")
        ... )  # doctest: +SKIP
        0
    """
    if out.is_dir():
        return _refuse(f"--out must be a file, but {out} is a folder")
    if out.exists() and not overwrite:
        return _refuse(f"{out} already exists; pass --overwrite to replace it")
    runs_dir = folder.default_runs_dir() if runs_dir is None else runs_dir
    try:
        spec = read_spec(spec_path)
        path, identity = runner_folder(spec, runs_dir)
        chosen = chosen_conditions(spec.name, identity["conditions"], conditions)
    except ValueError as error:
        return _refuse(str(error))
    if out.resolve().is_relative_to(path.resolve()):  # where --overwrite could
        return _refuse(  # replace experiment.json or the index
            f"--out must be outside {path}, which holds only what pushlab run "
            "writes; save the report in a folder of its own, such as "
            "/workspace/reports"
        )
    from pushlab.models import load  # here, so that this module loads no torch

    try:
        tokenizer_model = f"{spec.model}@{load.downloaded_revision(spec.model)}"
        tokenizer = load.load_tokenizer(spec.model)
    except (OSError, ValueError) as error:
        return _refuse(str(error))

    def decode(tokens: Sequence[int]) -> str:
        return tokenizer.decode(list(tokens), skip_special_tokens=True)

    try:
        text, replies = experiment_markdown(
            spec, path, identity, chosen, decode, tokenizer_model
        )
    except ValueError as error:
        return _refuse(str(error))
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    except OSError as error:
        return _refuse(f"cannot write {out}: {error}")
    noun = "reply" if replies == 1 else "replies"
    print(f"Saved the report on {replies} {noun}: {out}")
    return 0
