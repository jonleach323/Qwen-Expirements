"""A sheet of replies for you to label: which replies it samples, and what
its files say (decision 0021). `pushlab label export`
(pushlab.scoring.label_export) writes it.

gather reads the whole replies of one or more experiments' judged runs
(pushlab.scoring.items: "ok" or "collapsed", in each runner folder's
current version), blank ones left out, each as the judge reads it, with the
space around it stripped; distinct keeps each text once, the first in the
order of the experiments given and of their runs. sample takes an equal
share from each stratum, an experiment's condition with its random controls
apart (shares): one smaller than its share is taken whole, and the rest
spread over the others. random.Random(seed) chooses within each stratum and
then shuffles the chosen, so one seed always gives the same sheet.

labels.csv (sheet_csv) has a row per reply, under item_id, text, a column
per rubric and note, in UTF-8 with a byte-order mark, which spreadsheets
read; it names no run, condition, unit, dose or prompt, so you label blind,
as the judge does. An item's ID is the folder's name, the sample's token
(sample_token), which covers each reply and the persona or personas asked
about, and a number, so that labels of another sample, of replies changed
since or about another persona, even in a folder of the same name, match
none of its items. instructions.md (instructions) says how to label, that
a rubric needs MIN_LABELS of your labels to pass, and each rubric's answers
and their meanings. `pushlab label check` writes its disagreements file
beside them as labels.csv is written (guarded_csv).
"""

import codecs
import csv
import hashlib
import io
import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pushlab.scoring.items import judged_runs
from pushlab.scoring.personas import NONE, Persona
from pushlab.scoring.rubrics import Rubric
from pushlab.store.records import RunRecord
from pushlab.store.scores import output_sha256

SHEET_FILE = "labels.csv"
INSTRUCTIONS_FILE = "instructions.md"
KEY_FILE = "key.json"
NOTE = "note"  # the sheet's last column, for the labeler's own notes
TOKEN_DIGITS = 4  # hex digits of a sample's token, in its items' IDs
# The plan's hand labels per rubric: a rubric with fewer cannot pass.
MIN_LABELS = 200
# A cell that starts with one of these gets an apostrophe before it, so that
# a spreadsheet shows it rather than running it as a formula.
GUARDED = ("=", "+", "-", "@")
# A line of a score rubric's scale, such as "7 to 9: readable, ...".
_SCALE = re.compile(r"(\d+(?: to \d+)?): (.+)")


@dataclass(frozen=True)
class LabelItem:
    """A reply that may go on the sheet: its run, where the run is from, and
    its text with the space around it stripped, as the judge reads it."""

    spec: str  # the experiment file's path, as given
    experiment: str
    version: int
    run: RunRecord
    condition: str
    text: str

    @property
    def control(self) -> bool:
        """Whether its run is a random control, which names the run it mirrors."""
        return bool(self.run.control_of)

    @property
    def stratum(self) -> str:
        """The stratum it is sampled in (stratum_name)."""
        return stratum_name(self.experiment, self.condition, self.control)


def stratum_name(experiment: str, condition: str, control: bool) -> str:
    """A stratum's name: the experiment and the condition, and for the
    condition's random controls, "control".

    Example:
        >>> stratum_name("s2-pulses", "pulse length=16", True)
        's2-pulses / pulse length=16 / control'
    """
    return f"{experiment} / {condition}" + (" / control" if control else "")


def plural(number: int, one: str, many: str) -> str:
    """A number and the noun counted, one or many as it needs.

    Example:
        >>> plural(1, "reply", "replies"), plural(200, "reply", "replies")
        ('1 reply', '200 replies')
    """
    return f"{number} {one if number == 1 else many}"


def gather(
    specs: Sequence[Path], runs_dir: Path | None = None
) -> tuple[list[LabelItem], int, int]:
    """The replies of the experiments at specs, in the order of specs and of
    their runs, then how many blank replies were left out, and how many
    failed runs, which have none.

    ValueError says why an experiment cannot be read (items.judged_runs),
    and refuses one given twice.

    Example:
        >>> gather([Path("experiments/p0/s2_pulse_lengths.yaml")])  # doctest: +SKIP
        ([LabelItem(spec='experiments/p0/s2_pulse_lengths.yaml', ...), ...], 0, 0)
    """
    items: list[LabelItem] = []
    blank = failed = 0
    given: dict[str, Path] = {}
    for spec in specs:
        experiment = judged_runs(Path(spec), runs_dir)
        name = experiment.spec.name
        if name in given:
            raise ValueError(
                f"{given[name]} and {spec} are both the experiment {name}: give "
                "each experiment once"
            )
        given[name] = Path(spec)
        failed += experiment.failed
        for run in experiment.runs:
            text = run.output_text.strip()
            if not text:
                blank += 1
                continue
            condition = experiment.condition(run)
            version = experiment.version
            items.append(LabelItem(str(spec), name, version, run, condition, text))
    return items, blank, failed


def distinct(items: Sequence[LabelItem]) -> list[LabelItem]:
    """items with each text once: one whose text appeared already is left out.

    Example:
        >>> distinct([])
        []
    """
    seen: set[str] = set()
    kept = []
    for item in items:
        if item.text not in seen:
            seen.add(item.text)
            kept.append(item)
    return kept


def shares(sizes: Sequence[int], total: int) -> list[int]:
    """How many to take from each stratum, of the sizes given, to take total.

    Each stratum gets an equal share; the shares are whole numbers by the
    largest remainder, so, the remainders being equal, the first strata take
    one more. A stratum smaller than its share is taken whole and the rest
    shared out again among the others; with total at least the sizes' sum,
    every stratum is taken whole.

    Example:
        >>> shares([5, 5, 5, 5, 5], 12)
        [3, 3, 2, 2, 2]
        >>> shares([5, 1, 5, 5, 5], 12)
        [3, 1, 3, 3, 2]
        >>> shares([2, 3], 10)
        [2, 3]
    """
    take = [0] * len(sizes)
    left = min(total, sum(sizes))
    open_ = [place for place, size in enumerate(sizes) if size > 0]
    while left > 0 and open_:
        small = [place for place in open_ if sizes[place] * len(open_) <= left]
        if not small:
            each, extra = divmod(left, len(open_))
            for rank, place in enumerate(open_):
                take[place] = each + (rank < extra)
            break
        for place in small:
            take[place] = sizes[place]
            left -= sizes[place]
        open_ = [place for place in open_ if place not in small]
    return take


def sample(items: Sequence[LabelItem], size: int, seed: int) -> list[LabelItem]:
    """size of items, or all of them if there are fewer, shuffled.

    Each stratum, in the order of its first item, takes its share (shares);
    random.Random(seed) chooses which of its items, and then shuffles all
    those chosen, so the same items, size and seed give the same list.

    Example:
        >>> sample([], 200, 0)
        []
    """
    strata: dict[str, list[LabelItem]] = {}
    for item in items:
        strata.setdefault(item.stratum, []).append(item)
    chance = random.Random(seed)
    counts = shares([len(group) for group in strata.values()], size)
    chosen = [
        item
        for group, count in zip(strata.values(), counts, strict=True)
        for item in chance.sample(group, count)
    ]
    chance.shuffle(chosen)
    return chosen


def guard(cell: str) -> str:
    """cell, with an apostrophe before it if it starts with =, +, - or @, so
    that a spreadsheet shows it rather than running it as a formula.

    Example:
        >>> guard("=1+1"), guard("- item"), guard("Hello")
        ("'=1+1", "'- item", 'Hello')
    """
    return f"'{cell}" if cell.startswith(GUARDED) else cell


def sample_token(
    items: Sequence[LabelItem],
    persona: Persona | None = None,
    personas: Sequence[Persona] = (),
    rubrics: Sequence[Rubric] = (),
) -> str:
    """A sample's token: the first TOKEN_DIGITS hex digits of the SHA-256 of
    what its sheet asks you to label, a line each: each item, in sheet
    order, as "<experiment>/<run ID>/<its reply's SHA-256>" (output_sha256);
    persona, the one speaks_as asks about, as persona ["<name>",
    "<description>"] in JSON, which keeps a line break in its line; each of
    personas, those which_persona chooses among, as personas [...]; and
    each rubric, as rubric ["<name>", <version>, "<its file's SHA-256>"], so
    a rubric's new version gives new IDs. The runner folder's version is
    left out: replies a new version left unchanged give the same token.
    Another sample gives another, but for one in 65,536.

    Example:
        >>> sample_token([])  # the SHA-256 of no text starts with e3b0
        'e3b0'
    """
    lines = [f"{i.experiment}/{i.run.run_id}/{output_sha256(i.run)}" for i in items]
    asked = [] if persona is None else [("persona", persona)]
    asked += [("personas", each) for each in personas]
    lines += [f"{kind} {json.dumps([p.name, p.description])}" for kind, p in asked]
    lines += [f"rubric {json.dumps([r.name, r.version, r.sha256])}" for r in rubrics]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:TOKEN_DIGITS]


def item_ids(prefix: str, count: int) -> list[str]:
    """The IDs of a sheet's count items: prefix, which export makes of the
    folder's name and the sample's token, and a number from 1, in at least
    three digits.

    Example:
        >>> item_ids("coherence-1-3f2a", 2)
        ['coherence-1-3f2a-001', 'coherence-1-3f2a-002']
    """
    width = max(3, len(str(count)))
    return [f"{prefix}-{number:0{width}d}" for number in range(1, count + 1)]


def guarded_csv(rows: Iterable[Sequence[object]]) -> bytes:
    """rows as CSV that spreadsheets read: every cell guarded, in UTF-8 with
    a byte-order mark; a character that UTF-8 cannot hold, such as a lone
    surrogate that an error may quote, is written as its escape.

    Example:
        >>> guarded_csv([["=1+1", "\\ud83d"]])
        b"\\xef\\xbb\\xbf'=1+1,\\\\ud83d\\r\\n"
    """
    buffer = io.StringIO()
    csv.writer(buffer).writerows([guard(str(cell)) for cell in row] for row in rows)
    return codecs.BOM_UTF8 + buffer.getvalue().encode("utf-8", "backslashreplace")


def sheet_csv(
    ids: Sequence[str], items: Sequence[LabelItem], rubrics: Sequence[str]
) -> bytes:
    """labels.csv: a header of item_id, text, the rubrics and note, then a
    row per item, its label cells empty, every cell guarded, in UTF-8 with
    a byte-order mark.

    Example:
        >>> sheet_csv([], [], ["coherence"])
        b'\\xef\\xbb\\xbfitem_id,text,coherence,note\\r\\n'
    """
    empty = [""] * (len(rubrics) + 1)  # the labels' cells and the note's
    pairs = zip(ids, items, strict=True)
    rows = ([item_id, item.text, *empty] for item_id, item in pairs)
    return guarded_csv([["item_id", "text", *rubrics, NOTE], *rows])


def score_scale(rubric: Rubric) -> list[tuple[str, str]]:
    """The scale a score rubric's instructions spell out, a point or a range
    and its meaning a line; empty if they spell none.

    Example:
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> score_scale(load_rubric("coherence"))[:2]  # doctest: +NORMALIZE_WHITESPACE
        [('10', 'fluent and clear throughout.'),
         ('7 to 9', 'readable, with small slips or odd turns.')]
    """
    lines = (_SCALE.fullmatch(line.strip()) for line in rubric.instructions.split("\n"))
    return [(found[1], found[2]) for found in lines if found]


def _described(persona: Persona) -> str:
    return " ".join(persona.description.split()) or "(no description)"


def _answers(
    rubric: Rubric, persona: Persona | None, personas: Sequence[Persona]
) -> list[str]:
    """What a rubric's column takes, and what each answer means."""
    if rubric.kind == "label":
        lines = []
        if persona is not None:
            lines += [f"The persona: **{persona.name}**. {_described(persona)}", ""]
        lines.append("Write one of these labels:")
        return lines + [f"- `{label}`: {meaning}" for label, meaning in rubric.labels]
    if rubric.kind == "choice":
        lines = ["Write one of these personas' names, exactly as here, or `none`:"]
        lines += [f"- `{each.name}`: {_described(each)}" for each in personas]
        return lines + [f"- `{NONE}`: none of these personas"]
    lines = [f"Write a whole number from {rubric.low} to {rubric.high}."]
    if scale := score_scale(rubric):
        lines += ["", "The scale:", *(f"- {at}: {meaning}" for at, meaning in scale)]
    return lines


def instructions(
    rubrics: Sequence[Rubric],
    persona: Persona | None,
    personas: Sequence[Persona],
    count: int,
) -> str:
    """instructions.md for a sheet of count replies: how to label it, then
    for each rubric what its column takes and what each answer means, and
    the judge's own instructions, word for word. persona is the one a label
    rubric asks about, and personas those a choice rubric chooses among.

    Example:
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> text = instructions([load_rubric("coherence")], None, [], 200)
        >>> text.splitlines()[0]
        '# How to label these replies'
    """
    columns = ", ".join(f"`{rubric.name}`" for rubric in rubrics)
    lines = [
        "# How to label these replies",
        "",
        f"labels.csv holds {plural(count, 'reply', 'replies')} written by a "
        "language model, one per row. A judge model labels the same replies, "
        "and `pushlab label check` compares its labels with yours: a judge "
        "scores replies at scale only once it agrees with you "
        '(docs/plan/development-plan.md, "Judge validation").',
        "",
        "**Do not open key.json while you label.** It names the condition that "
        "made each reply, and knowing it would bias your labels.",
        "",
        "## Steps",
        "",
        "1. Open labels.csv in any spreadsheet, such as LibreOffice Calc, Excel "
        "or Google Sheets. Each row is one reply: `item_id` names it and `text` "
        "holds it.",
        f"2. For each reply, fill in its cell in each of the columns {columns}, "
        "with an answer spelled exactly as in the sections below.",
        f"3. A rubric needs {MIN_LABELS} of your labels to pass, so label every "
        "reply you can. Leave a cell empty to skip that reply for that column; "
        f"an empty cell is not a label, so it counts against the {MIN_LABELS}.",
        "4. The `note` column is yours, for anything you want to remember; it is "
        "not read as a label.",
        "5. Save the sheet as CSV (in Excel, CSV UTF-8) named labels.csv, with "
        "its `item_id` cells and its columns' names as they are.",
        "",
        "Judge each reply by its text alone, as the judge does: neither of you "
        "sees the prompt it answers. A text that starts with =, +, - or @ has "
        "an apostrophe put before it, so that the spreadsheet shows it rather "
        "than running it as a formula; the apostrophe is not part of the reply. "
        "Replies of pushed neurons can be hostile or disturbing: label in short "
        "sittings.",
    ]
    for rubric in rubrics:
        lines += ["", f"## {rubric.name}", "", *_answers(rubric, persona, personas)]
        told = rubric.ask(persona, tuple(personas)).system
        lines += [
            "",
            "What the judge is told, word for word (it answers in JSON; you write "
            "the answer alone):",
            "",
            *(f"> {line}".rstrip() for line in told.split("\n")),
        ]
    return "\n".join(lines) + "\n"
