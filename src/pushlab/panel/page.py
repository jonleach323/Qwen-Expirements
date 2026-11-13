"""The panel's page (decision 0023): a Markdown table of the personas that
pushlab panel select chose, and two example lines of each.

page_text writes it from the panel (pushlab.panel.panel_file) and each
persona's examples: the first line of two of its replies at its working
dose that speak as it, each with the prompt it answered. The page names the
Neuron Persona Library's personas, whose data states no licence, so it says
that it stays out of git, as .gitignore keeps docs/panel/ (decision 0020).
Text from the candidates list and the replies is escaped, so that Markdown
shows it as written. output_paths says where the panel file and the page
go: units/panel_<model>.yaml and docs/panel/<model>.md by default, both kept
out of git by .gitignore, or files that git would not keep either.
"""

import os
import re
import uuid
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path

from pushlab import PROJECT_ROOT
from pushlab.panel.panel_file import Mean, PanelFile, PanelPersona, Share
from pushlab.runs.estimate import exact
from pushlab.runs.ladder_command import inside_repository

LINE_LIMIT = 200  # the most characters of an example line shown
# Characters Markdown may read as markup within a line, a table's pipe and
# HTML's angle brackets and ampersand among them: each gets a backslash. A
# line's own start is never a reply's, so "#" or "-" there needs none.
_MARKUP = re.compile(r"([\\`*_\[\]<>|~&])")
Example = tuple[str, str]  # a prompt's ID, and the reply, whose first line shows
# Where .gitignore keeps the panel and its page out of git, in the repository.
IGNORED = ("units/panel_*.yaml", "docs/panel/")


def first_line(text: str, limit: int = LINE_LIMIT) -> str:
    """The first line of text that is not blank, its spaces evened, cut at a
    space before limit characters, with an ellipsis; "" for a blank text.

    Example:
        >>> first_line("\\n  Hoo!  I am   the owl.\\nSecond line.")
        'Hoo! I am the owl.'
        >>> first_line("one two three four", limit=10)
        'one two…'
    """
    line = next((line for line in text.splitlines() if line.strip()), "")
    shown = " ".join(line.split())
    if len(shown) <= limit:
        return shown
    cut = shown[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


def markdown_text(text: str) -> str:
    """text as Markdown shows it as written, on one line.

    Example:
        >>> print(markdown_text("*Hoo* | <b>"))
        \\*Hoo\\* \\| \\<b\\>
    """
    return _MARKUP.sub(r"\\\1", " ".join(text.split()))


def code_text(text: str) -> str:
    """text, such as a file's path, as Markdown shows it as written: as code,
    unless it holds a backtick.

    Example:
        >>> print(code_text("units/candidates_qwen3-8b.csv"), code_text("a`b"))
        `units/candidates_qwen3-8b.csv` a\\`b
    """
    return markdown_text(text) if "`" in text else f"`{' '.join(text.split())}`"


def share_text(share: Share) -> str:
    """A share, its rate and its interval.

    Example:
        >>> share_text(Share(28, 32, 0.875, 0.7194, 0.95))
        '28/32, 0.88 [0.72, 0.95]'
    """
    return f"{share.k}/{share.n}, {share.rate:.2f} [{share.low:.2f}, {share.high:.2f}]"


def mean_text(mean: Mean) -> str:
    """A mean and its interval.

    Example:
        >>> mean_text(Mean(32, 7.59375, 7.2, 8.0))
        '7.6 [7.2, 8.0]'
    """
    return f"{mean.mean:.1f} [{mean.low:.1f}, {mean.high:.1f}]"


def _row(rank: int, persona: PanelPersona) -> str:
    collapse = persona.collapse_dose
    cells = [
        str(rank),
        markdown_text(persona.persona),
        markdown_text(persona.key),
        persona.category,
        persona.kind,
        exact(persona.working_dose),
        "beyond the ladder" if collapse is None else exact(collapse),
        share_text(persona.reliability),
        mean_text(persona.coherence),
        share_text(persona.distinct),
    ]
    return "| " + " | ".join(cells) + " |"


def page_text(
    panel: PanelFile,
    examples: Mapping[str, Sequence[Example]],
    panel_path: str,
    confidence: float,
) -> str:
    """The page of panel, written to panel_path, its examples by key, with
    intervals at confidence, as the module's docstring says.

    Example:
        >>> chosen = PanelFile("qwen3-8b", "units/c.csv", "0" * 64, {}, ())
        >>> page_text(chosen, {}, "units/panel_qwen3-8b.yaml", 0.95).splitlines()[0]
        '# The persona panel of qwen3-8b'
    """
    percent = f"{round(confidence * 100, 6):g}%"
    lines = [
        f"# The persona panel of {panel.model}",
        "",
        "This page names the Neuron Persona Library's personas, whose data states "
        "no licence, so it stays out of git, as .gitignore keeps docs/panel/ "
        "(decision 0020).",
        "",
        f"`pushlab panel select` chose these {len(panel.personas)} personas "
        f"(decision 0023) from the candidates of {code_text(panel.candidates_file)}; "
        f"the panel file is {code_text(panel_path)}. It was made from:",
        "",
        *(f"- {name}: {markdown_text(text)}" for name, text in panel.made_from.items()),
        "",
        "Strongest first. Doses are as pushed, in each persona's direction. "
        "Reliability is the share of its replies at its working dose that speak "
        "as it, over the held-out prompts; coherence their mean score, 0 to 10; "
        "distinctness the share of its sampled replies the which_persona rubric "
        f"named it for. Intervals are {percent}.",
        "",
        "| # | Persona | Key | Category | Kind | Working dose | Collapse dose "
        "| Reliability | Coherence | Distinctness |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *(_row(rank, each) for rank, each in enumerate(panel.personas, 1)),
        "",
        "## Example lines",
        "",
        "The first line of two of each persona's replies at its working dose that "
        "speak as it, after the prompt each answered.",
    ]
    for rank, each in enumerate(panel.personas, 1):
        lines += ["", f"### {rank}. {markdown_text(each.persona)}", ""]
        shown = examples.get(each.key, ())
        lines += [
            f"- {markdown_text(prompt)}: {markdown_text(first_line(text))}"
            for prompt, text in shown
        ] or ["- (no reply speaks as it)"]
    return "\n".join(lines) + "\n"


def write_page(path: Path, text: str) -> Path:
    """Write the page's text to path, making its folder, whole beside it and
    then in its place; the path.

    Example:
        >>> write_page(Path("docs/panel/qwen3-8b.md"), text)  # doctest: +SKIP
        PosixPath('docs/panel/qwen3-8b.md')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def kept_out_of_git(path: Path) -> bool:
    """Whether a file written at path stays out of git: it is outside the
    repository, or where .gitignore keeps the panel and its page (IGNORED).

    Example:
        >>> kept_out_of_git(PROJECT_ROOT / "units" / "panel_qwen3-8b.yaml")
        True
        >>> kept_out_of_git(PROJECT_ROOT / "units" / "qwen3-8b.yaml")
        False
    """
    if not inside_repository(path):
        return True
    where = Path(path).absolute()
    parts = (where.parent.resolve() / where.name).relative_to(PROJECT_ROOT).parts
    if len(parts) == 2 and parts[0] == "units":
        return fnmatchcase(parts[1], "panel_*.yaml")
    return len(parts) > 2 and parts[:2] == ("docs", "panel")


def output_paths(out: Path | None, doc: Path | None, model: str) -> tuple[Path, Path]:
    """The panel file's path and its page's: out and doc, else the defaults
    for model. ValueError for a folder, one file given twice, or a path in
    the repository where git would keep it.

    Example:
        >>> out, doc = output_paths(None, None, "qwen3-8b")
        >>> out.relative_to(PROJECT_ROOT), doc.relative_to(PROJECT_ROOT)
        (PosixPath('units/panel_qwen3-8b.yaml'), PosixPath('docs/panel/qwen3-8b.md'))
    """
    out = PROJECT_ROOT / "units" / f"panel_{model}.yaml" if out is None else Path(out)
    doc = PROJECT_ROOT / "docs" / "panel" / f"{model}.md" if doc is None else Path(doc)
    for path, option in ((out, "--out"), (doc, "--doc")):
        if path.is_dir():
            raise ValueError(f"{option} must be a file, but {path} is a folder")
        if not kept_out_of_git(path):
            raise ValueError(
                f"{option} {path} is in the repository, where git would keep it: "
                "the panel names the Neuron Persona Library's personas, whose data "
                "stays out of git (decision 0020); write it where .gitignore keeps "
                f"it out ({' and '.join(IGNORED)}), or outside the repository"
            )
    if out.absolute() == doc.absolute():
        raise ValueError(f"--out and --doc are both {out}: give two files")
    return out, doc
