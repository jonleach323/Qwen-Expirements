"""Golden runs: each golden case run alone, and its reply's fingerprint.

run_case runs one case of a golden cases file (pushlab.runs.golden_cases)
alone, in one generate() call of its own, so no other case shares its batch
and changes it. Its schedule is the case's own (GoldenCase.schedule): none,
or a constant push of its neuron or set, with the fixed calibration's sizes
or the prompt's own (pushlab.units.calibration_mode).
fingerprint keeps what a reply is checked by: the SHA-256 of its token IDs,
with its token count, its first 8 token IDs and the start of its text,
which show what changed. pushlab.runs.golden_fingerprints keeps the
fingerprints with the environment that made them, and compares a new run
with them (decision 0023; development plan, "Golden tests").
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine import generate as generate_module  # looked up per call
from pushlab.engine.generate import Generation
from pushlab.runs.golden_cases import GoldenCase, GoldenCases
from pushlab.units.calibrate import load_calibration
from pushlab.units.calibration_mode import norms_for

FIRST_TOKENS = 8  # token IDs a fingerprint keeps, to show what changed
TEXT_START = 80  # characters of the reply's text a fingerprint keeps


@dataclass(frozen=True)
class Fingerprint:
    """What a reply is checked by: sha256, the SHA-256 of its token IDs; and,
    to show what changed, its token count, first token IDs and the start of
    its text."""

    sha256: str
    token_count: int
    first_tokens: tuple[int, ...]
    text_start: str


def fingerprint(reply: Generation) -> Fingerprint:
    """A reply's fingerprint. The SHA-256 is of its token IDs as compact JSON,
    such as [13048,248046], so any token changes it, the end token too.

    Example:
        >>> found = fingerprint(Generation("Hi", (13048, 248046), "end_of_turn", (1,)))
        >>> found.sha256[:16], found.token_count, found.first_tokens, found.text_start
        ('2886ea4529632ba9', 2, (13048, 248046), 'Hi')
    """
    ids = [int(token) for token in reply.token_ids]
    text = json.dumps(ids, separators=(",", ":"))
    return Fingerprint(
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=len(ids),
        first_tokens=tuple(ids[:FIRST_TOKENS]),
        text_start=reply.text[:TEXT_START],
    )


def fixed_norms(cases: GoldenCases, revision: str) -> tuple[float, ...] | None:
    """The fixed calibration's sizes for the model at revision, when a case
    takes them; else None. FileNotFoundError, naming `pushlab calibrate`,
    when the revision has none (pushlab.units.calibrate.load_calibration).

    Example:
        >>> len(fixed_norms(cases, loaded.revision))  # doctest: +SKIP
        36
    """
    if not cases.needs_fixed_calibration:
        return None
    return load_calibration(cases.model, revision).norms


def run_case(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    case: GoldenCase,
    fixed: Sequence[float] | None = None,
) -> Generation:
    """Run one case alone: one generate() call with its prompt and schedule.

    A push under the fixed calibration takes fixed, the model's sizes from
    fixed_norms; one under per_prompt measures the prompt's own first
    (pushlab.units.calibration_mode.per_prompt_norms).

    Example:
        >>> reply = run_case(
        ...     loaded.model, loaded.tokenizer, case, fixed
        ... )  # doctest: +SKIP
        >>> fingerprint(reply).token_count  # doctest: +SKIP
        32
    """
    norms = None
    if case.push is not None:
        try:
            norms = norms_for(
                case.push.calibration, model, tokenizer, [case.prompt], fixed
            )
        except ValueError as error:
            raise ValueError(f"{case.case_id}: {error}") from None
    (reply,) = generate_module.generate(
        model,
        tokenizer,
        [case.prompt],
        [],
        case.decoding,
        norms,
        schedules=[case.schedule()],
    )
    return reply


def run_cases(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cases: GoldenCases,
    fixed: Sequence[float] | None = None,
    report: Callable[[GoldenCase, Fingerprint], None] | None = None,
) -> dict[str, Fingerprint]:
    """Run every case alone, in order; each reply's fingerprint, by case ID.

    report, if given, is called with each case and its fingerprint as soon
    as it is done.

    Example:
        >>> found = run_cases(
        ...     loaded.model, loaded.tokenizer, cases, fixed
        ... )  # doctest: +SKIP
        >>> list(found)[:2]  # doctest: +SKIP
        ['kettle-greedy', 'kettle-neuron-fixed']
    """
    found: dict[str, Fingerprint] = {}
    for case in cases.cases:
        found[case.case_id] = fingerprint(run_case(model, tokenizer, case, fixed))
        if report is not None:
            report(case, found[case.case_id])
    return found
