"""The which_persona check of pushlab panel select (pushlab.panel.distinct,
decision 0023): each finalist's sample, the question, the verdicts saved and
read back, and who is told apart.

Replies and runs are made up here, none of them the Neuron Persona
Library's; a FakeBackend stands in for the judge.
"""

import doctest
import json
from collections.abc import Sequence
from fractions import Fraction

import pytest

from pushlab.panel import distinct
from pushlab.panel.distinct import (
    NO_VERDICT,
    Told,
    confusion_text,
    distinct_question,
    draw_sample,
    sampled_experiment,
    saved_choices,
    speaking,
    told_apart,
)
from pushlab.panel.stats import Reply
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.items import Experiment
from pushlab.scoring.judge import Judge
from pushlab.scoring.personas import NONE
from pushlab.scoring.rubrics import Request
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow, now, output_sha256

OWL, CAPTAIN, TEAPOT = "Brass Owl", "Salt Captain", "Velvet Teapot"


def _replies(role: str, count: int, speaks: bool | None = True) -> list[Reply]:
    return [
        Reply(f"{role}-{n:02d}", f"p{n}", f"The {role} answers prompt {n}.", speaks)
        for n in range(count)
    ]


def test_a_sample_draws_only_replies_that_speak_as_the_persona() -> None:
    replies = _replies("owl", 12) + _replies("owl-quiet", 3, speaks=False)
    replies += _replies("owl-unjudged", 2, speaks=None)
    assert speaking(replies) == replies[:12]
    sample = draw_sample(replies, 8, 0, "library/L9N1-")
    assert len(sample) == 8 and set(sample) <= set(replies[:12])
    # In the replies' order, and the same each time: the seed and the key
    # alone draw it, whoever else is a finalist.
    assert sample == sorted(sample, key=replies.index)
    assert draw_sample(replies, 8, 0, "library/L9N1-") == sample
    others = {draw_sample(replies, 8, seed, "library/L9N1-")[0] for seed in range(20)}
    assert len(others) > 1
    # Fewer than asked for: all of them.
    assert draw_sample(replies[:5], 8, 0, "library/L9N1-") == replies[:5]


def test_the_question_lists_every_finalist_by_name() -> None:
    question = distinct_question([OWL, CAPTAIN])
    assert question.rubric.name == "which_persona"
    assert question.rubric.options(question.personas) == [OWL, CAPTAIN, NONE]
    request = question.request("Hoo.")
    assert request.question == f"Personas:\n- {OWL}\n- {CAPTAIN}"
    # One question for every reply, whoever's it is.
    assert question.sha256 == distinct_question([OWL, CAPTAIN]).sha256
    assert question.sha256 != distinct_question([CAPTAIN, OWL]).sha256


def test_a_finalist_is_told_apart_when_named_for_enough_of_its_sample() -> None:
    told = Told("k", OWL, (OWL,) * 7 + (CAPTAIN,))
    assert (told.k, told.n, told.rate) == (7, 8, Fraction(7, 8))
    assert not told.passes(0.9) and told.passes(0.875)
    assert Told("k", OWL, (OWL,) * 9 + (None,)).passes(0.9)
    assert not Told("k", OWL, ()).passes(0.5)
    found = Told("k", OWL, (CAPTAIN, NONE, None, CAPTAIN, OWL))
    assert found.confusions == {CAPTAIN: 2, NONE: 1, NO_VERDICT: 1}
    assert confusion_text(found) == f"{CAPTAIN} 2, {NO_VERDICT} 1, {NONE} 1"
    share = Told("k", OWL, (OWL,) * 8).share(0.95)
    assert (share.k, share.n, share.rate, share.low, share.high) == (
        8,
        8,
        1.0,
        0.6756,
        1.0,
    )


def _choices(judge: Judge, names: Sequence[str], samples: dict) -> dict:
    """The judge's choice for each sampled reply, by run ID."""
    question = distinct_question(names)
    replies = [reply for each in samples.values() for reply in each]
    verdicts = judge.verdicts([question.job(reply.text) for reply in replies])
    found = zip(replies, verdicts, strict=True)
    return {reply.run_id: v.label if v.ok else None for reply, v in found}


def test_the_check_rejects_the_one_of_two_personas_the_judge_confuses() -> None:
    names = {"owl": OWL, "captain": CAPTAIN, "teapot": TEAPOT}

    def answer(request: Request) -> str:
        role = request.text.split()[1]
        # The captain's replies are taken for the owl's.
        chosen = names["owl" if role == "captain" else role]
        return json.dumps({"reason": "Ahoy.", "choice": chosen})

    fake = FakeBackend(answer)
    samples = {role: _replies(role, 4) for role in names}
    choices = _choices(Judge(fake), list(names.values()), samples)
    told = told_apart(samples, names, choices)
    assert fake.calls == 12
    assert [told[role].passes(0.9) for role in names] == [True, False, True]
    assert told["captain"].confusions == {OWL: 4}
    # A reply the judge could not answer counts against its persona.
    fake = FakeBackend(lambda request: "not an answer")
    choices = _choices(Judge(fake, attempts=1), list(names.values()), samples)
    assert told_apart(samples, names, choices)["owl"].confusions == {NO_VERDICT: 4}
    missing = told_apart(samples, names, {})["teapot"]
    assert (missing.k, missing.n, missing.choices[0]) == (0, 4, None)


def _run(run_id: str, text: str) -> RunRecord:
    return RunRecord(
        run_id, "x", "m", "p", [], {}, 0, "", "", text, list(text.encode()), "ok"
    )


def _row(run: RunRecord, choice: str | None, **changes: object) -> ScoreRow:
    sha = output_sha256(run)
    question = distinct_question([OWL, CAPTAIN])
    status = "ok" if choice else "failed"
    values: dict = {
        "run_id": run.run_id,
        "version": 1,
        "scorer": "which_persona",
        "scorer_version": 1,
        "judge": "api:judge",
        "question_sha256": question.sha256,
        "persona": "",
        "label": choice,
        "value": None,
        "status": status,
        "error": "" if choice else "the judge declined",
        "attempts": 1,
        "details": {},
        "text_sha256": sha,
        "output_sha256": sha,
        "created_at": now(),
    }
    return ScoreRow(**(values | changes))


def test_saved_choices_read_the_newest_verdict_of_each_sampled_reply() -> None:
    runs = [_run(f"x-00000{n}", f"Hoo {n}.") for n in range(1, 5)]
    experiment = Experiment(None, None, {"version": 1}, runs, 0)  # type: ignore[arg-type]
    question = distinct_question([OWL, CAPTAIN])
    one, two, three, four = runs
    rows = [
        _row(one, CAPTAIN, created_at="2026-11-13T09:00:00Z"),
        _row(one, OWL, created_at="2026-11-13T10:00:00Z"),  # the newest wins
        _row(two, None),  # failed
        _row(three, OWL, judge="api:other"),  # another judge's
        _row(three, OWL, question_sha256="a" * 64),  # another question's
        _row(four, OWL, version=2),  # another version's
    ]
    choices = saved_choices(rows, experiment, question, "api:judge")
    assert choices == {one.run_id: OWL, two.run_id: None}
    part = sampled_experiment(experiment, [four.run_id, one.run_id])
    assert [run.run_id for run in part.runs] == [one.run_id, four.run_id]


def test_the_examples_in_the_docstrings_hold() -> None:
    result = doctest.testmod(distinct)
    assert result.failed == 0 and result.attempted > 0


@pytest.mark.parametrize("count", [0, 1])
def test_a_sample_of_none_or_one(count: int) -> None:
    replies = _replies("owl", 3)
    assert len(draw_sample(replies, count, 0, "k")) == count
