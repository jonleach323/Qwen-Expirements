"""Calibration modes: which residual sizes turn a run's doses into amounts.

"fixed", the default, is decision 0007's: each layer's median residual size,
measured once per model revision on prompts/calibration.jsonl and saved in
calibration/ (pushlab.units.calibrate), the same for every prompt.

"per_prompt" is the Neuron Persona Library's: each prompt's own sizes. The
library scales a dose by r_L, the "median residual-stream norm after layer L
over the prompt's tokens (first token excluded)", with its prompt in the
chat template, thinking off (neuronpersonas.com, "Method"). per_prompt_norms
measures that for every prompt of a batch: the prompt runs alone through the
chat template with thinking off and no push active, and each layer's size
is the median length of that decoder layer's output over the prompt's own
tokens, its first left out, taken in float32. It is calibrate's measure on a
set of one prompt, so with an even count the median is the mean of the
middle two, as in decision 0007 (decision 0019).

The sizes go to compile or generate as their norms: a plain list for fixed,
a PerPromptNorms for per_prompt, whose row i belongs to prompt i.
"""

from collections.abc import Sequence

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.units.calibrate import layer_outputs, median_norms
from pushlab.units.dose import PerPromptNorms

CALIBRATION_MODES = ("fixed", "per_prompt")
DEFAULT_MODE = "fixed"


def prompt_norms(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str
) -> tuple[float, ...]:
    """One prompt's own median residual size at every layer, from layer 0.

    The prompt runs alone, in the chat template with thinking off, with no
    push active (layer_outputs refuses a model with one). Layer L's size is
    the median length of decoder layer L's output over the prompt's tokens,
    its first token left out.

    Example:
        >>> prompt_norms(model, tokenizer, "Who are you?")  # doctest: +SKIP
        (1.62, 2.31, ...)
    """
    if not isinstance(prompt, str):
        raise ValueError(f"prompt must be one text, not {prompt!r}")
    (outputs,) = layer_outputs(model, tokenizer, [prompt])
    return tuple(median_norms([outputs]))


def per_prompt_norms(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> PerPromptNorms:
    """Each prompt's own residual sizes, in the prompts' order.

    Row i is prompt_norms of prompts[i]. A prompt given more than once is
    measured once, and each prompt's outputs are let go before the next, so
    memory holds one prompt's at a time.

    Example:
        >>> norms = per_prompt_norms(model, tokenizer, ["Hi", "Bye"])  # doctest: +SKIP
        >>> replies = generate(  # doctest: +SKIP
        ...     model, tokenizer, ["Hi", "Bye"], [], greedy, norms, schedules=schedules
        ... )
    """
    if isinstance(prompts, str):  # else each character would be a prompt
        raise ValueError("prompts must be a list of prompts, not one text")
    prompts = list(prompts)
    if not prompts:
        raise ValueError("per-prompt calibration needs at least one prompt")
    measured: dict[str, tuple[float, ...]] = {}
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError(f"each prompt must be text, not {prompt!r}")
        if prompt not in measured:
            measured[prompt] = prompt_norms(model, tokenizer, prompt)
    return PerPromptNorms(tuple(measured[prompt] for prompt in prompts))


def norms_for(
    mode: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    fixed: Sequence[float] | None = None,
) -> tuple[float, ...] | PerPromptNorms:
    """The norms for compile or generate in a calibration mode.

    "fixed" returns the sizes given as fixed, such as
    load_calibration(...).norms, which every prompt shares; "per_prompt"
    measures each prompt's own and needs no fixed sizes.

    Example:
        >>> norms = norms_for(  # doctest: +SKIP
        ...     "fixed", model, tokenizer, prompts, calibration.norms
        ... )
    """
    if mode == "fixed":
        if fixed is None:
            raise ValueError(
                "the fixed calibration needs the model's sizes, such as "
                "load_calibration(...).norms (pushlab calibrate)"
            )
        return tuple(fixed)
    if mode == "per_prompt":
        return per_prompt_norms(model, tokenizer, prompts)
    raise ValueError(
        f"unknown calibration mode {mode!r}; the modes are "
        f"{', '.join(CALIBRATION_MODES)}"
    )
