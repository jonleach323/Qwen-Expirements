"""Greedy decoding, one token at a time, reusing the model's cache."""

from collections.abc import Collection, Sequence

import torch
from torch import nn


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    input_ids: Sequence[int],
    max_new_tokens: int,
    stop_token_ids: Collection[int] = (),
) -> list[int]:
    """Return the most likely next token, again and again, after input_ids.

    The first pass reads the whole prompt; each later pass feeds only the
    newest token and reuses the model's cache of everything before it. Each
    pass scores only the newest position, as Hugging Face's generate() does,
    so a long prompt does not need a score for every word at every position.
    It stops after max_new_tokens tokens, or right after a token in
    stop_token_ids, which is kept as the last token. The loop is pushlab's
    own rather than generate(), so no setting from the model's generation
    config can change it.

    Example:
        >>> from transformers import Qwen3Config, Qwen3ForCausalLM
        >>> config = Qwen3Config(
        ...     vocab_size=512,
        ...     hidden_size=64,
        ...     intermediate_size=128,
        ...     num_hidden_layers=2,
        ...     num_attention_heads=4,
        ...     num_key_value_heads=2,
        ...     head_dim=16,
        ... )
        >>> model = Qwen3ForCausalLM(config).eval()
        >>> len(greedy_generate(model, [5, 17, 42], max_new_tokens=4))
        4
    """
    if not input_ids:
        raise ValueError("need at least one prompt token")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, not {max_new_tokens}")
    device = next(model.parameters()).device
    output = model(
        input_ids=torch.tensor([list(input_ids)], device=device),
        use_cache=True,
        logits_to_keep=1,
    )
    tokens: list[int] = []
    while True:
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
        if token in stop_token_ids or len(tokens) == max_new_tokens:
            return tokens
        output = model(
            input_ids=torch.tensor([[token]], device=device),
            past_key_values=output.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
