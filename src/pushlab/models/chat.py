"""Chat formatting: a user's prompt as the token IDs a chat model expects."""

from transformers import PreTrainedModel, PreTrainedTokenizerBase


def chat_prompt_ids(
    tokenizer: PreTrainedTokenizerBase, prompt: str, *, thinking: bool
) -> list[int]:
    """Token IDs for one user turn followed by the start of the model's reply.

    The tokenizer's chat template does the formatting. With thinking off,
    Qwen's template ends the prompt with an empty thinking block, so the
    reply starts with the answer. Qwen3.5 can only be switched this way: it
    ignores the /think and /nothink commands.

    Example:
        >>> ids = chat_prompt_ids(tokenizer, "Hi", thinking=False)  # doctest: +SKIP
        >>> tokenizer.decode(ids)  # doctest: +SKIP
        '<|im_start|>user\\nHi<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=thinking,
        tokenize=False,
    )
    # The template already holds every special token the model expects.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def stop_token_ids(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> list[int]:
    """The tokens that end a reply: the end of the chat turn and the model's own.

    Qwen3.5-4B ends a chat turn with <|im_end|>, the tokenizer's end token,
    but its config names only <|endoftext|>. Stopping at either keeps the
    reply to one turn.

    Example:
        >>> stop_token_ids(model, tokenizer)  # doctest: +SKIP
        [248044, 248046]
    """
    ids = {tokenizer.eos_token_id}
    configured = model.generation_config.eos_token_id
    if isinstance(configured, int):
        ids.add(configured)
    elif configured is not None:
        ids.update(configured)
    ids.discard(None)
    return sorted(ids)
