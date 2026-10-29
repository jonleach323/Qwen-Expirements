"""Turning a prompt into chat-formatted token IDs: pushlab.models.chat."""

from transformers import (
    PreTrainedTokenizerBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3ForCausalLM,
)

from pushlab.models.chat import chat_prompt_ids, stop_token_ids


def test_thinking_off_closes_an_empty_thinking_block(
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    ids = chat_prompt_ids(tiny_tokenizer, "Hi there", thinking=False)
    assert tiny_tokenizer.decode(ids) == (
        "<|im_start|>user\nHi there<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_thinking_on_leaves_the_thinking_block_open(
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    ids = chat_prompt_ids(tiny_tokenizer, "Hi there", thinking=True)
    assert tiny_tokenizer.decode(ids).endswith("<|im_start|>assistant\n<think>\n")


def test_the_chat_markers_are_single_tokens(
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    ids = chat_prompt_ids(tiny_tokenizer, "Hi", thinking=False)
    start, think = tiny_tokenizer.convert_tokens_to_ids(["<|im_start|>", "<think>"])
    assert ids[0] == start
    assert ids.count(think) == 1


def test_stop_ids_are_the_end_of_turn_and_the_models_end_of_text(
    tiny_tokenizer: PreTrainedTokenizerBase,
    tiny_vl_model: Qwen3_5ForConditionalGeneration,
) -> None:
    # Qwen3.5 ends a chat turn with <|im_end|>, but its config names only
    # <|endoftext|>, so both must stop the reply.
    expected = tiny_tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    assert stop_token_ids(tiny_vl_model, tiny_tokenizer) == sorted(expected)


def test_stop_ids_take_every_end_token_the_model_lists(
    tiny_tokenizer: PreTrainedTokenizerBase, tiny_model: Qwen3ForCausalLM
) -> None:
    end_of_turn = tiny_tokenizer.eos_token_id
    tiny_model.generation_config.eos_token_id = [end_of_turn, 300, 301]
    assert stop_token_ids(tiny_model, tiny_tokenizer) == [end_of_turn, 300, 301]
    tiny_model.generation_config.eos_token_id = None
    assert stop_token_ids(tiny_model, tiny_tokenizer) == [end_of_turn]
