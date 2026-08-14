import math

import pytest
import torch

from atcg.models import (
    GenomicLanguageModel,
    MemoryMode,
    attention_tiny,
    titans_mac_tiny,
    titans_memory_tiny,
)
from atcg.runtime import generate, score_sequence
from atcg.sequence import FixedAlphabetTokenizer


def _model_and_tokenizer() -> tuple[GenomicLanguageModel, FixedAlphabetTokenizer]:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGTN")
    torch.manual_seed(37)
    model = GenomicLanguageModel(
        attention_tiny(
            tokenizer.vocab_size,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_seq_len=32,
        )
    )
    return model, tokenizer


def _stateful_model(
    architecture: str,
) -> tuple[GenomicLanguageModel, FixedAlphabetTokenizer]:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGTN")
    if architecture == "memory":
        config = titans_memory_tiny(
            tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            max_seq_len=4,
            expansion_factor=2,
            projection_kernel_size=2,
        )
    elif architecture == "mac":
        config = titans_mac_tiny(
            tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            segment_length=4,
            n_heads=2,
            persistent_tokens=2,
            memory_expansion_factor=2,
            projection_kernel_size=2,
        )
    else:
        raise ValueError(f"unknown architecture {architecture}")
    torch.manual_seed(43)
    return GenomicLanguageModel(config), tokenizer


def test_sequence_score_reports_consistent_units() -> None:
    model, tokenizer = _model_and_tokenizer()

    score = score_sequence(model, tokenizer, "ACGTN")

    assert score.token_count == 6
    assert math.isfinite(score.total_nll)
    assert score.mean_nll == score.total_nll / score.token_count
    assert score.bits_per_token == score.mean_nll / math.log(2.0)


def test_generation_can_be_restricted_to_biological_tokens() -> None:
    model, tokenizer = _model_and_tokenizer()
    allowed = tuple(range(len(tokenizer.alphabet)))

    first = generate(
        model,
        tokenizer,
        "AC",
        max_new_tokens=5,
        allowed_token_ids=allowed,
        seed=41,
    )
    second = generate(
        model,
        tokenizer,
        "AC",
        max_new_tokens=5,
        allowed_token_ids=allowed,
        seed=41,
    )

    assert first == second
    assert set(first.sequence) <= set(tokenizer.alphabet)
    assert len(first.generated_token_ids) == 5


@pytest.mark.parametrize("architecture", ("memory", "mac"))
@pytest.mark.parametrize("memory_mode", ("adaptive", "frozen", "disabled"))
def test_stateful_score_and_generation_share_segmented_inference(
    architecture: str,
    memory_mode: MemoryMode,
) -> None:
    model, tokenizer = _stateful_model(architecture)

    score = score_sequence(model, tokenizer, "ACGTAC", memory_mode=memory_mode)
    allowed = tuple(range(len(tokenizer.alphabet)))
    first = generate(
        model,
        tokenizer,
        "AC",
        max_new_tokens=5,
        allowed_token_ids=allowed,
        seed=47,
        memory_mode=memory_mode,
    )
    second = generate(
        model,
        tokenizer,
        "AC",
        max_new_tokens=5,
        allowed_token_ids=allowed,
        seed=47,
        memory_mode=memory_mode,
    )

    assert score.token_count == 7
    assert math.isfinite(score.total_nll)
    assert first == second
    assert len(first.generated_token_ids) == 5
    assert len(first.token_ids) > model.config.segment_length
