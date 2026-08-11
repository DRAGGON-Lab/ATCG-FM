import math

import torch
from atcg.models import GenomicLanguageModel, attention_tiny
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
