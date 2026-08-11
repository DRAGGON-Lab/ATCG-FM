import pytest
import torch

from atcg.models import GenomicLanguageModel, ModelConfig, attention_tiny, hybrid_tiny

MODEL_CONFIGS = [
    attention_tiny(vocab_size=20, d_model=16, n_heads=4, n_layers=4, max_seq_len=16),
    hybrid_tiny(vocab_size=20, d_model=16, n_heads=4, n_layers=4, max_seq_len=16),
]


@pytest.mark.parametrize("config", MODEL_CONFIGS)
def test_language_model_returns_per_token_logits(config: ModelConfig) -> None:
    model = GenomicLanguageModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 11))

    logits = model(input_ids)

    assert logits.shape == (2, 11, config.vocab_size)
    assert model.parameter_count() > 0


def test_language_model_is_causal_across_hybrid_stack() -> None:
    torch.manual_seed(23)
    config = hybrid_tiny(vocab_size=20, d_model=16, n_heads=4, n_layers=4, max_seq_len=16)
    model = GenomicLanguageModel(config).eval()
    inputs = torch.randint(0, config.vocab_size, (1, 12))
    changed = inputs.clone()
    changed[:, 8:] = torch.randint(0, config.vocab_size, (1, 4))

    with torch.inference_mode():
        original_prefix = model(inputs)[:, :8]
        changed_prefix = model(changed)[:, :8]

    torch.testing.assert_close(original_prefix, changed_prefix, atol=1e-5, rtol=1e-5)


def test_language_model_rejects_sequences_beyond_configured_context() -> None:
    model = GenomicLanguageModel(
        attention_tiny(vocab_size=20, d_model=16, n_heads=4, n_layers=1, max_seq_len=8)
    )

    with pytest.raises(ValueError, match="exceeds maximum"):
        model(torch.zeros((1, 9), dtype=torch.long))
