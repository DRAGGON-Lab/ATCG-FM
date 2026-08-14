import numpy as np
import pytest
import torch

from atcg.eval import AtcgGFMModel, StatefulInferencePolicy
from atcg.models import GenomicLanguageModel, attention_tiny, titans_memory_tiny
from atcg.sequence import FixedAlphabetTokenizer


def _adapter(*, pooling: str = "last") -> tuple[AtcgGFMModel, FixedAlphabetTokenizer]:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGTN")
    torch.manual_seed(61)
    model = GenomicLanguageModel(
        attention_tiny(
            tokenizer.vocab_size,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_seq_len=16,
        )
    )
    return AtcgGFMModel(model, tokenizer, pooling=pooling), tokenizer


def test_adapter_returns_base_aligned_gfmbench_outputs() -> None:
    adapter, tokenizer = _adapter()
    sequences = ["ACGT", "TGCA"]

    probabilities, embeddings, representatives = adapter.infer_sequence_to_sequence(sequences)

    assert probabilities.shape == (2, 4)
    assert embeddings.shape == (2, 4, 16)
    assert representatives.shape == (2, 16)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    np.testing.assert_allclose(representatives, embeddings[:, -1])

    first_targets = torch.tensor([tokenizer.encode(sequences[0])])
    first_inputs = torch.tensor([[tokenizer.bos_id, *first_targets[0].tolist()]])
    with torch.inference_mode():
        logits = adapter.model(first_inputs).logits[:, :-1]
        expected = logits.softmax(-1).gather(-1, first_targets.unsqueeze(-1)).squeeze(-1)
    np.testing.assert_allclose(probabilities[0], expected[0].numpy(), rtol=1e-6)


def test_adapter_has_explicit_position_and_padding_semantics() -> None:
    adapter, _ = _adapter(pooling="mean")

    probabilities, embeddings, representatives = adapter.infer_sequence_to_sequence(["APGT"])

    assert probabilities.shape == (1, 4)
    np.testing.assert_allclose(representatives, embeddings.mean(axis=1), rtol=1e-6)
    np.testing.assert_array_equal(
        adapter.sequence_pos_to_prob_pos(["ACGT", "AC"], 3),
        np.array([3, -1]),
    )
    assert adapter.infer_masked_sequence_to_token_probs(["ACGT"], 1, ["G"], ["C"]) == (
        None,
        None,
    )


def test_adapter_rejects_ragged_or_too_long_batches() -> None:
    adapter, _ = _adapter()

    with pytest.raises(ValueError, match="equal-length"):
        adapter.infer_sequence_to_sequence(["AC", "ACG"])
    with pytest.raises(ValueError, match="exceeds adapter maximum"):
        adapter.infer_sequence_to_sequence(["A" * 16])


def test_stateful_adapter_requires_policy_and_resets_each_sequence_call() -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGTN")
    model = GenomicLanguageModel(
        titans_memory_tiny(
            tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            max_seq_len=4,
            expansion_factor=2,
            projection_kernel_size=2,
        )
    )
    with pytest.raises(ValueError, match="explicit inference policy"):
        AtcgGFMModel(model, tokenizer)

    adapter = AtcgGFMModel(
        model,
        tokenizer,
        stateful_policy=StatefulInferencePolicy(
            memory_mode="adaptive",
            max_sequence_length=8,
        ),
    )
    first = adapter.infer_sequence_to_sequence(["ACGTAC"])
    second = adapter.infer_sequence_to_sequence(["ACGTAC"])

    assert first[0].shape == (1, 6)
    assert first[1].shape == (1, 6, 8)
    np.testing.assert_allclose(first[0], second[0], rtol=1e-6)
    np.testing.assert_allclose(first[1], second[1], rtol=1e-6)
