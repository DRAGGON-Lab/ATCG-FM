import math

import torch
from atcg.eval import evaluate_language_model
from atcg.models import GenomicLanguageModel, attention_tiny
from atcg.sequence import CausalWindowDataset, FixedAlphabetTokenizer, SequenceRecord


def test_language_model_evaluation_counts_only_real_targets() -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT")
    dataset = CausalWindowDataset(
        [SequenceRecord(identifier="fixture", sequence="ACGTAC")],
        tokenizer,
        context_length=4,
        stride=4,
        include_eos=True,
    )
    torch.manual_seed(53)
    model = GenomicLanguageModel(
        attention_tiny(
            tokenizer.vocab_size,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_seq_len=4,
        )
    )

    metrics = evaluate_language_model(model, dataset, pad_id=tokenizer.pad_id, batch_size=2)

    assert metrics.example_count == 2
    assert metrics.token_count == sum(len(example.target_ids) for example in dataset)
    assert math.isfinite(metrics.total_nll)
    assert metrics.perplexity == math.exp(metrics.mean_nll)
    assert 0.0 <= metrics.token_accuracy <= 1.0
