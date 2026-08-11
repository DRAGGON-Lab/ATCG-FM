from pathlib import Path

from atcg.eval import evaluate_language_model
from atcg.models import GenomicLanguageModel, hybrid_tiny
from atcg.runtime import TrainingConfig, fit, generate, load_model_checkpoint, score_sequence
from atcg.sequence import CausalWindowDataset, FixedAlphabetTokenizer, SequenceRecord


def test_hybrid_training_to_inference_vertical_slice(tmp_path: Path) -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT")
    dataset = CausalWindowDataset(
        [SequenceRecord(identifier="synthetic", sequence="ACGT" * 10)],
        tokenizer,
        context_length=8,
        stride=8,
        include_eos=True,
        include_reverse_complements=True,
    )
    model = GenomicLanguageModel(
        hybrid_tiny(
            tokenizer.vocab_size,
            max_seq_len=8,
            d_model=16,
            n_layers=4,
            n_heads=4,
        )
    )

    run = fit(
        model,
        dataset,
        pad_id=tokenizer.pad_id,
        config=TrainingConfig(max_steps=2, batch_size=2, seed=59),
        run_dir=tmp_path / "run",
    )
    assert run.checkpoint_path is not None
    restored, loaded = load_model_checkpoint(run.checkpoint_path)
    metrics = evaluate_language_model(restored, dataset, pad_id=tokenizer.pad_id)
    score = score_sequence(restored, tokenizer, "ACGT")
    generated = generate(
        restored,
        tokenizer,
        "AC",
        max_new_tokens=3,
        temperature=0.0,
        allowed_token_ids=tuple(range(len(tokenizer.alphabet))),
    )

    assert loaded.training_state == run.state
    assert metrics.token_count == sum(len(example.target_ids) for example in dataset)
    assert score.token_count == 5
    assert len(generated.generated_token_ids) == 3
    assert set(generated.sequence) <= set(tokenizer.alphabet)
