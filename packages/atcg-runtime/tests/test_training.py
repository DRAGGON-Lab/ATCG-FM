from pathlib import Path

import torch
from torch.optim import AdamW

from atcg.models import GenomicLanguageModel, attention_tiny
from atcg.runtime import CausalBatch, Trainer, TrainingConfig, fit
from atcg.sequence import CausalWindowDataset, FixedAlphabetTokenizer, SequenceRecord


def _training_fixture() -> tuple[
    GenomicLanguageModel,
    FixedAlphabetTokenizer,
    CausalWindowDataset,
]:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT")
    dataset = CausalWindowDataset(
        [SequenceRecord(identifier="repeat", sequence="ACGT" * 16)],
        tokenizer,
        context_length=8,
        stride=4,
        include_eos=False,
        drop_incomplete=True,
    )
    torch.manual_seed(43)
    model = GenomicLanguageModel(
        attention_tiny(
            tokenizer.vocab_size,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_seq_len=8,
        )
    )
    return model, tokenizer, dataset


def test_trainer_can_overfit_a_repeated_batch() -> None:
    model, _, dataset = _training_fixture()
    example = dataset[0]
    batch = CausalBatch(
        input_ids=torch.tensor([example.input_ids]),
        target_ids=torch.tensor([example.target_ids]),
        lengths=torch.tensor([len(example.input_ids)]),
    )
    trainer = Trainer(model, AdamW(model.parameters(), lr=3e-2), gradient_clip_norm=10.0)

    losses = [trainer.train_step(batch).loss for _ in range(25)]

    assert losses[-1] < losses[0] * 0.25
    assert trainer.state.tokens_seen == 25 * len(example.input_ids)


def test_fit_writes_a_reproducible_run_bundle(tmp_path: Path) -> None:
    model, tokenizer, dataset = _training_fixture()

    run = fit(
        model,
        dataset,
        pad_id=tokenizer.pad_id,
        config=TrainingConfig(max_steps=2, batch_size=2, seed=47),
        run_dir=tmp_path / "run",
    )

    assert run.state.step == 2
    assert run.checkpoint_path is not None and run.checkpoint_path.is_file()
    assert (tmp_path / "run" / "manifest.json").is_file()
    assert len((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()) == 2
