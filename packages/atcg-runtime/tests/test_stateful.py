from pathlib import Path

import torch
from torch.optim import AdamW

from atcg.models import GenomicLanguageModel, titans_memory_tiny
from atcg.models.titans import NeuralMemoryState
from atcg.runtime import (
    StatefulTrainer,
    StreamStateStore,
    collate_horizons,
    load_checkpoint,
    save_checkpoint,
)
from atcg.sequence import FixedAlphabetTokenizer, OrderedCausalStreamDataset, SequenceRecord


def _components() -> tuple[
    GenomicLanguageModel,
    FixedAlphabetTokenizer,
    OrderedCausalStreamDataset,
]:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT")
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
    dataset = OrderedCausalStreamDataset(
        [
            SequenceRecord(identifier="a", sequence="ACGT" * 5),
            SequenceRecord(identifier="b", sequence="TGCA" * 5),
        ],
        tokenizer,
        segment_length=4,
        gradient_horizon=2,
    )
    return model, tokenizer, dataset


def test_stateful_trainer_carries_detached_state_by_stream() -> None:
    torch.manual_seed(53)
    model, tokenizer, dataset = _components()
    trainer = StatefulTrainer(model, AdamW(model.parameters(), lr=1e-3))
    batches = dataset.iter_batches(batch_size=2)

    first = collate_horizons(next(batches), pad_id=tokenizer.pad_id, segment_length=4)
    metrics = trainer.train_step(first)

    assert metrics.tokens == first.token_count
    assert trainer.state_store.active_streams == ("a", "b")
    for stream_id in trainer.state_store.active_streams:
        state = trainer.state_store.state_for(stream_id)
        for block_state in state.blocks:
            if isinstance(block_state, NeuralMemoryState):
                assert all(tensor.grad_fn is None for tensor in block_state.fast_weights.values())

    second = collate_horizons(next(batches), pad_id=tokenizer.pad_id, segment_length=4)
    trainer.train_step(second)
    assert trainer.state.step == 2


def test_stream_state_round_trips_through_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(59)
    model, tokenizer, dataset = _components()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    trainer = StatefulTrainer(model, optimizer)
    horizons = next(dataset.iter_batches(batch_size=2))
    trainer.train_step(collate_horizons(horizons, pad_id=tokenizer.pad_id, segment_length=4))

    path = save_checkpoint(
        tmp_path / "stateful.pt",
        model=model,
        optimizer=optimizer,
        training_state=trainer.state,
        stream_state=trainer.state_store.state_dict(),
    )
    restored_model, _, _ = _components()
    loaded = load_checkpoint(path, model=restored_model)
    assert loaded.stream_state is not None
    restored_store = StreamStateStore(restored_model)
    restored_store.load_state_dict(loaded.stream_state)

    assert restored_store.active_streams == trainer.state_store.active_streams
    assert restored_store.state_dict()["format_version"] == 1
