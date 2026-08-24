import pytest
import torch

from atcg.models import GenomicLanguageModel, attention_tiny, titans_memory_tiny
from atcg.runtime import (
    OrderedComparisonConfig,
    OrderedComparisonTrainer,
    ProfileMeasurement,
    ordered_dataset_fingerprint,
    select_largest_fitting_profile,
    train_ordered_epoch,
    validate_ordered_model,
)
from atcg.sequence import FixedAlphabetTokenizer, OrderedCausalStreamDataset, SequenceRecord


def _dataset() -> tuple[FixedAlphabetTokenizer, OrderedCausalStreamDataset]:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT")
    dataset = OrderedCausalStreamDataset(
        [
            SequenceRecord(identifier="a", sequence="ACGT" * 3),
            SequenceRecord(identifier="b", sequence="TGCA" * 3),
        ],
        tokenizer,
        segment_length=4,
        gradient_horizon=1,
    )
    return tokenizer, dataset


@pytest.mark.parametrize("architecture", ["attention", "titans"])
def test_ordered_trainer_consumes_one_identical_epoch(architecture: str) -> None:
    tokenizer, dataset = _dataset()
    config = (
        attention_tiny(
            tokenizer.vocab_size,
            d_model=8,
            n_heads=2,
            n_layers=1,
            max_seq_len=4,
        )
        if architecture == "attention"
        else titans_memory_tiny(
            tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            max_seq_len=4,
            expansion_factor=1,
            projection_kernel_size=2,
        )
    )
    torch.manual_seed(17)
    model = GenomicLanguageModel(config)
    trainer = OrderedComparisonTrainer(
        model,
        pad_id=tokenizer.pad_id,
        segment_length=4,
        config=OrderedComparisonConfig(
            global_batch_size=2,
            microbatch_size=1,
            precision="float32",
            device="cpu",
        ),
    )

    metrics = train_ordered_epoch(trainer, dataset)

    assert sum(row.tokens for row in metrics) == sum(
        len(segment.target_ids) for horizon in dataset for segment in horizon.segments
    )
    assert trainer.state_store.active_streams == ()


def test_ordered_validation_caps_tokens_and_resets_streams() -> None:
    tokenizer, dataset = _dataset()
    model = GenomicLanguageModel(
        attention_tiny(
            tokenizer.vocab_size,
            d_model=8,
            n_heads=2,
            n_layers=1,
            max_seq_len=4,
        )
    )

    metrics = validate_ordered_model(
        model,
        dataset,
        pad_id=tokenizer.pad_id,
        batch_size=2,
        max_tokens=7,
    )

    assert metrics.token_count == 7
    assert metrics.memory_mode is None
    assert ordered_dataset_fingerprint(dataset) == ordered_dataset_fingerprint(dataset)


def test_ordered_validation_reports_stream_offset_bins() -> None:
    tokenizer, dataset = _dataset()
    model = GenomicLanguageModel(
        attention_tiny(
            tokenizer.vocab_size,
            d_model=8,
            n_heads=2,
            n_layers=1,
            max_seq_len=4,
        )
    )

    metrics = validate_ordered_model(
        model,
        dataset,
        pad_id=tokenizer.pad_id,
        batch_size=2,
        offset_boundaries=(4, 8, 16),
    )

    assert metrics.offset_bins is not None
    assert list(metrics.offset_bins) == ["1-4", "5-8", "9-16"]
    assert sum(row.token_count for row in metrics.offset_bins.values()) == metrics.token_count


def test_ordered_validation_can_reset_state_each_segment() -> None:
    tokenizer, dataset = _dataset()
    model = GenomicLanguageModel(
        titans_memory_tiny(
            tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            max_seq_len=4,
            expansion_factor=1,
            projection_kernel_size=2,
        )
    )

    metrics = validate_ordered_model(
        model,
        dataset,
        pad_id=tokenizer.pad_id,
        batch_size=2,
        reset_state_each_segment=True,
    )

    assert metrics.token_count > 0


def test_profile_selection_uses_capacity_after_budget_filters() -> None:
    selected = select_largest_fitting_profile(
        [
            ProfileMeasurement("small", 100.0, 1_000, 64, 2),
            ProfileMeasurement("medium", 200.0, 2_000, 96, 3),
            ProfileMeasurement("too-slow", 301.0, 2_000, 128, 4),
        ],
        maximum_seconds=300.0,
        maximum_memory_bytes=3_000,
    )

    assert selected.profile_id == "medium"
