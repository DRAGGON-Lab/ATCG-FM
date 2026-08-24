from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from atcg.models import GenomicLanguageModel, attention_tiny
from atcg.runtime import TrainingState, load_checkpoint, save_checkpoint


def _model() -> GenomicLanguageModel:
    return GenomicLanguageModel(
        attention_tiny(vocab_size=9, d_model=16, n_heads=4, n_layers=1, max_seq_len=8)
    )


def test_checkpoint_round_trip_restores_exact_logits_and_state(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = _model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, model.config.vocab_size, (2, 6))
    original = model(inputs).logits.detach()
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        training_state=TrainingState(step=3, tokens_seen=72),
        metadata={"study": "round-trip"},
    )

    restored_model = _model()
    restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
    loaded = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    torch.testing.assert_close(restored_model(inputs).logits, original)
    assert loaded.training_state == TrainingState(step=3, tokens_seen=72)
    assert loaded.metadata == {"study": "round-trip"}
    assert loaded.model_interface == "explicit_block_state_v1"
    assert loaded.training_sequence_format == "ordered_segment_causal_v1"
    assert loaded.stream_state is None


def test_checkpoint_rejects_a_different_model_configuration(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=_model())
    incompatible = GenomicLanguageModel(
        attention_tiny(vocab_size=10, d_model=16, n_heads=4, n_layers=1, max_seq_len=8)
    )

    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, model=incompatible)


def test_checkpoint_rejects_an_older_model_schema(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=_model())
    payload = torch.load(path, weights_only=True)
    payload["schema_version"] = 2
    torch.save(payload, path)

    with pytest.raises(ValueError, match="unsupported checkpoint schema 2"):
        load_checkpoint(path)


def test_checkpoint_round_trips_experiment_resume_state(tmp_path: Path) -> None:
    path = save_checkpoint(
        tmp_path / "resume.pt",
        model=_model(),
        grad_scaler_state={"scale": 1024.0},
        experiment_state={"dataset_fingerprint": "abc", "global_batch_index": 7},
    )

    loaded = load_checkpoint(path)

    assert loaded.grad_scaler_state == {"scale": 1024.0}
    assert loaded.experiment_state == {
        "dataset_fingerprint": "abc",
        "global_batch_index": 7,
    }
