"""Versioned, atomic training checkpoints."""

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from atcg.models import GenomicLanguageModel, ModelConfig
from atcg.runtime.training_state import TrainingState
from torch.optim import Optimizer

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated non-tensor state restored from a checkpoint."""

    model_config: ModelConfig
    training_state: TrainingState
    metadata: Mapping[str, str]


def save_checkpoint(
    path: str | Path,
    *,
    model: GenomicLanguageModel,
    optimizer: Optimizer | None = None,
    training_state: TrainingState | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Atomically save trusted training state using PyTorch's weights-only types."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "training_state": asdict(training_state or TrainingState()),
        "torch_rng_state": torch.get_rng_state(),
        "metadata": dict(metadata or {}),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: GenomicLanguageModel | None = None,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> LoadedCheckpoint:
    """Load a checkpoint, optionally restoring model, optimizer, and CPU RNG state."""

    raw_payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(raw_payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    payload = cast(dict[str, object], raw_payload)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {payload.get('schema_version')!r}")

    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint does not contain a model configuration")
    model_config = ModelConfig.from_dict(cast(dict[str, object], raw_config))
    if model is not None:
        if model.config != model_config:
            raise ValueError("checkpoint model configuration does not match the target model")
        model_state = payload.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("checkpoint does not contain model weights")
        model.load_state_dict(cast(dict[str, torch.Tensor], model_state), strict=True)

    optimizer_state = payload.get("optimizer_state")
    if optimizer is not None:
        if not isinstance(optimizer_state, dict):
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(cast(dict[str, object], optimizer_state))

    raw_training_state = payload.get("training_state")
    if not isinstance(raw_training_state, dict):
        raise ValueError("checkpoint does not contain training state")
    state_values = cast(dict[str, object], raw_training_state)
    step = state_values.get("step")
    tokens_seen = state_values.get("tokens_seen")
    if not isinstance(step, int) or not isinstance(tokens_seen, int):
        raise ValueError("checkpoint training state is invalid")
    training_state = TrainingState(step=step, tokens_seen=tokens_seen)

    rng_state = payload.get("torch_rng_state")
    if restore_rng:
        if not isinstance(rng_state, torch.Tensor):
            raise ValueError("checkpoint CPU RNG state is invalid")
        torch.set_rng_state(rng_state.cpu())

    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata must contain string keys and values")
    metadata: dict[str, str] = {}
    for key, value in cast(dict[object, object], raw_metadata).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("checkpoint metadata must contain string keys and values")
        metadata[key] = value

    return LoadedCheckpoint(
        model_config=model_config,
        training_state=training_state,
        metadata=metadata,
    )
