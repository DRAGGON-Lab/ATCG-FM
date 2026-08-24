"""Versioned, atomic training checkpoints."""

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch.optim import Optimizer

from atcg.models import GenomicLanguageModel, ModelConfig
from atcg.runtime.training_state import TrainingState

CHECKPOINT_SCHEMA_VERSION = 4
READABLE_CHECKPOINT_SCHEMA_VERSIONS = frozenset({3, 4})
MODEL_INTERFACE = "explicit_block_state_v1"
TRAINING_SEQUENCE_FORMAT = "ordered_segment_causal_v1"


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated non-tensor state restored from a checkpoint."""

    model_config: ModelConfig
    training_state: TrainingState
    metadata: Mapping[str, str]
    model_interface: str
    training_sequence_format: str
    stream_state: Mapping[str, object] | None
    grad_scaler_state: Mapping[str, object] | None
    experiment_state: Mapping[str, object] | None


def save_checkpoint(
    path: str | Path,
    *,
    model: GenomicLanguageModel,
    optimizer: Optimizer | None = None,
    training_state: TrainingState | None = None,
    stream_state: Mapping[str, object] | None = None,
    grad_scaler_state: Mapping[str, object] | None = None,
    experiment_state: Mapping[str, object] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Atomically save trusted training state using PyTorch's weights-only types."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_interface": MODEL_INTERFACE,
        "training_sequence_format": TRAINING_SEQUENCE_FORMAT,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "training_state": asdict(training_state or TrainingState()),
        "stream_state": dict(stream_state) if stream_state is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "grad_scaler_state": dict(grad_scaler_state) if grad_scaler_state is not None else None,
        "experiment_state": dict(experiment_state) if experiment_state is not None else None,
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
    """Load a checkpoint, optionally restoring model, optimizer, and available RNG state."""

    raw_payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(raw_payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    payload = cast(dict[str, object], raw_payload)
    if payload.get("schema_version") not in READABLE_CHECKPOINT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported checkpoint schema {payload.get('schema_version')!r}")
    if payload.get("model_interface") != MODEL_INTERFACE:
        raise ValueError(f"unsupported model interface {payload.get('model_interface')!r}")
    if payload.get("training_sequence_format") != TRAINING_SEQUENCE_FORMAT:
        raise ValueError(
            f"unsupported training sequence format {payload.get('training_sequence_format')!r}"
        )

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
        cuda_rng_state = payload.get("cuda_rng_state")
        if torch.cuda.is_available() and cuda_rng_state is not None:
            if not isinstance(cuda_rng_state, list) or not all(
                isinstance(value, torch.Tensor) for value in cuda_rng_state
            ):
                raise ValueError("checkpoint CUDA RNG state is invalid")
            torch.cuda.set_rng_state_all(cuda_rng_state)

    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata must contain string keys and values")
    metadata: dict[str, str] = {}
    for key, value in cast(dict[object, object], raw_metadata).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("checkpoint metadata must contain string keys and values")
        metadata[key] = value

    raw_stream_state = payload.get("stream_state")
    stream_state: Mapping[str, object] | None
    if raw_stream_state is None:
        stream_state = None
    elif isinstance(raw_stream_state, dict):
        stream_state = cast(dict[str, object], raw_stream_state)
    else:
        raise ValueError("checkpoint stream state must be a mapping or null")

    grad_scaler_state = _optional_mapping(payload, "grad_scaler_state")
    experiment_state = _optional_mapping(payload, "experiment_state")

    return LoadedCheckpoint(
        model_config=model_config,
        training_state=training_state,
        metadata=metadata,
        model_interface=MODEL_INTERFACE,
        training_sequence_format=TRAINING_SEQUENCE_FORMAT,
        stream_state=stream_state,
        grad_scaler_state=grad_scaler_state,
        experiment_state=experiment_state,
    )


def _optional_mapping(payload: Mapping[str, object], name: str) -> Mapping[str, object] | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"checkpoint {name} must be a string-keyed mapping or null")
    return cast(dict[str, object], value)


def load_model_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[GenomicLanguageModel, LoadedCheckpoint]:
    """Instantiate and strictly restore a model from its recorded configuration."""

    loaded = load_checkpoint(path, map_location="cpu")
    model = GenomicLanguageModel(loaded.model_config)
    loaded = load_checkpoint(path, model=model, map_location="cpu")
    model.to(device)
    model.eval()
    return model, loaded
