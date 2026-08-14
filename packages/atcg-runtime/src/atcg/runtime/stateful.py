"""Ordered-stream training for explicit recurrent model state."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer

from atcg.models import GenomicLanguageModel, MemoryMode, ModelState
from atcg.runtime.batching import TARGET_IGNORE_ID, StatefulCausalBatch, collate_horizons
from atcg.runtime.training import (
    StepMetrics,
    TrainingConfig,
    TrainingRun,
    _gradient_norm,
    _manifest,
    _write_json,
    _write_metrics,
)
from atcg.runtime.training_state import TrainingState
from atcg.sequence import OrderedCausalStreamDataset


class StreamStateStore:
    """Runtime-owned state table keyed by stable biological stream identity."""

    def __init__(self, model: GenomicLanguageModel) -> None:
        self.model = model
        self._states: dict[str, ModelState] = {}

    @property
    def active_streams(self) -> tuple[str, ...]:
        return tuple(sorted(self._states))

    def state_for(self, stream_id: str) -> ModelState:
        try:
            return self._states[stream_id]
        except KeyError as error:
            raise KeyError(f"stream {stream_id!r} has no active state") from error

    def acquire(
        self,
        stream_ids: tuple[str, ...],
        stream_starts: tuple[bool, ...],
        *,
        device: torch.device | str,
    ) -> tuple[ModelState, ...]:
        if len(stream_ids) != len(stream_starts):
            raise ValueError("stream start flags must align with stream ids")
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("a stateful batch cannot acquire a stream twice")
        states: list[ModelState] = []
        for stream_id, starts in zip(stream_ids, stream_starts, strict=True):
            if starts:
                if stream_id in self._states:
                    raise RuntimeError(f"stream {stream_id!r} restarted before its prior end")
                state = self.model.initial_state()
            else:
                state = self._states.get(stream_id)
                if state is None:
                    raise RuntimeError(f"stream {stream_id!r} has no carried state")
            states.append(state.to(device))
        return tuple(states)

    def commit(
        self,
        stream_ids: tuple[str, ...],
        states: tuple[ModelState, ...],
        stream_ends: tuple[bool, ...],
    ) -> None:
        if not len(stream_ids) == len(states) == len(stream_ends):
            raise ValueError("committed state and stream metadata must align")
        for stream_id, state, ends in zip(stream_ids, states, stream_ends, strict=True):
            if ends:
                self._states.pop(stream_id, None)
            else:
                self._states[stream_id] = state.detach()

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "streams": {
                stream_id: state.state_dict() for stream_id, state in sorted(self._states.items())
            },
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("format_version") != 1:
            raise ValueError("unsupported stream-state format")
        raw_streams = payload.get("streams")
        if not isinstance(raw_streams, Mapping):
            raise TypeError("stream-state payload must contain a stream mapping")
        restored: dict[str, ModelState] = {}
        for stream_id, raw_state in raw_streams.items():
            if not isinstance(stream_id, str) or not isinstance(raw_state, Mapping):
                raise TypeError("stream-state entries must map strings to model states")
            restored[stream_id] = self.model.state_from_dict(cast(Mapping[str, object], raw_state))
        self._states = restored


class StatefulTrainer:
    """Optimizer wrapper for segment-ordered truncated backpropagation."""

    def __init__(
        self,
        model: GenomicLanguageModel,
        optimizer: Optimizer,
        *,
        memory_mode: MemoryMode = "adaptive",
        device: torch.device | str = "cpu",
        precision: str = "float32",
        gradient_clip_norm: float | None = 1.0,
        state: TrainingState | None = None,
        state_store: StreamStateStore | None = None,
    ) -> None:
        if precision not in {"float32", "bfloat16"}:
            raise ValueError(f"unsupported precision {precision!r}")
        self.model = model
        self.optimizer = optimizer
        self.memory_mode = memory_mode
        self.device = torch.device(device)
        self.precision = precision
        self.gradient_clip_norm = gradient_clip_norm
        self.state = state or TrainingState()
        self.state_store = state_store or StreamStateStore(model)
        self.model.to(self.device)

    def train_step(self, batch: StatefulCausalBatch) -> StepMetrics:
        started = time.perf_counter()
        local_batch = batch.to(self.device)
        carried = list(
            self.state_store.acquire(
                local_batch.stream_ids,
                local_batch.stream_starts,
                device=self.device,
            )
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.precision == "bfloat16"
            else nullcontext()
        )
        total_nll: Tensor | None = None
        token_count = 0
        with autocast_context:
            for segment_index in range(local_batch.input_ids.shape[1]):
                segment_mask = local_batch.valid_mask[:, segment_index]
                active_indices = segment_mask.any(dim=1).nonzero(as_tuple=False).flatten()
                if active_indices.numel() == 0:
                    continue
                indices = [int(value) for value in active_indices.tolist()]
                output = self.model.forward_segment(
                    local_batch.input_ids[active_indices, segment_index],
                    tuple(carried[index] for index in indices),
                    valid_mask=segment_mask[active_indices],
                    memory_mode=self.memory_mode,
                )
                targets = local_batch.target_ids[active_indices, segment_index]
                segment_nll = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=TARGET_IGNORE_ID,
                    reduction="sum",
                )
                total_nll = segment_nll if total_nll is None else total_nll + segment_nll
                token_count += int(targets.ne(TARGET_IGNORE_ID).sum().item())
                for local_index, batch_index in enumerate(indices):
                    carried[batch_index] = output.states[local_index]
        if total_nll is None or token_count == 0:
            raise ValueError("stateful batch contains no training targets")
        loss = total_nll / token_count
        loss.backward()
        if self.gradient_clip_norm is None:
            gradient_norm = _gradient_norm(self.model.parameters())
        else:
            gradient_norm = float(
                clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm).item()
            )
        self.optimizer.step()
        self.state_store.commit(
            local_batch.stream_ids,
            tuple(carried),
            local_batch.stream_ends,
        )
        self.state = TrainingState(
            step=self.state.step + 1,
            tokens_seen=self.state.tokens_seen + token_count,
        )
        return StepMetrics(
            step=self.state.step,
            loss=float(loss.detach().item()),
            gradient_norm=gradient_norm,
            tokens=token_count,
            tokens_seen=self.state.tokens_seen,
            elapsed_seconds=time.perf_counter() - started,
        )


def fit_stateful(
    model: GenomicLanguageModel,
    dataset: OrderedCausalStreamDataset,
    *,
    pad_id: int,
    config: TrainingConfig,
    memory_mode: MemoryMode = "adaptive",
    run_dir: str | Path | None = None,
) -> TrainingRun:
    """Train an explicit-state model over deterministic ordered stream batches."""

    if not dataset:
        raise ValueError("cannot train on an empty ordered-stream dataset")
    if config.shuffle:
        raise ValueError("stateful training requires TrainingConfig(shuffle=False)")
    torch.manual_seed(config.seed)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trainer = StatefulTrainer(
        model,
        optimizer,
        memory_mode=memory_mode,
        device=config.device,
        precision=config.precision,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    metrics: list[StepMetrics] = []
    while trainer.state.step < config.max_steps:
        produced_batch = False
        for horizons in dataset.iter_batches(config.batch_size):
            produced_batch = True
            batch = collate_horizons(
                horizons,
                pad_id=pad_id,
                segment_length=dataset.segment_length,
            )
            metrics.append(trainer.train_step(batch))
            if trainer.state.step >= config.max_steps:
                break
        if not produced_batch:
            raise ValueError("ordered-stream dataset did not produce a batch")

    checkpoint_path: Path | None = None
    if run_dir is not None:
        destination = Path(run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = _manifest(model, config, len(dataset))
        manifest["stateful_execution"] = {
            "memory_mode": memory_mode,
            "segment_length": dataset.segment_length,
            "gradient_horizon": dataset.gradient_horizon,
        }
        _write_json(destination / "manifest.json", manifest)
        _write_metrics(destination / "metrics.jsonl", metrics)
        from atcg.runtime.checkpoint import save_checkpoint

        checkpoint_path = save_checkpoint(
            destination / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            training_state=trainer.state,
            stream_state=trainer.state_store.state_dict(),
            metadata={"run_directory": str(destination), "memory_mode": memory_mode},
        )
    return TrainingRun(
        state=trainer.state,
        metrics=tuple(metrics),
        checkpoint_path=checkpoint_path,
    )
