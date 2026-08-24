"""Ordered, microbatched training and evaluation for mixer comparisons."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer

from atcg.models import GenomicLanguageModel, MemoryMode
from atcg.runtime.batching import TARGET_IGNORE_ID, collate_horizons
from atcg.runtime.checkpoint import load_checkpoint
from atcg.runtime.stateful import StreamStateStore
from atcg.runtime.training import _gradient_norm
from atcg.runtime.training_state import TrainingState
from atcg.sequence import LanguageModelHorizon, OrderedCausalStreamDataset, SequenceRecord

type ComparisonPrecision = Literal["float32", "float16"]


@dataclass(frozen=True, slots=True)
class OrderedComparisonConfig:
    """Shared optimizer-step protocol with a candidate-specific microbatch."""

    global_batch_size: int = 32
    microbatch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip_norm: float | None = 1.0
    precision: ComparisonPrecision = "float16"
    device: str = "cuda"
    seed: int = 17

    def __post_init__(self) -> None:
        if self.global_batch_size < 1 or self.microbatch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.global_batch_size % self.microbatch_size:
            raise ValueError("microbatch size must divide global batch size")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings are invalid")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive or null")
        if self.precision == "float16" and not self.device.startswith("cuda"):
            raise ValueError("float16 comparison training requires CUDA")


@dataclass(frozen=True, slots=True)
class OrderedStepMetrics:
    step: int
    loss: float
    gradient_norm: float
    tokens: int
    tokens_seen: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class OrderedValidationMetrics:
    total_nll: float
    mean_nll: float
    bits_per_token: float
    perplexity: float
    token_accuracy: float
    token_count: int
    segment_count: int
    memory_mode: MemoryMode | None
    offset_bins: Mapping[str, OffsetValidationMetrics] | None = None


@dataclass(frozen=True, slots=True)
class OffsetValidationMetrics:
    """Quality metrics for target positions within one ordered-stream offset range."""

    start: int
    stop: int
    mean_nll: float
    bits_per_token: float
    perplexity: float
    token_accuracy: float
    token_count: int
    segment_count: int


@dataclass(frozen=True, slots=True)
class RepeatContextMetrics:
    context_length: int
    repeated_mean_nll: float
    repeated_bits_per_token: float
    repeated_tokens: int
    novel_mean_nll: float
    novel_bits_per_token: float
    novel_tokens: int


@dataclass(frozen=True, slots=True)
class ProfileMeasurement:
    profile_id: str
    projected_seconds: float
    peak_memory_bytes: int
    d_model: int
    n_layers: int


@dataclass(frozen=True, slots=True)
class MicrobatchMeasurement:
    microbatch_size: int
    tokens_per_second: float
    peak_memory_bytes: int


class CudaUtilizationSampler:
    """Periodically sample CUDA utilization without synchronizing the training loop."""

    def __init__(self, *, interval_seconds: float = 5.0, device: int = 0) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self.device = device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[int] = []

    def __enter__(self) -> CudaUtilizationSampler:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA utilization sampling requires CUDA")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(int(torch.cuda.utilization(self.device)))
            except (AttributeError, ImportError, OSError):
                return
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, float | int | None]:
        if not self._samples:
            return {"samples": 0, "mean_percent": None, "maximum_percent": None}
        return {
            "samples": len(self._samples),
            "mean_percent": sum(self._samples) / len(self._samples),
            "maximum_percent": max(self._samples),
        }


def select_largest_fitting_profile(
    measurements: Sequence[ProfileMeasurement],
    *,
    maximum_seconds: float = 9_000.0,
    maximum_memory_bytes: int = 13 * 1024**3,
) -> ProfileMeasurement:
    """Select the largest measured profile satisfying fixed time and memory bounds."""

    fitting = [
        row
        for row in measurements
        if row.projected_seconds <= maximum_seconds
        and row.peak_memory_bytes <= maximum_memory_bytes
    ]
    if not fitting:
        raise RuntimeError("no measured model profile fits the declared T4 budget")
    return max(fitting, key=lambda row: (row.d_model * row.d_model * row.n_layers, row.profile_id))


def benchmark_ordered_candidate(
    model_factory: Callable[[], GenomicLanguageModel],
    dataset: OrderedCausalStreamDataset,
    *,
    pad_id: int,
    base_config: OrderedComparisonConfig,
    microbatch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32),
    timed_global_batches: int = 2,
) -> tuple[MicrobatchMeasurement, ...]:
    """Measure safe microbatches using one warm-up and a bounded ordered sample."""

    if not base_config.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("candidate benchmarking requires a CUDA device")
    if timed_global_batches < 1:
        raise ValueError("timed global batches must be positive")
    results: list[MicrobatchMeasurement] = []
    for microbatch_size in microbatch_sizes:
        if base_config.global_batch_size % microbatch_size:
            continue
        config = OrderedComparisonConfig(
            global_batch_size=base_config.global_batch_size,
            microbatch_size=microbatch_size,
            learning_rate=base_config.learning_rate,
            weight_decay=base_config.weight_decay,
            gradient_clip_norm=base_config.gradient_clip_norm,
            precision=base_config.precision,
            device=base_config.device,
            seed=base_config.seed,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trainer: OrderedComparisonTrainer | None = None
        try:
            trainer = OrderedComparisonTrainer(
                model_factory(),
                pad_id=pad_id,
                segment_length=dataset.segment_length,
                config=config,
            )
            batches = dataset.iter_batches(config.global_batch_size)
            trainer.train_global_batch(next(batches))
            torch.cuda.synchronize()
            started = time.perf_counter()
            tokens = 0
            for _ in range(timed_global_batches):
                try:
                    metrics = trainer.train_global_batch(next(batches))
                except StopIteration:
                    break
                tokens += metrics.tokens
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if tokens:
                results.append(
                    MicrobatchMeasurement(
                        microbatch_size=microbatch_size,
                        tokens_per_second=tokens / elapsed,
                        peak_memory_bytes=torch.cuda.max_memory_allocated(),
                    )
                )
        except torch.OutOfMemoryError:
            pass
        finally:
            del trainer
            torch.cuda.empty_cache()
    if not results:
        raise RuntimeError("no candidate microbatch completed the T4 benchmark")
    return tuple(results)


class OrderedComparisonTrainer:
    """Train one candidate over identical ordered global batches."""

    def __init__(
        self,
        model: GenomicLanguageModel,
        *,
        pad_id: int,
        segment_length: int,
        config: OrderedComparisonConfig,
        memory_mode: MemoryMode = "adaptive",
        optimizer: Optimizer | None = None,
        state: TrainingState | None = None,
        state_store: StreamStateStore | None = None,
    ) -> None:
        self.model = model
        self.pad_id = pad_id
        self.segment_length = segment_length
        self.config = config
        self.memory_mode = memory_mode
        self.device = torch.device(config.device)
        self.optimizer = optimizer or AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.state = state or TrainingState()
        self.state_store = state_store or StreamStateStore(model)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "float16")
        self.model.to(self.device)

    def train_global_batch(self, horizons: Sequence[LanguageModelHorizon]) -> OrderedStepMetrics:
        if not horizons:
            raise ValueError("global batch must not be empty")
        if len(horizons) > self.config.global_batch_size:
            raise ValueError("batch exceeds configured global batch size")
        started = time.perf_counter()
        token_count = sum(
            len(segment.target_ids) for horizon in horizons for segment in horizon.segments
        )
        if token_count == 0:
            raise ValueError("global batch contains no targets")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_nll_value = 0.0

        for start in range(0, len(horizons), self.config.microbatch_size):
            values = horizons[start : start + self.config.microbatch_size]
            batch = collate_horizons(
                values,
                pad_id=self.pad_id,
                segment_length=self.segment_length,
            ).to(self.device)
            carried = list(
                self.state_store.acquire(
                    batch.stream_ids,
                    batch.stream_starts,
                    device=self.device,
                )
            )
            local_nll: Tensor | None = None
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.config.precision == "float16",
            ):
                for segment_index in range(batch.input_ids.shape[1]):
                    segment_mask = batch.valid_mask[:, segment_index]
                    active = segment_mask.any(dim=1).nonzero(as_tuple=False).flatten()
                    if active.numel() == 0:
                        continue
                    indices = [int(value) for value in active.tolist()]
                    output = self.model.forward_segment(
                        batch.input_ids[active, segment_index],
                        tuple(carried[index] for index in indices),
                        valid_mask=segment_mask[active],
                        memory_mode=self.memory_mode,
                    )
                    targets = batch.target_ids[active, segment_index]
                    segment_nll = F.cross_entropy(
                        output.logits.reshape(-1, output.logits.shape[-1]),
                        targets.reshape(-1),
                        ignore_index=TARGET_IGNORE_ID,
                        reduction="sum",
                    )
                    local_nll = segment_nll if local_nll is None else local_nll + segment_nll
                    for local_index, batch_index in enumerate(indices):
                        carried[batch_index] = output.states[local_index]
            if local_nll is None:
                raise ValueError("microbatch contains no targets")
            total_nll_value += float(local_nll.detach().item())
            self.scaler.scale(local_nll / token_count).backward()
            self.state_store.commit(batch.stream_ids, tuple(carried), batch.stream_ends)

        self.scaler.unscale_(self.optimizer)
        if self.config.gradient_clip_norm is None:
            gradient_norm = _gradient_norm(self.model.parameters())
        else:
            gradient_norm = float(
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm).item()
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.state = TrainingState(
            step=self.state.step + 1,
            tokens_seen=self.state.tokens_seen + token_count,
        )
        return OrderedStepMetrics(
            step=self.state.step,
            loss=total_nll_value / token_count,
            gradient_norm=gradient_norm,
            tokens=token_count,
            tokens_seen=self.state.tokens_seen,
            elapsed_seconds=time.perf_counter() - started,
        )


def train_ordered_epoch(
    trainer: OrderedComparisonTrainer,
    dataset: OrderedCausalStreamDataset,
) -> tuple[OrderedStepMetrics, ...]:
    """Consume an ordered dataset exactly once."""

    metrics = [
        trainer.train_global_batch(horizons)
        for horizons in dataset.iter_batches(trainer.config.global_batch_size)
    ]
    if not metrics:
        raise ValueError("ordered dataset produced no batches")
    return tuple(metrics)


def restore_ordered_trainer(
    checkpoint_path: str,
    trainer: OrderedComparisonTrainer,
    *,
    dataset_fingerprint: str,
) -> int:
    """Restore an exact ordered experiment and return its next global-batch index."""

    loaded = load_checkpoint(
        checkpoint_path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        map_location="cpu",
        restore_rng=True,
    )
    experiment = loaded.experiment_state
    if experiment is None:
        raise ValueError("checkpoint does not contain ordered experiment state")
    if experiment.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("checkpoint dataset fingerprint does not match the requested experiment")
    batch_index = experiment.get("global_batch_index")
    if not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("checkpoint global batch index is invalid")
    trainer.state = loaded.training_state
    if loaded.stream_state is not None:
        trainer.state_store.load_state_dict(loaded.stream_state)
    if loaded.grad_scaler_state is not None:
        trainer.scaler.load_state_dict(dict(loaded.grad_scaler_state))
    trainer.model.to(trainer.device)
    return batch_index


def validate_ordered_model(
    model: GenomicLanguageModel,
    dataset: OrderedCausalStreamDataset,
    *,
    pad_id: int,
    batch_size: int,
    device: str = "cpu",
    memory_mode: MemoryMode = "adaptive",
    max_tokens: int | None = None,
    offset_boundaries: Sequence[int] | None = None,
    reset_state_each_segment: bool = False,
) -> OrderedValidationMetrics:
    """Evaluate ordered streams with fresh per-stream state and an optional token cap."""

    if batch_size < 1 or (max_tokens is not None and max_tokens < 1):
        raise ValueError("validation batch and token cap must be positive")
    boundaries = tuple(offset_boundaries or ())
    invalid_boundaries = any(value < 1 for value in boundaries)
    invalid_order = tuple(sorted(set(boundaries))) != boundaries
    if boundaries and (invalid_boundaries or invalid_order):
        raise ValueError("offset boundaries must be strictly increasing positive integers")
    offset_totals = [{"nll": 0.0, "correct": 0, "tokens": 0, "segments": 0} for _ in boundaries]
    model.to(device)
    model.eval()
    store = StreamStateStore(model)
    total_nll = 0.0
    correct = 0
    token_count = 0
    segment_count = 0
    stop = False
    for horizons in dataset.iter_batches(batch_size):
        batch = collate_horizons(
            horizons,
            pad_id=pad_id,
            segment_length=dataset.segment_length,
        ).to(device)
        carried = list(store.acquire(batch.stream_ids, batch.stream_starts, device=device))
        for segment_index in range(batch.input_ids.shape[1]):
            mask = batch.valid_mask[:, segment_index]
            active = mask.any(dim=1).nonzero(as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            indices = [int(value) for value in active.tolist()]
            if reset_state_each_segment:
                for batch_index in indices:
                    carried[batch_index] = model.initial_state().to(device)
            context = (
                torch.enable_grad()
                if model.config.is_stateful and memory_mode == "adaptive"
                else torch.inference_mode()
            )
            with context:
                output = model.forward_segment(
                    batch.input_ids[active, segment_index],
                    tuple(carried[index] for index in indices),
                    valid_mask=mask[active],
                    memory_mode=memory_mode,
                )
            targets = batch.target_ids[active, segment_index]
            logits = output.logits.detach()
            valid = targets != TARGET_IGNORE_ID
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=TARGET_IGNORE_ID,
                reduction="none",
            ).reshape_as(targets)
            if max_tokens is not None:
                remaining = max_tokens - token_count
                if int(valid.sum().item()) > remaining:
                    flat_valid = valid.flatten()
                    positions = flat_valid.nonzero(as_tuple=False).flatten()
                    flat_valid[positions[remaining:]] = False
                    valid = flat_valid.reshape_as(valid)
                    stop = True
            total_nll += float(losses[valid].sum().item())
            predictions = logits.argmax(-1)
            correct += int((predictions[valid] == targets[valid]).sum().item())
            token_count += int(valid.sum().item())
            segment_count += 1
            if boundaries:
                for local_index, row_index in enumerate(indices):
                    example = horizons[row_index].segments[segment_index]
                    offsets = torch.arange(
                        example.token_offset + 1,
                        example.token_offset + 1 + targets.shape[1],
                        device=targets.device,
                    )
                    lower = 1
                    for bin_index, upper in enumerate(boundaries):
                        selected = valid[local_index] & (offsets >= lower) & (offsets <= upper)
                        selected_count = int(selected.sum().item())
                        if selected_count:
                            offset_totals[bin_index]["nll"] += float(
                                losses[local_index][selected].sum().item()
                            )
                            selected_predictions = predictions[local_index][selected]
                            selected_targets = targets[local_index][selected]
                            offset_totals[bin_index]["correct"] += int(
                                (selected_predictions == selected_targets).sum().item()
                            )
                            offset_totals[bin_index]["tokens"] += selected_count
                            offset_totals[bin_index]["segments"] += 1
                        lower = upper + 1
            for local_index, batch_index in enumerate(indices):
                carried[batch_index] = output.states[local_index]
            if stop:
                break
        store.commit(batch.stream_ids, tuple(carried), batch.stream_ends)
        if stop:
            break
    if token_count == 0:
        raise ValueError("ordered validation produced no tokens")
    mean_nll = total_nll / token_count
    offset_metrics: dict[str, OffsetValidationMetrics] | None = None
    if boundaries:
        offset_metrics = {}
        lower = 1
        for upper, values in zip(boundaries, offset_totals, strict=True):
            count = int(values["tokens"])
            if count:
                bin_nll = float(values["nll"]) / count
                offset_metrics[f"{lower}-{upper}"] = OffsetValidationMetrics(
                    start=lower,
                    stop=upper,
                    mean_nll=bin_nll,
                    bits_per_token=bin_nll / math.log(2.0),
                    perplexity=math.exp(bin_nll),
                    token_accuracy=int(values["correct"]) / count,
                    token_count=count,
                    segment_count=int(values["segments"]),
                )
            lower = upper + 1
    return OrderedValidationMetrics(
        total_nll=total_nll,
        mean_nll=mean_nll,
        bits_per_token=mean_nll / math.log(2.0),
        perplexity=math.exp(mean_nll),
        token_accuracy=correct / token_count,
        token_count=token_count,
        segment_count=segment_count,
        memory_mode=memory_mode if model.config.is_stateful else None,
        offset_bins=offset_metrics,
    )


def validate_repeat_contexts(
    model: GenomicLanguageModel,
    dataset: OrderedCausalStreamDataset,
    *,
    source_records: Sequence[SequenceRecord],
    training_records: Sequence[SequenceRecord],
    context_length: int = 15,
    device: str = "cpu",
    memory_mode: MemoryMode = "adaptive",
    max_tokens: int = 32_768,
) -> RepeatContextMetrics:
    """Stratify ordered next-token loss by exact contexts observed in training."""

    if context_length < 1 or max_tokens < 1:
        raise ValueError("repeat context length and token cap must be positive")
    sequences = {record.identifier: record.sequence.upper() for record in source_records}
    if len(sequences) != len(source_records):
        raise ValueError("repeat-context source identifiers must be unique")
    seen = {
        sequence[start : start + context_length]
        for record in training_records
        for sequence in (record.sequence.upper(),)
        for start in range(0, max(len(sequence) - context_length + 1, 0))
        if set(sequence[start : start + context_length]) <= set("ACGT")
    }
    totals = {True: 0.0, False: 0.0}
    counts = {True: 0, False: 0}
    state = None
    model.to(device)
    model.eval()
    classified = 0
    for horizon in dataset:
        if horizon.stream_start:
            state = model.initial_state()
        sequence = sequences[horizon.stream_id]
        for segment in horizon.segments:
            inputs = torch.tensor(segment.input_ids, dtype=torch.long, device=device)[None, :]
            valid_mask = torch.ones_like(inputs, dtype=torch.bool)
            context = (
                torch.enable_grad()
                if model.config.is_stateful and memory_mode == "adaptive"
                else torch.inference_mode()
            )
            with context:
                output = model.forward_segment(
                    inputs,
                    (state,) if state is not None else None,
                    valid_mask=valid_mask,
                    memory_mode=memory_mode,
                )
            state = output.states[0].detach()
            targets = torch.tensor(segment.target_ids, dtype=torch.long, device=device)
            losses = F.cross_entropy(output.logits[0].detach(), targets, reduction="none")
            for local_index, loss in enumerate(losses.tolist()):
                token_position = segment.token_offset + local_index + 1
                character_index = token_position - 1
                if character_index < context_length or character_index >= len(sequence):
                    continue
                kmer = sequence[character_index - context_length : character_index]
                repeated = kmer in seen
                totals[repeated] += float(loss)
                counts[repeated] += 1
                classified += 1
                if classified >= max_tokens:
                    break
            if classified >= max_tokens:
                break
        if classified >= max_tokens:
            break
        if horizon.stream_end:
            state = None
    if not counts[True] or not counts[False]:
        raise ValueError("repeat validation requires both repeated and novel contexts")
    repeated_mean = totals[True] / counts[True]
    novel_mean = totals[False] / counts[False]
    return RepeatContextMetrics(
        context_length=context_length,
        repeated_mean_nll=repeated_mean,
        repeated_bits_per_token=repeated_mean / math.log(2.0),
        repeated_tokens=counts[True],
        novel_mean_nll=novel_mean,
        novel_bits_per_token=novel_mean / math.log(2.0),
        novel_tokens=counts[False],
    )


def ordered_dataset_fingerprint(dataset: OrderedCausalStreamDataset) -> str:
    """Hash the exact horizon and segment schedule consumed by a run."""

    digest = hashlib.sha256()
    for horizon in dataset:
        digest.update(horizon.stream_id.encode())
        digest.update(b"\0")
        for segment in horizon.segments:
            digest.update(segment.token_offset.to_bytes(8, "little"))
            for token_id in segment.target_ids:
                digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()
