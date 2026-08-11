"""Small native-PyTorch training loop for controlled experiments."""

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from atcg.models import GenomicLanguageModel
from atcg.runtime.batching import (
    TARGET_IGNORE_ID,
    CausalBatch,
    SequenceDatasetAdapter,
    collate_examples,
)
from atcg.runtime.training_state import TrainingState
from atcg.sequence import LanguageModelExample

Precision = Literal["float32", "bfloat16"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Parameters that determine a finite local training run."""

    max_steps: int
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip_norm: float | None = 1.0
    seed: int = 17
    device: str = "cpu"
    precision: Precision = "float32"
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.batch_size < 1:
            raise ValueError("max_steps and batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must not be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive or null")
        if self.precision == "bfloat16" and self.device == "cpu":
            raise ValueError("bfloat16 training is not enabled for the CPU reference path")


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Observed optimization metrics for one step."""

    step: int
    loss: float
    gradient_norm: float
    tokens: int
    tokens_seen: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Result of a completed finite training run."""

    state: TrainingState
    metrics: tuple[StepMetrics, ...]
    checkpoint_path: Path | None = None


def causal_language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean next-token cross entropy, excluding padded targets."""

    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets have incompatible shapes")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=TARGET_IGNORE_ID,
    )


class Trainer:
    """Stateful optimizer wrapper with no ownership of data iteration."""

    def __init__(
        self,
        model: GenomicLanguageModel,
        optimizer: Optimizer,
        *,
        device: torch.device | str = "cpu",
        precision: Precision = "float32",
        gradient_clip_norm: float | None = 1.0,
        state: TrainingState | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.precision = precision
        self.gradient_clip_norm = gradient_clip_norm
        self.state = state or TrainingState()
        self.model.to(self.device)

    def train_step(self, batch: CausalBatch) -> StepMetrics:
        """Run one optimizer step and update token-based progress."""

        started = time.perf_counter()
        local_batch = batch.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.precision == "bfloat16"
            else nullcontext()
        )
        with autocast_context:
            logits = self.model(local_batch.input_ids)
            loss = causal_language_model_loss(logits, local_batch.target_ids)
        loss.backward()
        if self.gradient_clip_norm is None:
            gradient_norm = _gradient_norm(self.model.parameters())
        else:
            gradient_norm = float(
                clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm).item()
            )
        self.optimizer.step()

        self.state = TrainingState(
            step=self.state.step + 1,
            tokens_seen=self.state.tokens_seen + local_batch.token_count,
        )
        return StepMetrics(
            step=self.state.step,
            loss=float(loss.detach().item()),
            gradient_norm=gradient_norm,
            tokens=local_batch.token_count,
            tokens_seen=self.state.tokens_seen,
            elapsed_seconds=time.perf_counter() - started,
        )


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().float().square().sum().item())
    return squared_norm**0.5


def fit(
    model: GenomicLanguageModel,
    dataset: Sequence[LanguageModelExample],
    *,
    pad_id: int,
    config: TrainingConfig,
    run_dir: str | Path | None = None,
) -> TrainingRun:
    """Train for exactly ``max_steps`` and optionally materialize a run bundle."""

    if not dataset:
        raise ValueError("cannot train on an empty dataset")
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    loader = cast(
        DataLoader[CausalBatch],
        DataLoader(
            SequenceDatasetAdapter(dataset),
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            generator=generator,
            collate_fn=partial(collate_examples, pad_id=pad_id),
            num_workers=0,
        ),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trainer = Trainer(
        model,
        optimizer,
        device=config.device,
        precision=config.precision,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    metrics: list[StepMetrics] = []
    while trainer.state.step < config.max_steps:
        for batch in loader:
            metrics.append(trainer.train_step(batch))
            if trainer.state.step >= config.max_steps:
                break

    checkpoint_path: Path | None = None
    if run_dir is not None:
        destination = Path(run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "manifest.json", _manifest(model, config, len(dataset)))
        _write_metrics(destination / "metrics.jsonl", metrics)
        from atcg.runtime.checkpoint import save_checkpoint

        checkpoint_path = save_checkpoint(
            destination / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            training_state=trainer.state,
            metadata={"run_directory": str(destination)},
        )

    return TrainingRun(
        state=trainer.state,
        metrics=tuple(metrics),
        checkpoint_path=checkpoint_path,
    )


def _manifest(
    model: GenomicLanguageModel,
    config: TrainingConfig,
    dataset_examples: int,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "model_config": model.config.to_dict(),
        "training_config": asdict(config),
        "dataset_examples": dataset_examples,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": config.device,
        },
        "source": _source_state(),
    }


def _source_state() -> dict[str, object]:
    source: dict[str, object] = {"git_commit": None, "git_dirty": None, "uv_lock_sha256": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        source["git_commit"] = commit
        source["git_dirty"] = bool(dirty_result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    lock_path = Path.cwd() / "uv.lock"
    if lock_path.is_file():
        source["uv_lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return source


def _write_json(path: Path, values: dict[str, object]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_metrics(path: Path, metrics: Sequence[StepMetrics]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for metric in metrics:
            output.write(json.dumps(asdict(metric), sort_keys=True) + "\n")
