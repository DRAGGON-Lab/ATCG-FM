"""Dataset-level validation for autoregressive training runs."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from atcg.models import GenomicLanguageModel
from atcg.runtime.batching import (
    TARGET_IGNORE_ID,
    CausalBatch,
    SequenceDatasetAdapter,
    collate_examples,
)
from atcg.sequence import LanguageModelExample


@dataclass(frozen=True, slots=True)
class CausalValidationMetrics:
    """Dataset-level causal prediction metrics in explicit units."""

    total_nll: float
    mean_nll: float
    bits_per_token: float
    perplexity: float
    token_accuracy: float
    token_count: int
    example_count: int


def validate_causal_language_model(
    model: GenomicLanguageModel,
    dataset: Sequence[LanguageModelExample],
    *,
    pad_id: int,
    batch_size: int = 8,
    device: torch.device | str | None = None,
) -> CausalValidationMetrics:
    """Evaluate each non-padding target exactly once without changing weights."""

    if not dataset:
        raise ValueError("cannot evaluate an empty dataset")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved_device = next(model.parameters()).device if device is None else torch.device(device)
    model.to(resolved_device)
    model.eval()
    loader = cast(
        DataLoader[CausalBatch],
        DataLoader(
            SequenceDatasetAdapter(dataset),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(collate_examples, pad_id=pad_id),
            num_workers=0,
        ),
    )

    total_nll = 0.0
    correct_tokens = 0
    token_count = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = raw_batch.to(resolved_device)
            logits = model(batch.input_ids).logits
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = batch.target_ids.reshape(-1)
            valid = flat_targets != TARGET_IGNORE_ID
            losses = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=TARGET_IGNORE_ID,
                reduction="none",
            )
            total_nll += float(losses[valid].sum().item())
            correct_tokens += int(
                (flat_logits[valid].argmax(-1) == flat_targets[valid]).sum().item()
            )
            token_count += int(valid.sum().item())

    mean_nll = total_nll / token_count
    return CausalValidationMetrics(
        total_nll=total_nll,
        mean_nll=mean_nll,
        bits_per_token=mean_nll / math.log(2.0),
        perplexity=math.exp(mean_nll),
        token_accuracy=correct_tokens / token_count,
        token_count=token_count,
        example_count=len(dataset),
    )
