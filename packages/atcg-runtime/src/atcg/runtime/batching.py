"""Tensor collation for variable-length causal examples."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor
from torch.utils.data import Dataset

from atcg.sequence import LanguageModelExample, LanguageModelHorizon

TARGET_IGNORE_ID = -100


class SequenceDatasetAdapter[ExampleT](Dataset[ExampleT]):
    """Expose a framework-neutral sequence through PyTorch's dataset protocol."""

    def __init__(self, values: Sequence[ExampleT]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> ExampleT:
        return self._values[index]


@dataclass(frozen=True, slots=True)
class CausalBatch:
    """Right-padded inputs and shifted targets."""

    input_ids: Tensor
    target_ids: Tensor
    lengths: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.target_ids.shape != self.input_ids.shape:
            raise ValueError("input_ids and target_ids must have matching rank-two shapes")
        if self.lengths.shape != (self.input_ids.shape[0],):
            raise ValueError("lengths must contain one value per example")

    @property
    def token_count(self) -> int:
        return int(self.lengths.sum().item())

    def to(self, device: torch.device | str) -> Self:
        return type(self)(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            lengths=self.lengths.to(device),
        )


@dataclass(frozen=True, slots=True)
class StatefulCausalBatch:
    """Ordered stream segments grouped into a truncated-gradient horizon."""

    input_ids: Tensor
    target_ids: Tensor
    valid_mask: Tensor
    stream_ids: tuple[str, ...]
    stream_starts: tuple[bool, ...]
    stream_ends: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 3 or self.target_ids.shape != self.input_ids.shape:
            raise ValueError("stateful inputs and targets must have matching rank-three shapes")
        if self.valid_mask.shape != self.input_ids.shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("stateful valid_mask must be boolean and match input_ids")
        batch_size = self.input_ids.shape[0]
        if not (
            len(self.stream_ids) == len(self.stream_starts) == len(self.stream_ends) == batch_size
        ):
            raise ValueError("stateful stream metadata must contain one entry per batch row")
        if len(set(self.stream_ids)) != len(self.stream_ids):
            raise ValueError("a stateful batch cannot contain a stream more than once")

    @property
    def token_count(self) -> int:
        return int(self.valid_mask.sum().item())

    def to(self, device: torch.device | str) -> Self:
        return type(self)(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            valid_mask=self.valid_mask.to(device),
            stream_ids=self.stream_ids,
            stream_starts=self.stream_starts,
            stream_ends=self.stream_ends,
        )


def collate_examples(
    examples: Sequence[LanguageModelExample],
    *,
    pad_id: int,
    target_ignore_id: int = TARGET_IGNORE_ID,
) -> CausalBatch:
    """Collate examples without allowing padding targets to affect the loss."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    maximum_length = max(len(example.input_ids) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, maximum_length), pad_id, dtype=torch.long)
    target_ids = torch.full((batch_size, maximum_length), target_ignore_id, dtype=torch.long)
    lengths = torch.empty(batch_size, dtype=torch.long)

    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        target_ids[row, :length] = torch.tensor(example.target_ids, dtype=torch.long)
        lengths[row] = length

    return CausalBatch(input_ids=input_ids, target_ids=target_ids, lengths=lengths)


def collate_horizons(
    horizons: Sequence[LanguageModelHorizon],
    *,
    pad_id: int,
    segment_length: int,
    target_ignore_id: int = TARGET_IGNORE_ID,
) -> StatefulCausalBatch:
    """Collate ordered horizons without crossing or duplicating logical streams."""

    if not horizons:
        raise ValueError("cannot collate an empty horizon batch")
    if segment_length < 1:
        raise ValueError("segment_length must be positive")
    horizon_width = max(len(horizon.segments) for horizon in horizons)
    shape = (len(horizons), horizon_width, segment_length)
    input_ids = torch.full(shape, pad_id, dtype=torch.long)
    target_ids = torch.full(shape, target_ignore_id, dtype=torch.long)
    valid_mask = torch.zeros(shape, dtype=torch.bool)
    for row, horizon in enumerate(horizons):
        for column, segment in enumerate(horizon.segments):
            length = len(segment.input_ids)
            if length > segment_length:
                raise ValueError("a horizon segment exceeds the configured segment length")
            input_ids[row, column, :length] = torch.tensor(segment.input_ids, dtype=torch.long)
            target_ids[row, column, :length] = torch.tensor(segment.target_ids, dtype=torch.long)
            valid_mask[row, column, :length] = True
    return StatefulCausalBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        valid_mask=valid_mask,
        stream_ids=tuple(horizon.stream_id for horizon in horizons),
        stream_starts=tuple(horizon.stream_start for horizon in horizons),
        stream_ends=tuple(horizon.stream_end for horizon in horizons),
    )
