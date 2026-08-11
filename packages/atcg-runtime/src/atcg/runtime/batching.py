"""Tensor collation for variable-length causal examples."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor
from torch.utils.data import Dataset

from atcg.sequence import LanguageModelExample

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
