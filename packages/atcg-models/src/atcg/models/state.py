"""Explicit recurrent state contracts for stateful sequence architectures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, Self, runtime_checkable

import torch
from torch import Tensor

type MemoryMode = Literal["adaptive", "frozen", "disabled"]


@runtime_checkable
class RecurrentState(Protocol):
    """Serializable runtime state owned by one logical sequence."""

    def detach(self) -> Self: ...

    def to(self, device: torch.device | str) -> Self: ...

    def state_dict(self) -> dict[str, object]: ...


type StateBatch = tuple[RecurrentState | None, ...]


@dataclass(frozen=True, slots=True)
class SegmentExecution:
    """Execution policy shared by mixers and whole-block substitutions."""

    valid_mask: Tensor
    memory_mode: MemoryMode = "adaptive"

    def __post_init__(self) -> None:
        if self.valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape (batch, sequence)")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")


@dataclass(frozen=True, slots=True)
class ModelState:
    """One logical sequence's recurrent state, aligned with model blocks."""

    blocks: tuple[RecurrentState | None, ...]

    def detach(self) -> ModelState:
        return ModelState(
            tuple(state.detach() if state is not None else None for state in self.blocks)
        )

    def to(self, device: torch.device | str) -> ModelState:
        return ModelState(
            tuple(state.to(device) if state is not None else None for state in self.blocks)
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "blocks": [state.state_dict() if state is not None else None for state in self.blocks],
        }


@dataclass(frozen=True, slots=True)
class ArchitectureDiagnostics:
    """Detached scalar diagnostics produced without changing model semantics."""

    values: Mapping[str, Tensor]
