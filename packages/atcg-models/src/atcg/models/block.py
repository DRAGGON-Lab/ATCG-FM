"""Common block contract and the controlled standard mixer shell."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atcg.models.config import BlockSpec, StandardBlockSpec
from atcg.models.mixers import build_mixer
from atcg.models.state import RecurrentState, SegmentExecution, StateBatch


@dataclass(frozen=True, slots=True)
class BlockOutput:
    """Transformed segment and replacement state for each logical sequence."""

    hidden_states: Tensor
    states: StateBatch
    diagnostics: Mapping[str, Tensor] = field(default_factory=lambda: dict[str, Tensor]())


class SequenceBlock(nn.Module, ABC):
    """Substitution boundary for complete causal sequence blocks."""

    @abstractmethod
    def initial_state(self) -> RecurrentState | None: ...

    @abstractmethod
    def forward_segment(
        self,
        hidden_states: Tensor,
        states: StateBatch,
        *,
        execution: SegmentExecution,
    ) -> BlockOutput: ...

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, channels)")
        execution = SegmentExecution(
            valid_mask=torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        )
        states = tuple(self.initial_state() for _ in range(hidden_states.shape[0]))
        return self.forward_segment(hidden_states, states, execution=execution).hidden_states


class SwiGLU(nn.Module):
    """Gated feed-forward network held constant across mixer experiments."""

    def __init__(self, d_model: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(d_model, 2 * hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, values = self.input_projection(hidden_states).chunk(2, dim=-1)
        return self.output_projection(self.dropout(F.silu(gate) * values))


class StandardMixerBlock(SequenceBlock):
    """Fixed pre-norm residual shell used for controlled mixer substitutions."""

    def __init__(
        self,
        *,
        spec: StandardBlockSpec,
        d_model: int,
        max_seq_len: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.mixer_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.mixer = build_mixer(
            spec.mixer,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=spec.dropout,
        )
        self.mlp_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLU(
            d_model=d_model,
            hidden_size=spec.mlp_hidden_size,
            dropout=spec.dropout,
        )
        self.residual_dropout = nn.Dropout(spec.dropout)

    def initial_state(self) -> RecurrentState | None:
        return self.mixer.initial_state()

    def forward_segment(
        self,
        hidden_states: Tensor,
        states: StateBatch,
        *,
        execution: SegmentExecution,
    ) -> BlockOutput:
        mixed = self.mixer.mix_segment(
            self.mixer_norm(hidden_states),
            states,
            execution=execution,
        )
        hidden_states = hidden_states + self.residual_dropout(mixed.hidden_states)
        hidden_states = hidden_states + self.residual_dropout(
            self.mlp(self.mlp_norm(hidden_states))
        )
        return BlockOutput(hidden_states, mixed.states, mixed.diagnostics)


def build_block(
    spec: BlockSpec,
    *,
    d_model: int,
    max_seq_len: int,
    norm_eps: float,
) -> SequenceBlock:
    """Construct either a standard mixer shell or a complete composite block."""

    if isinstance(spec, StandardBlockSpec):
        return StandardMixerBlock(
            spec=spec,
            d_model=d_model,
            max_seq_len=max_seq_len,
            norm_eps=norm_eps,
        )
    from atcg.models.titans.mac import TitansMACBlock

    return TitansMACBlock(spec=spec, d_model=d_model, norm_eps=norm_eps)
