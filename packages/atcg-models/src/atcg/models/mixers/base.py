"""Sequence-mixer contracts and construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from atcg.models.config import (
    AttentionSpec,
    HyenaLISpec,
    HyenaMRSpec,
    HyenaSESpec,
    MixerSpec,
)
from atcg.models.state import RecurrentState, SegmentExecution, StateBatch


@dataclass(frozen=True, slots=True)
class MixerOutput:
    """Length-preserving activations plus replacement per-stream state."""

    hidden_states: Tensor
    states: StateBatch
    diagnostics: Mapping[str, Tensor] = field(default_factory=lambda: dict[str, Tensor]())


class SequenceMixer(nn.Module):
    """A causal transformation over a segment, optionally carrying stream state."""

    def initial_state(self) -> RecurrentState | None:
        return None

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Stateless convenience path used by ordinary attention and Hyena mixers."""

        raise NotImplementedError

    def mix_segment(
        self,
        hidden_states: Tensor,
        states: StateBatch,
        *,
        execution: SegmentExecution,
    ) -> MixerOutput:
        """Default vectorized implementation for stateless mixers."""

        if len(states) != hidden_states.shape[0]:
            raise ValueError("one mixer state is required per batch row")
        if any(state is not None for state in states):
            raise TypeError("a stateless mixer received recurrent state")
        if execution.valid_mask.shape != hidden_states.shape[:2]:
            raise ValueError("valid_mask must match mixer batch and sequence dimensions")
        outputs = self(hidden_states)
        mask = execution.valid_mask.unsqueeze(-1).to(outputs.dtype)
        return MixerOutput(outputs * mask, states)


def build_mixer(
    spec: MixerSpec,
    *,
    d_model: int,
    max_seq_len: int,
    dropout: float,
) -> SequenceMixer:
    """Construct a causal mixer from a typed specification."""

    if isinstance(spec, AttentionSpec):
        from atcg.models.mixers.attention import CausalSelfAttention

        return CausalSelfAttention(d_model=d_model, n_heads=spec.n_heads, dropout=dropout)

    from atcg.models.mixers.hyena import HyenaLI, HyenaMR, HyenaSE

    if isinstance(spec, HyenaSESpec):
        return HyenaSE(d_model=d_model, kernel_size=spec.kernel_size)
    if isinstance(spec, HyenaMRSpec):
        return HyenaMR(d_model=d_model, kernel_size=spec.kernel_size)
    if isinstance(spec, HyenaLISpec):
        return HyenaLI(
            d_model=d_model,
            max_seq_len=max_seq_len,
            filter_hidden_size=spec.filter_hidden_size,
        )
    from atcg.models.mixers.titans_memory import TitansMemoryMixer

    return TitansMemoryMixer(d_model=d_model, spec=spec)


def fresh_state_batch(mixer: SequenceMixer, batch_size: int) -> StateBatch:
    """Create independent initial states for every logical sequence in a batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return tuple(mixer.initial_state() for _ in range(batch_size))


def full_valid_mask(hidden_states: Tensor) -> Tensor:
    return torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
