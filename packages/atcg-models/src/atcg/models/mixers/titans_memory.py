"""TITANS neural memory as a controlled standard-shell mixer substitution."""

from __future__ import annotations

import torch
from torch import Tensor

from atcg.models.config import TitansMemorySpec
from atcg.models.mixers.base import MixerOutput, SequenceMixer, fresh_state_batch
from atcg.models.state import RecurrentState, SegmentExecution, StateBatch
from atcg.models.titans.memory import MemoryStepOutput, NeuralMemory
from atcg.models.titans.state import NeuralMemoryState


class TitansMemoryMixer(SequenceMixer):
    """Neural long-term memory isolated inside the standard ATCG block shell."""

    def __init__(self, *, d_model: int, spec: TitansMemorySpec) -> None:
        super().__init__()
        self.memory = NeuralMemory(
            d_model,
            expansion_factor=spec.expansion_factor,
            projection_kernel_size=spec.projection_kernel_size,
            alpha_initial=spec.alpha_initial,
            eta_initial=spec.eta_initial,
            theta_initial=spec.theta_initial,
        )

    def initial_state(self) -> RecurrentState:
        return self.memory.initial_state()

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, channels)")
        execution = SegmentExecution(
            valid_mask=torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        )
        return self.mix_segment(
            hidden_states,
            fresh_state_batch(self, hidden_states.shape[0]),
            execution=execution,
        ).hidden_states

    def mix_segment(
        self,
        hidden_states: Tensor,
        states: StateBatch,
        *,
        execution: SegmentExecution,
    ) -> MixerOutput:
        if len(states) != hidden_states.shape[0]:
            raise ValueError("one mixer state is required per batch row")
        if execution.valid_mask.shape != hidden_states.shape[:2]:
            raise ValueError("valid_mask must match mixer batch and sequence dimensions")
        if execution.memory_mode == "disabled":
            return MixerOutput(torch.zeros_like(hidden_states), states)

        rows: list[MemoryStepOutput] = []
        create_graph = self.training and torch.is_grad_enabled()
        for row, state in enumerate(states):
            if not isinstance(state, NeuralMemoryState):
                raise TypeError("TITANS memory mixer requires NeuralMemoryState per row")
            rows.append(
                self.memory.update_and_read(
                    state,
                    hidden_states[row],
                    valid_mask=execution.valid_mask[row],
                    update_memory=execution.memory_mode == "adaptive",
                    create_graph=create_graph,
                )
            )
        return MixerOutput(
            hidden_states=torch.stack([row.retrieval for row in rows]).to(hidden_states),
            states=tuple(row.state for row in rows),
            diagnostics={
                "memory_alpha": torch.stack([row.mean_alpha for row in rows]).mean(),
                "memory_eta": torch.stack([row.mean_eta for row in rows]).mean(),
                "memory_theta": torch.stack([row.mean_theta for row in rows]).mean(),
            },
        )
