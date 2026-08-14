"""Autoregressive genomic language model with explicit per-stream state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atcg.models.block import SequenceBlock, build_block
from atcg.models.config import ModelConfig
from atcg.models.state import MemoryMode, ModelState, RecurrentState, SegmentExecution
from atcg.models.titans.state import NeuralMemoryState


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    """Activations, diagnostics, and replacement state for each batch row."""

    logits: Tensor
    hidden_states: Tensor
    states: tuple[ModelState, ...]
    intermediate_hidden_states: tuple[Tensor, ...] | None = None
    diagnostics: Mapping[str, Tensor] = field(default_factory=lambda: dict[str, Tensor]())


class GenomicLanguageModel(nn.Module):
    """Decoder-only model supporting controlled mixer and whole-block schedules."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                build_block(
                    spec,
                    d_model=config.d_model,
                    max_seq_len=config.max_seq_len,
                    norm_eps=config.norm_eps,
                )
                for spec in config.blocks
            ]
        )
        self.final_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_projection: nn.Linear | None = (
            None
            if config.tie_embeddings
            else nn.Linear(config.d_model, config.vocab_size, bias=False)
        )
        self.apply(self._initialize)
        self._restore_special_initialization()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _restore_special_initialization(self) -> None:
        from atcg.models.titans.memory import NeuralMemory

        for module in self.modules():
            if isinstance(module, NeuralMemory):
                module.reset_update_mechanism()

    def initial_state(self) -> ModelState:
        return ModelState(
            tuple(cast(SequenceBlock, block).initial_state() for block in self.blocks)
        )

    def state_from_dict(self, payload: Mapping[str, object]) -> ModelState:
        """Restore one logical stream's state against this model's block schedule."""

        if payload.get("format_version") != 1:
            raise ValueError("unsupported model-state format")
        raw_blocks_value = payload.get("blocks")
        if not isinstance(raw_blocks_value, list):
            raise ValueError("model-state blocks must be a list")
        raw_blocks = cast(list[object], raw_blocks_value)
        if len(raw_blocks) != len(self.blocks):
            raise ValueError("model-state block count does not match the model")
        restored: list[RecurrentState | None] = []
        for index, raw_state in enumerate(raw_blocks):
            expected = cast(SequenceBlock, self.blocks[index]).initial_state()
            if expected is None:
                if raw_state is not None:
                    raise ValueError(f"stateless block {index} received serialized state")
                restored.append(None)
            elif isinstance(expected, NeuralMemoryState):
                if not isinstance(raw_state, Mapping):
                    raise TypeError(f"stateful block {index} requires a mapping")
                restored.append(
                    NeuralMemoryState.from_state_dict(cast(Mapping[str, object], raw_state))
                )
            else:
                raise TypeError(f"unsupported recurrent state type at block {index}")
        return ModelState(tuple(restored))

    def forward_segment(
        self,
        input_ids: Tensor,
        states: Sequence[ModelState] | None = None,
        *,
        valid_mask: Tensor | None = None,
        memory_mode: MemoryMode = "adaptive",
        return_intermediate_hidden_states: bool = False,
    ) -> CausalLMOutput:
        """Process one ordered segment and return replacement state without mutation."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds maximum {self.config.max_seq_len}"
            )
        batch_size = input_ids.shape[0]
        resolved_states = (
            tuple(self.initial_state() for _ in range(batch_size))
            if states is None
            else tuple(states)
        )
        if len(resolved_states) != batch_size:
            raise ValueError("one model state is required per batch row")
        for state in resolved_states:
            if len(state.blocks) != len(self.blocks):
                raise ValueError("model state block count does not match the model")
        if valid_mask is None:
            valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        execution = SegmentExecution(valid_mask=valid_mask.bool(), memory_mode=memory_mode)

        hidden_states = self.token_embedding(input_ids)
        intermediate: list[Tensor] | None = [] if return_intermediate_hidden_states else None
        next_rows: list[list[RecurrentState | None]] = [[] for _ in range(batch_size)]
        diagnostics: dict[str, Tensor] = {}
        for block_index, module in enumerate(self.blocks):
            block = cast(SequenceBlock, module)
            block_states = tuple(state.blocks[block_index] for state in resolved_states)
            output = block.forward_segment(hidden_states, block_states, execution=execution)
            hidden_states = output.hidden_states
            for row_index, state in enumerate(output.states):
                next_rows[row_index].append(state)
            for name, value in output.diagnostics.items():
                diagnostics[f"block.{block_index}.{name}"] = value
            if intermediate is not None:
                intermediate.append(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        logits = (
            self.output_projection(hidden_states)
            if self.output_projection is not None
            else F.linear(hidden_states, self.token_embedding.weight)
        )
        return CausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            states=tuple(ModelState(tuple(row)) for row in next_rows),
            intermediate_hidden_states=tuple(intermediate) if intermediate is not None else None,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        valid_mask: Tensor | None = None,
        memory_mode: MemoryMode = "adaptive",
        return_intermediate_hidden_states: bool = False,
    ) -> CausalLMOutput:
        """Fresh-state convenience path for independent examples."""

        return self.forward_segment(
            input_ids,
            valid_mask=valid_mask,
            memory_mode=memory_mode,
            return_intermediate_hidden_states=return_intermediate_hidden_states,
        )

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def recurrent_state_elements(self) -> int:
        """Count dynamic scalar state carried by one logical sequence."""

        state = self.initial_state()
        total = 0
        for block_state in state.blocks:
            if isinstance(block_state, NeuralMemoryState):
                tensors = (
                    *block_state.fast_weights.values(),
                    *block_state.surprise.values(),
                    block_state.context_query_history,
                    block_state.output_query_history,
                    block_state.write_history,
                )
                total += sum(tensor.numel() for tensor in tensors)
        return total
