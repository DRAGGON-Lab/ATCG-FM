"""Causal TITANS Memory-as-Context composite block."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atcg.models.block import BlockOutput, SequenceBlock, SwiGLU
from atcg.models.config import TitansMACBlockSpec
from atcg.models.state import RecurrentState, SegmentExecution, StateBatch
from atcg.models.titans.memory import MemoryStepOutput, NeuralMemory
from atcg.models.titans.state import NeuralMemoryState


def mac_attention_mask(
    persistent_tokens: int,
    segment_length: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Boolean ``[persistent; retrieval; segment]`` causal attention mask."""

    layout_length = persistent_tokens + 2 * segment_length
    mask = torch.ones(layout_length, layout_length, dtype=torch.bool, device=device)
    mask[:persistent_tokens, :persistent_tokens] = False
    retrieval_start = persistent_tokens
    sequence_start = persistent_tokens + segment_length
    for position in range(segment_length):
        mask[retrieval_start + position, :persistent_tokens] = False
        mask[retrieval_start + position, retrieval_start : retrieval_start + position + 1] = False
        mask[retrieval_start + position, sequence_start : sequence_start + position + 1] = False
        mask[sequence_start + position, :persistent_tokens] = False
        mask[sequence_start + position, retrieval_start : retrieval_start + position + 1] = False
        mask[sequence_start + position, sequence_start : sequence_start + position + 1] = False
    return mask


class TitansMACBlock(SequenceBlock):
    """Whole-block substitution coupling local attention and adaptive neural memory."""

    def __init__(self, *, spec: TitansMACBlockSpec, d_model: int, norm_eps: float) -> None:
        super().__init__()
        self.spec = spec
        self.input_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.memory_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.memory = NeuralMemory(
            d_model,
            expansion_factor=spec.memory_expansion_factor,
            projection_kernel_size=spec.projection_kernel_size,
            alpha_initial=spec.alpha_initial,
            eta_initial=spec.eta_initial,
            theta_initial=spec.theta_initial,
        )
        self.persistent_tokens = nn.Parameter(torch.empty(spec.persistent_tokens, d_model))
        self.position_embeddings = nn.Parameter(torch.empty(spec.segment_length, d_model))
        self.attention = nn.MultiheadAttention(
            d_model,
            spec.n_heads,
            dropout=spec.dropout,
            batch_first=True,
        )
        self.output_gate = nn.Linear(2 * d_model, d_model)
        self.mlp_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLU(d_model, spec.mlp_hidden_size, spec.dropout)
        self.residual_dropout = nn.Dropout(spec.dropout)
        nn.init.normal_(self.persistent_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

    def initial_state(self) -> RecurrentState:
        return self.memory.initial_state()

    def _attend(self, layout: Tensor, blocked: Tensor, key_padding: Tensor) -> Tensor:
        """Manual attention path supporting higher-order fast-memory gradients."""

        packed = F.linear(layout, self.attention.in_proj_weight, self.attention.in_proj_bias)
        query, key, value = packed.chunk(3, dim=-1)
        batch, length, width = query.shape
        heads = self.attention.num_heads
        head_width = width // heads

        def split(values: Tensor) -> Tensor:
            return values.view(batch, length, heads, head_width).transpose(1, 2)

        query, key, value = map(split, (query, key, value))
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_width**0.5)
        combined_mask = blocked.unsqueeze(0) | key_padding.unsqueeze(1)
        scores = scores.masked_fill(combined_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.spec.dropout, training=self.training)
        attended = torch.matmul(weights, value)
        merged = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return F.linear(
            merged,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )

    def forward_segment(
        self,
        hidden_states: Tensor,
        states: StateBatch,
        *,
        execution: SegmentExecution,
    ) -> BlockOutput:
        batch_size, segment_length, _ = hidden_states.shape
        if segment_length != self.spec.segment_length:
            raise ValueError(
                "TITANS MAC requires segment length "
                f"{self.spec.segment_length}, got {segment_length}"
            )
        if len(states) != batch_size:
            raise ValueError("one MAC state is required per batch row")
        normalized = self.input_norm(hidden_states) + self.position_embeddings.to(hidden_states)
        sanitized = normalized * execution.valid_mask.unsqueeze(-1).to(normalized.dtype)

        context_histories: list[Tensor | None] = []
        retrieval_rows: list[Tensor] = []
        for row, state in enumerate(states):
            if not isinstance(state, NeuralMemoryState):
                raise TypeError("TITANS MAC requires NeuralMemoryState per row")
            if execution.memory_mode == "disabled":
                retrieval_rows.append(torch.zeros_like(sanitized[row]))
                context_histories.append(None)
            else:
                retrieval, history = self.memory.read_context(state, sanitized[row])
                retrieval_rows.append(retrieval.to(hidden_states))
                context_histories.append(history)
        retrieval = torch.stack(retrieval_rows)

        persistent = (
            self.persistent_tokens.to(hidden_states).unsqueeze(0).expand(batch_size, -1, -1)
        )
        layout = torch.cat((persistent, retrieval, sanitized), dim=1)
        blocked = mac_attention_mask(
            self.spec.persistent_tokens,
            segment_length,
            device=hidden_states.device,
        )
        key_padding = torch.cat(
            (
                torch.zeros(
                    (batch_size, self.spec.persistent_tokens),
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
                ~execution.valid_mask,
                ~execution.valid_mask,
            ),
            dim=1,
        )
        attended = self._attend(layout, blocked, key_padding)
        sequence_start = self.spec.persistent_tokens + segment_length
        sequence = hidden_states + self.residual_dropout(attended[:, sequence_start:])

        if execution.memory_mode == "disabled":
            next_states = states
            diagnostics: dict[str, Tensor] = {}
        else:
            rows: list[MemoryStepOutput] = []
            create_graph = self.training and torch.is_grad_enabled()
            for row, state in enumerate(states):
                assert isinstance(state, NeuralMemoryState)
                rows.append(
                    self.memory.update_and_read(
                        state,
                        self.memory_norm(sequence[row]),
                        valid_mask=execution.valid_mask[row],
                        context_query_history=context_histories[row],
                        update_memory=execution.memory_mode == "adaptive",
                        create_graph=create_graph,
                    )
                )
            post_retrieval = torch.stack([row.retrieval for row in rows]).to(sequence)
            gate = torch.sigmoid(self.output_gate(torch.cat((sequence, post_retrieval), dim=-1)))
            sequence = sequence + self.residual_dropout(gate * post_retrieval)
            next_states = tuple(row.state for row in rows)
            diagnostics = {
                "memory_alpha": torch.stack([row.mean_alpha for row in rows]).mean(),
                "memory_eta": torch.stack([row.mean_eta for row in rows]).mean(),
                "memory_theta": torch.stack([row.mean_theta for row in rows]).mean(),
                "memory_output_gate": gate.float().mean(),
            }

        sequence = sequence + self.residual_dropout(self.mlp(self.mlp_norm(sequence)))
        return BlockOutput(sequence, next_states, diagnostics)
