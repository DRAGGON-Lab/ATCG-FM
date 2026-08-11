"""Causal multi-head attention using PyTorch's native SDPA dispatch."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atcg.models.mixers.base import SequenceMixer


def _apply_rotary(hidden_states: Tensor, inverse_frequencies: Tensor) -> Tensor:
    sequence_length = hidden_states.shape[-2]
    positions = torch.arange(sequence_length, device=hidden_states.device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequencies.float())
    cosine = angles.cos().to(dtype=hidden_states.dtype)[None, None, :, :]
    sine = angles.sin().to(dtype=hidden_states.dtype)[None, None, :, :]
    even = hidden_states[..., 0::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class CausalSelfAttention(SequenceMixer):
    """Pre-projection causal self-attention with rotary positions."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        if head_dim % 2:
            raise ValueError("head dimension must be even for rotary embeddings")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.input_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        inverse_frequencies = 1.0 / (
            10_000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.inverse_frequencies: Tensor
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        projected = self.input_projection(hidden_states)
        query, key, value = projected.chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(
                1, 2
            )

        query = _apply_rotary(heads(query), self.inverse_frequencies)
        key = _apply_rotary(heads(key), self.inverse_frequencies)
        value = heads(value)
        mixed = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        mixed = mixed.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        return self.output_projection(mixed)
