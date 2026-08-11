"""Residual decoder block with an interchangeable sequence mixer."""

from atcg.models.config import MixerSpec
from atcg.models.mixers import build_mixer
from torch import Tensor, nn
from torch.nn import functional as F


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


class MixerBlock(nn.Module):
    """Pre-normalized causal mixer and feed-forward residual block."""

    def __init__(
        self,
        *,
        mixer_spec: MixerSpec,
        d_model: int,
        n_heads: int,
        mlp_hidden_size: int,
        max_seq_len: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.mixer_kind = mixer_spec.kind
        self.mixer_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.mixer = build_mixer(
            mixer_spec,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.mlp_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLU(d_model=d_model, hidden_size=mlp_hidden_size, dropout=dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.residual_dropout(
            self.mixer(self.mixer_norm(hidden_states))
        )
        return hidden_states + self.residual_dropout(self.mlp(self.mlp_norm(hidden_states)))
