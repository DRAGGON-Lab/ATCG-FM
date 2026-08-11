"""Sequence-mixer interface and construction."""

from abc import ABC, abstractmethod

from atcg.models.config import MixerSpec
from torch import Tensor, nn


class SequenceMixer(nn.Module, ABC):
    """A length-preserving, causal transformation over hidden states."""

    @abstractmethod
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Mix a batch with shape ``(batch, sequence, channels)``."""


def build_mixer(
    spec: MixerSpec,
    *,
    d_model: int,
    n_heads: int,
    max_seq_len: int,
    dropout: float,
) -> SequenceMixer:
    """Construct a reference mixer from a validated specification."""

    if spec.kind == "attention":
        from atcg.models.mixers.attention import CausalSelfAttention

        return CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

    from atcg.models.mixers.hyena import HyenaLI, HyenaMR, HyenaSE

    if spec.kind == "hyena_se":
        return HyenaSE(d_model=d_model, kernel_size=spec.kernel_size or 7)
    if spec.kind == "hyena_mr":
        return HyenaMR(d_model=d_model, kernel_size=spec.kernel_size or 128)
    if spec.kind == "hyena_li":
        return HyenaLI(
            d_model=d_model,
            max_seq_len=max_seq_len,
            filter_hidden_size=spec.filter_hidden_size,
        )
    raise AssertionError(f"unhandled mixer kind {spec.kind!r}")
