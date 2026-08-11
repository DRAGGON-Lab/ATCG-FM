"""Readable Hyena-style gated convolution operators."""

from torch import Tensor, nn

from atcg.models.mixers.base import SequenceMixer
from atcg.models.mixers.convolution import (
    CausalDepthwiseConv1d,
    ImplicitLongFilter,
    RegularizedCausalConv1d,
)


class _HyenaReference(SequenceMixer):
    """Three-stream gated convolution shared by the reference variants."""

    def __init__(self, *, d_model: int, inner_filter: nn.Module) -> None:
        super().__init__()
        self.input_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.short_filter = CausalDepthwiseConv1d(3 * d_model, kernel_size=3)
        self.inner_filter = inner_filter
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        projected = self.short_filter(self.input_projection(hidden_states))
        outer_gate, inner_gate, values = projected.chunk(3, dim=-1)
        mixed = self.inner_filter(inner_gate * values)
        return self.output_projection(outer_gate * mixed)


class HyenaSE(_HyenaReference):
    """Short explicit Hyena-style operator."""

    def __init__(self, *, d_model: int, kernel_size: int = 7) -> None:
        super().__init__(
            d_model=d_model,
            inner_filter=CausalDepthwiseConv1d(d_model, kernel_size=kernel_size),
        )


class HyenaMR(_HyenaReference):
    """Medium explicit Hyena-style operator with decayed filters."""

    def __init__(self, *, d_model: int, kernel_size: int = 128) -> None:
        super().__init__(
            d_model=d_model,
            inner_filter=RegularizedCausalConv1d(d_model, kernel_size=kernel_size),
        )


class HyenaLI(_HyenaReference):
    """Long implicit Hyena-style operator evaluated by FFT convolution."""

    def __init__(self, *, d_model: int, max_seq_len: int, filter_hidden_size: int = 64) -> None:
        super().__init__(
            d_model=d_model,
            inner_filter=ImplicitLongFilter(
                channels=d_model,
                hidden_size=filter_hidden_size,
                max_seq_len=max_seq_len,
            ),
        )
