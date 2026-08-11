"""Reference causal depthwise convolutions."""

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def explicit_causal_convolution(inputs: Tensor, filters: Tensor) -> Tensor:
    """Apply per-channel filters whose final axis is ordered by increasing lag."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (batch, sequence, channels)")
    if filters.ndim != 2 or filters.shape[0] != inputs.shape[-1]:
        raise ValueError("filters must have shape (channels, kernel_size)")
    kernel_size = filters.shape[-1]
    transposed = inputs.transpose(1, 2)
    padded = F.pad(transposed, (kernel_size - 1, 0))
    weights = filters.flip(-1).unsqueeze(1)
    return F.conv1d(padded, weights, groups=inputs.shape[-1]).transpose(1, 2)


def fft_causal_convolution(inputs: Tensor, filters: Tensor) -> Tensor:
    """Apply a per-channel causal convolution using a zero-padded FFT."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (batch, sequence, channels)")
    sequence_length = inputs.shape[1]
    if filters.shape != (inputs.shape[-1], sequence_length):
        raise ValueError("filters must have shape (channels, sequence_length)")
    fft_size = 1 << max(1, (2 * sequence_length - 1).bit_length())
    transposed = inputs.transpose(1, 2)
    input_spectrum = cast(Tensor, torch.fft.rfft(transposed.float(), n=fft_size))
    filter_spectrum = cast(Tensor, torch.fft.rfft(filters.float(), n=fft_size))
    convolved = cast(
        Tensor,
        torch.fft.irfft(input_spectrum * filter_spectrum[None, :, :], n=fft_size),
    )
    return convolved[..., :sequence_length].transpose(1, 2).to(dtype=inputs.dtype)


class CausalDepthwiseConv1d(nn.Module):
    """Explicit learned finite impulse response filter."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if channels < 1 or kernel_size < 1:
            raise ValueError("channels and kernel_size must be positive")
        self.filters = nn.Parameter(torch.empty(channels, kernel_size))
        nn.init.normal_(self.filters, mean=0.0, std=1.0 / math.sqrt(kernel_size))

    def forward(self, inputs: Tensor) -> Tensor:
        return explicit_causal_convolution(inputs, self.filters)


class RegularizedCausalConv1d(nn.Module):
    """Explicit medium filter with a learned positive exponential decay."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if channels < 1 or kernel_size < 1:
            raise ValueError("channels and kernel_size must be positive")
        self.raw_filters = nn.Parameter(torch.empty(channels, kernel_size))
        self.log_decay = nn.Parameter(torch.full((channels,), -2.0))
        nn.init.normal_(self.raw_filters, mean=0.0, std=1.0 / math.sqrt(kernel_size))

    def filters(self) -> Tensor:
        lags = torch.arange(
            self.raw_filters.shape[-1],
            device=self.raw_filters.device,
            dtype=self.raw_filters.dtype,
        )
        decay_rates = F.softplus(self.log_decay).unsqueeze(-1)
        return self.raw_filters * torch.exp(-decay_rates * lags)

    def forward(self, inputs: Tensor) -> Tensor:
        return explicit_causal_convolution(inputs, self.filters())


class Sine(nn.Module):
    """Sine activation used by the implicit long-filter parameterization."""

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.sin(inputs)


class ImplicitLongFilter(nn.Module):
    """Generate a free-form long filter with a learned exponential envelope."""

    def __init__(self, channels: int, hidden_size: int, max_seq_len: int) -> None:
        super().__init__()
        if channels < 1 or hidden_size < 1 or max_seq_len < 1:
            raise ValueError("filter dimensions must be positive")
        self.channels = channels
        self.max_seq_len = max_seq_len
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            Sine(),
            nn.Linear(hidden_size, channels, bias=False),
        )
        self.log_decay = nn.Parameter(torch.full((channels,), -1.0))

    def filters(self, length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if not 1 <= length <= self.max_seq_len:
            raise ValueError(
                f"filter length {length} exceeds configured maximum {self.max_seq_len}"
            )
        positions = torch.arange(length, device=device, dtype=torch.float32)
        normalized = positions / max(length - 1, 1)
        values = self.network(normalized[:, None]).to(dtype=dtype)
        decay_rates = F.softplus(self.log_decay.float())
        envelope = torch.exp(-normalized[:, None] * decay_rates[None, :]).to(dtype=dtype)
        return (values * envelope).transpose(0, 1) / math.sqrt(length)

    def forward(self, inputs: Tensor) -> Tensor:
        filters = self.filters(inputs.shape[1], device=inputs.device, dtype=inputs.dtype)
        return fft_causal_convolution(inputs, filters)
