import torch
from atcg.models.mixers.convolution import (
    explicit_causal_convolution,
    fft_causal_convolution,
)


def _naive_causal_convolution(inputs: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(inputs)
    for position in range(inputs.shape[1]):
        for lag in range(min(filters.shape[-1], position + 1)):
            output[:, position] += inputs[:, position - lag] * filters[:, lag]
    return output


def test_explicit_convolution_matches_naive_reference() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(2, 7, 3, dtype=torch.float64)
    filters = torch.randn(3, 4, dtype=torch.float64)

    actual = explicit_causal_convolution(inputs, filters)

    torch.testing.assert_close(actual, _naive_causal_convolution(inputs, filters))


def test_fft_convolution_matches_naive_reference() -> None:
    torch.manual_seed(11)
    inputs = torch.randn(2, 9, 3, dtype=torch.float64)
    filters = torch.randn(3, 9, dtype=torch.float64)

    actual = fft_causal_convolution(inputs, filters)

    torch.testing.assert_close(
        actual,
        _naive_causal_convolution(inputs, filters),
        atol=1e-6,
        rtol=1e-6,
    )


def test_causal_convolution_has_finite_gradients() -> None:
    inputs = torch.randn(2, 6, 3, requires_grad=True)
    filters = torch.randn(3, 6, requires_grad=True)

    fft_causal_convolution(inputs, filters).square().mean().backward()

    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert filters.grad is not None and torch.isfinite(filters.grad).all()
