import pytest
import torch

from atcg.models import AttentionSpec, HyenaLISpec, HyenaMRSpec, HyenaSESpec, MixerSpec
from atcg.models.mixers import build_mixer

STATELESS_SPECS: tuple[MixerSpec, ...] = (
    AttentionSpec(n_heads=4),
    HyenaSESpec(kernel_size=5),
    HyenaMRSpec(kernel_size=5),
    HyenaLISpec(filter_hidden_size=8),
)


@pytest.mark.parametrize("spec", STATELESS_SPECS)
def test_reference_mixers_preserve_shape_and_backpropagate(spec: MixerSpec) -> None:
    mixer = build_mixer(spec, d_model=16, max_seq_len=16, dropout=0.0)
    inputs = torch.randn(2, 11, 16, requires_grad=True)

    outputs = mixer(inputs)
    outputs.square().mean().backward()

    assert outputs.shape == inputs.shape
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in mixer.parameters())


@pytest.mark.parametrize("spec", STATELESS_SPECS)
def test_reference_mixers_do_not_leak_future_inputs(spec: MixerSpec) -> None:
    torch.manual_seed(19)
    mixer = build_mixer(spec, d_model=16, max_seq_len=16, dropout=0.0).eval()
    inputs = torch.randn(1, 12, 16)
    changed = inputs.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:])

    with torch.inference_mode():
        original_prefix = mixer(inputs)[:, :7]
        changed_prefix = mixer(changed)[:, :7]

    torch.testing.assert_close(original_prefix, changed_prefix, atol=1e-5, rtol=1e-5)
