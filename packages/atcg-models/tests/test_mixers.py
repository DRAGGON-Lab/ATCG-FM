import pytest
import torch

from atcg.models import MixerKind, MixerSpec
from atcg.models.mixers import build_mixer


@pytest.mark.parametrize("kind", ["attention", "hyena_se", "hyena_mr", "hyena_li"])
def test_reference_mixers_preserve_shape_and_backpropagate(kind: MixerKind) -> None:
    spec = MixerSpec(kind=kind, kernel_size=5)
    mixer = build_mixer(spec, d_model=16, n_heads=4, max_seq_len=16, dropout=0.0)
    inputs = torch.randn(2, 11, 16, requires_grad=True)

    outputs = mixer(inputs)
    outputs.square().mean().backward()

    assert outputs.shape == inputs.shape
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in mixer.parameters())


@pytest.mark.parametrize("kind", ["attention", "hyena_se", "hyena_mr", "hyena_li"])
def test_reference_mixers_do_not_leak_future_inputs(kind: MixerKind) -> None:
    torch.manual_seed(19)
    spec = MixerSpec(kind=kind, kernel_size=5)
    mixer = build_mixer(spec, d_model=16, n_heads=4, max_seq_len=16, dropout=0.0).eval()
    inputs = torch.randn(1, 12, 16)
    changed = inputs.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:])

    with torch.inference_mode():
        original_prefix = mixer(inputs)[:, :7]
        changed_prefix = mixer(changed)[:, :7]

    torch.testing.assert_close(original_prefix, changed_prefix, atol=1e-5, rtol=1e-5)
