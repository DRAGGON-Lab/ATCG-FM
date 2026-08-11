import pytest

from atcg.models import MixerSpec, ModelConfig, hybrid_tiny


def test_hybrid_pattern_repeats_to_requested_layer_count() -> None:
    config = hybrid_tiny(vocab_size=20, d_model=32, n_heads=4, n_layers=6)

    assert [spec.kind for spec in config.layer_mixers] == [
        "hyena_se",
        "hyena_mr",
        "hyena_li",
        "attention",
        "hyena_se",
        "hyena_mr",
    ]


def test_config_round_trip_preserves_mixer_semantics() -> None:
    original = hybrid_tiny(vocab_size=20, d_model=32, n_heads=4, n_layers=4)

    restored = ModelConfig.from_dict(original.to_dict())

    assert restored == original


def test_config_rejects_invalid_rotary_head_dimension() -> None:
    with pytest.raises(ValueError, match="head dimension"):
        ModelConfig(vocab_size=20, d_model=12, n_heads=4)


def test_mixer_spec_rejects_empty_filter() -> None:
    with pytest.raises(ValueError, match="positive"):
        MixerSpec("hyena_se", kernel_size=0)
