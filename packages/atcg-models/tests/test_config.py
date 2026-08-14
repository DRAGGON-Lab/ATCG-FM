import pytest

from atcg.models import (
    AttentionSpec,
    HyenaSESpec,
    ModelConfig,
    StandardBlockSpec,
    TitansMACBlockSpec,
    hybrid_tiny,
)


def test_hybrid_schedule_repeats_to_requested_layer_count() -> None:
    config = hybrid_tiny(vocab_size=20, d_model=32, n_heads=4, n_layers=6)

    mixer_kinds = [
        block.mixer.kind for block in config.blocks if isinstance(block, StandardBlockSpec)
    ]
    assert mixer_kinds == [
        "hyena_se",
        "hyena_mr",
        "hyena_li",
        "attention",
        "hyena_se",
        "hyena_mr",
    ]


def test_config_round_trip_preserves_block_and_mixer_semantics() -> None:
    original = ModelConfig(
        vocab_size=20,
        d_model=32,
        max_seq_len=16,
        blocks=(
            StandardBlockSpec(HyenaSESpec(kernel_size=5), mlp_hidden_size=96),
            TitansMACBlockSpec(
                n_heads=4,
                mlp_hidden_size=96,
                segment_length=16,
                persistent_tokens=2,
            ),
        ),
    )

    restored = ModelConfig.from_dict(original.to_dict())

    assert restored == original
    assert restored.is_stateful


def test_config_rejects_invalid_rotary_head_dimension() -> None:
    with pytest.raises(ValueError, match="head dimension"):
        ModelConfig(
            vocab_size=20,
            d_model=12,
            blocks=(StandardBlockSpec(AttentionSpec(n_heads=4), mlp_hidden_size=36),),
        )


def test_typed_mixer_spec_rejects_empty_filter() -> None:
    with pytest.raises(ValueError, match="positive"):
        HyenaSESpec(kernel_size=0)
