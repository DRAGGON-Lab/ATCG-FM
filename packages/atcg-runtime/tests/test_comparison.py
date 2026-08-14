import pytest

from atcg.models import (
    AttentionSpec,
    GenomicLanguageModel,
    HyenaSESpec,
    ModelConfig,
    StandardBlockSpec,
    titans_mac_tiny,
)
from atcg.runtime import (
    ComparisonCandidate,
    ComparisonInvariants,
    ComparisonPlan,
)


def _invariants() -> ComparisonInvariants:
    return ComparisonInvariants(
        tokenizer_id="iupac_v1",
        dataset_split="pretrain_v1",
        training_tokens=10_000,
        optimizer_protocol="adamw_v1",
        segment_length=8,
        gradient_horizon=1,
    )


def _mixer_config(mixer: AttentionSpec | HyenaSESpec) -> ModelConfig:
    return ModelConfig(
        vocab_size=20,
        d_model=16,
        max_seq_len=8,
        blocks=(StandardBlockSpec(mixer, mlp_hidden_size=48),),
    )


def test_mixer_comparison_manifest_records_static_and_recurrent_capacity() -> None:
    attention = _mixer_config(AttentionSpec(n_heads=4))
    hyena = _mixer_config(HyenaSESpec(kernel_size=5))
    plan = ComparisonPlan(
        comparison_id="mixer-small",
        substitution_unit="mixer",
        invariants=_invariants(),
        candidates=(
            ComparisonCandidate("attention", attention),
            ComparisonCandidate("hyena", hyena),
        ),
    )

    manifest = plan.manifest(
        {
            "attention": GenomicLanguageModel(attention),
            "hyena": GenomicLanguageModel(hyena),
        }
    )

    assert manifest["substitution_unit"] == "mixer"
    rows = manifest["candidates"]
    assert isinstance(rows, list)
    assert all(row["recurrent_state_elements_per_stream"] == 0 for row in rows)


def test_mixer_comparison_rejects_a_composite_mac_block() -> None:
    attention = _mixer_config(AttentionSpec(n_heads=4))
    mac = titans_mac_tiny(
        vocab_size=20,
        d_model=16,
        n_layers=1,
        n_heads=4,
        segment_length=8,
        persistent_tokens=2,
    )

    with pytest.raises(ValueError, match="only standard blocks"):
        ComparisonPlan(
            comparison_id="invalid-mixer-claim",
            substitution_unit="mixer",
            invariants=_invariants(),
            candidates=(
                ComparisonCandidate("attention", attention),
                ComparisonCandidate("mac", mac),
            ),
        )


def test_parameter_tolerance_is_checked_on_concrete_models() -> None:
    attention = _mixer_config(AttentionSpec(n_heads=4))
    hyena = _mixer_config(HyenaSESpec(kernel_size=5))
    plan = ComparisonPlan(
        comparison_id="exact-parameter-match",
        substitution_unit="block",
        invariants=_invariants(),
        candidates=(
            ComparisonCandidate("attention", attention),
            ComparisonCandidate("hyena", hyena),
        ),
        max_parameter_delta_fraction=0.0,
    )

    with pytest.raises(ValueError, match="parameter delta"):
        plan.manifest(
            {
                "attention": GenomicLanguageModel(attention),
                "hyena": GenomicLanguageModel(hyena),
            }
        )
