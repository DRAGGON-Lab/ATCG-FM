import pytest
import torch

from atcg.models import (
    GenomicLanguageModel,
    ModelConfig,
    ModelState,
    attention_tiny,
    hybrid_tiny,
    titans_mac_tiny,
    titans_memory_tiny,
)
from atcg.models.titans import NeuralMemoryState

MODEL_CONFIGS = [
    attention_tiny(vocab_size=20, d_model=16, n_heads=4, n_layers=4, max_seq_len=16),
    hybrid_tiny(vocab_size=20, d_model=16, n_heads=4, n_layers=4, max_seq_len=16),
]


def _memory_state(model_state: ModelState) -> NeuralMemoryState:
    block_state = model_state.blocks[0]
    assert isinstance(block_state, NeuralMemoryState)
    return block_state


def _first_fast_weight(state: NeuralMemoryState) -> torch.Tensor:
    return next(iter(state.fast_weights.values()))


@pytest.mark.parametrize("config", MODEL_CONFIGS)
def test_language_model_returns_evaluation_ready_activations(config: ModelConfig) -> None:
    model = GenomicLanguageModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 11))

    output = model(input_ids, return_intermediate_hidden_states=True)

    assert output.logits.shape == (2, 11, config.vocab_size)
    assert output.hidden_states.shape == (2, 11, config.d_model)
    assert len(output.states) == 2
    assert output.intermediate_hidden_states is not None
    assert len(output.intermediate_hidden_states) == config.n_layers
    assert model.parameter_count() > 0
    assert model.recurrent_state_elements() == 0


def test_language_model_is_causal_across_hybrid_stack() -> None:
    torch.manual_seed(23)
    config = hybrid_tiny(vocab_size=20, d_model=16, n_heads=4, n_layers=4, max_seq_len=16)
    model = GenomicLanguageModel(config).eval()
    inputs = torch.randint(0, config.vocab_size, (1, 12))
    changed = inputs.clone()
    changed[:, 8:] = torch.randint(0, config.vocab_size, (1, 4))

    with torch.inference_mode():
        original_prefix = model(inputs).logits[:, :8]
        changed_prefix = model(changed).logits[:, :8]

    torch.testing.assert_close(original_prefix, changed_prefix, atol=1e-5, rtol=1e-5)


def test_language_model_rejects_sequences_beyond_configured_context() -> None:
    model = GenomicLanguageModel(
        attention_tiny(vocab_size=20, d_model=16, n_heads=4, n_layers=1, max_seq_len=8)
    )

    with pytest.raises(ValueError, match="exceeds"):
        model(torch.ones((1, 9), dtype=torch.long))


@pytest.mark.parametrize(
    "config",
    [
        titans_memory_tiny(
            vocab_size=20,
            d_model=8,
            n_layers=1,
            max_seq_len=4,
            expansion_factor=2,
            projection_kernel_size=2,
        ),
        titans_mac_tiny(
            vocab_size=20,
            d_model=8,
            n_layers=1,
            n_heads=2,
            segment_length=4,
            persistent_tokens=2,
            memory_expansion_factor=2,
            projection_kernel_size=2,
        ),
    ],
)
def test_stateful_candidates_update_explicit_state_and_backpropagate(config: ModelConfig) -> None:
    torch.manual_seed(31)
    model = GenomicLanguageModel(config)
    ids = torch.randint(0, config.vocab_size, (1, 4))
    initial = model.initial_state()

    first = model.forward_segment(ids, (initial,))
    second = model.forward_segment(ids, first.states)
    second.logits.square().mean().backward()

    assert model.recurrent_state_elements() > 0
    assert not torch.equal(
        _first_fast_weight(_memory_state(first.states[0])),
        _first_fast_weight(_memory_state(initial)),
    )
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mac_block_is_causal_with_adaptive_memory() -> None:
    torch.manual_seed(37)
    config = titans_mac_tiny(
        vocab_size=20,
        d_model=8,
        n_layers=1,
        n_heads=2,
        segment_length=4,
        persistent_tokens=2,
        memory_expansion_factor=2,
        projection_kernel_size=2,
    )
    model = GenomicLanguageModel(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 4))
    changed = ids.clone()
    changed[:, 2:] = torch.randint(0, config.vocab_size, (1, 2))

    original = model(ids).logits[:, :2]
    modified = model(changed).logits[:, :2]

    torch.testing.assert_close(original, modified, atol=1e-5, rtol=1e-5)


def test_memory_modes_separate_updates_from_retrieval() -> None:
    torch.manual_seed(41)
    config = titans_memory_tiny(
        vocab_size=20,
        d_model=8,
        n_layers=1,
        max_seq_len=4,
        expansion_factor=2,
        projection_kernel_size=2,
    )
    model = GenomicLanguageModel(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 4))
    initial = model.initial_state()

    adaptive = model.forward_segment(ids, (initial,), memory_mode="adaptive")
    frozen = model.forward_segment(ids, (initial,), memory_mode="frozen")
    disabled = model.forward_segment(ids, (initial,), memory_mode="disabled")

    initial_memory = _memory_state(initial)
    adaptive_memory = _memory_state(adaptive.states[0])
    frozen_memory = _memory_state(frozen.states[0])
    disabled_memory = _memory_state(disabled.states[0])
    assert not torch.equal(_first_fast_weight(adaptive_memory), _first_fast_weight(initial_memory))
    torch.testing.assert_close(
        _first_fast_weight(frozen_memory), _first_fast_weight(initial_memory)
    )
    assert not torch.equal(frozen_memory.output_query_history, initial_memory.output_query_history)
    assert disabled_memory is initial_memory
    assert not torch.equal(adaptive.logits, disabled.logits)
