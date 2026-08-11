"""Small reference presets suitable for correctness and local experiments."""

from atcg.models.config import MixerSpec, ModelConfig


def attention_tiny(
    vocab_size: int,
    *,
    max_seq_len: int = 4096,
    d_model: int = 256,
    n_layers: int = 12,
    n_heads: int = 8,
) -> ModelConfig:
    """Return a conventional attention-only control configuration."""

    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_hidden_size=3 * d_model,
        max_seq_len=max_seq_len,
        mixer_pattern=(MixerSpec("attention"),),
    )


def hybrid_tiny(
    vocab_size: int,
    *,
    max_seq_len: int = 4096,
    d_model: int = 256,
    n_layers: int = 12,
    n_heads: int = 8,
) -> ModelConfig:
    """Return an Evo 2-inspired SE-MR-LI-attention research configuration."""

    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_hidden_size=3 * d_model,
        max_seq_len=max_seq_len,
        mixer_pattern=(
            MixerSpec("hyena_se", kernel_size=7),
            MixerSpec("hyena_mr", kernel_size=128),
            MixerSpec("hyena_li", filter_hidden_size=64),
            MixerSpec("attention"),
        ),
    )
