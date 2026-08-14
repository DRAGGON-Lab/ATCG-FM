"""Small presets for controlled mixer and whole-block comparisons."""

from atcg.models.config import (
    AttentionSpec,
    HyenaLISpec,
    HyenaMRSpec,
    HyenaSESpec,
    ModelConfig,
    StandardBlockSpec,
    TitansMACBlockSpec,
    TitansMemorySpec,
)


def attention_tiny(
    vocab_size: int,
    *,
    max_seq_len: int = 4096,
    d_model: int = 256,
    n_layers: int = 12,
    n_heads: int = 8,
) -> ModelConfig:
    """Conventional attention-only mixer control."""

    block = StandardBlockSpec(
        mixer=AttentionSpec(n_heads=n_heads),
        mlp_hidden_size=3 * d_model,
    )
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        blocks=(block,) * n_layers,
        max_seq_len=max_seq_len,
    )


def hybrid_tiny(
    vocab_size: int,
    *,
    max_seq_len: int = 4096,
    d_model: int = 256,
    n_layers: int = 12,
    n_heads: int = 8,
) -> ModelConfig:
    """Evo 2-inspired SE-MR-LI-attention mixer schedule."""

    pattern = (
        StandardBlockSpec(HyenaSESpec(kernel_size=7), 3 * d_model),
        StandardBlockSpec(HyenaMRSpec(kernel_size=128), 3 * d_model),
        StandardBlockSpec(HyenaLISpec(filter_hidden_size=64), 3 * d_model),
        StandardBlockSpec(AttentionSpec(n_heads=n_heads), 3 * d_model),
    )
    blocks = tuple(pattern[index % len(pattern)] for index in range(n_layers))
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        blocks=blocks,
        max_seq_len=max_seq_len,
    )


def titans_memory_tiny(
    vocab_size: int,
    *,
    max_seq_len: int = 32,
    d_model: int = 128,
    n_layers: int = 2,
    expansion_factor: int = 2,
    projection_kernel_size: int = 4,
) -> ModelConfig:
    """TITANS neural memory isolated in the standard mixer shell."""

    block = StandardBlockSpec(
        mixer=TitansMemorySpec(
            expansion_factor=expansion_factor,
            projection_kernel_size=projection_kernel_size,
        ),
        mlp_hidden_size=3 * d_model,
    )
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        blocks=(block,) * n_layers,
        max_seq_len=max_seq_len,
    )


def titans_mac_tiny(
    vocab_size: int,
    *,
    segment_length: int = 32,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    persistent_tokens: int = 4,
    memory_expansion_factor: int = 2,
    projection_kernel_size: int = 4,
) -> ModelConfig:
    """Complete TITANS Memory-as-Context whole-block substitution."""

    block = TitansMACBlockSpec(
        n_heads=n_heads,
        mlp_hidden_size=3 * d_model,
        segment_length=segment_length,
        persistent_tokens=persistent_tokens,
        memory_expansion_factor=memory_expansion_factor,
        projection_kernel_size=projection_kernel_size,
    )
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        blocks=(block,) * n_layers,
        max_seq_len=segment_length,
    )
