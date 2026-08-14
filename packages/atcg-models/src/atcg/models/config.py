"""Typed, serializable architecture specifications for controlled comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast


def _positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """Causal multi-head self-attention mixer."""

    kind: Literal["attention"] = "attention"
    n_heads: int = 8

    def __post_init__(self) -> None:
        _positive("attention n_heads", self.n_heads)


@dataclass(frozen=True, slots=True)
class HyenaSESpec:
    """Short explicit Hyena convolution mixer."""

    kind: Literal["hyena_se"] = "hyena_se"
    kernel_size: int = 7

    def __post_init__(self) -> None:
        _positive("hyena_se kernel_size", self.kernel_size)


@dataclass(frozen=True, slots=True)
class HyenaMRSpec:
    """Medium-range Hyena convolution mixer."""

    kind: Literal["hyena_mr"] = "hyena_mr"
    kernel_size: int = 128

    def __post_init__(self) -> None:
        _positive("hyena_mr kernel_size", self.kernel_size)


@dataclass(frozen=True, slots=True)
class HyenaLISpec:
    """Long implicit Hyena convolution mixer."""

    kind: Literal["hyena_li"] = "hyena_li"
    filter_hidden_size: int = 64

    def __post_init__(self) -> None:
        _positive("hyena_li filter_hidden_size", self.filter_hidden_size)


@dataclass(frozen=True, slots=True)
class TitansMemorySpec:
    """TITANS neural long-term memory used inside the standard block shell."""

    kind: Literal["titans_memory"] = "titans_memory"
    expansion_factor: int = 4
    projection_kernel_size: int = 4
    alpha_initial: float = 0.001
    eta_initial: float = 0.9
    theta_initial: float = 0.001

    def __post_init__(self) -> None:
        _positive("TITANS memory expansion_factor", self.expansion_factor)
        _positive("TITANS memory projection_kernel_size", self.projection_kernel_size)
        for name, value in (
            ("alpha_initial", self.alpha_initial),
            ("eta_initial", self.eta_initial),
            ("theta_initial", self.theta_initial),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"TITANS memory {name} must be in (0, 1)")


type MixerSpec = AttentionSpec | HyenaSESpec | HyenaMRSpec | HyenaLISpec | TitansMemorySpec
type MixerKind = Literal["attention", "hyena_se", "hyena_mr", "hyena_li", "titans_memory"]


@dataclass(frozen=True, slots=True)
class StandardBlockSpec:
    """Controlled RMSNorm/mixer/residual/SwiGLU block shell."""

    mixer: MixerSpec
    mlp_hidden_size: int
    kind: Literal["standard"] = "standard"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _positive("standard block mlp_hidden_size", self.mlp_hidden_size)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("standard block dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TitansMACBlockSpec:
    """Complete TITANS Memory-as-Context block substitution."""

    n_heads: int
    mlp_hidden_size: int
    kind: Literal["titans_mac"] = "titans_mac"
    segment_length: int = 32
    persistent_tokens: int = 4
    memory_expansion_factor: int = 4
    projection_kernel_size: int = 4
    alpha_initial: float = 0.001
    eta_initial: float = 0.9
    theta_initial: float = 0.001
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("TITANS MAC n_heads", self.n_heads),
            ("TITANS MAC mlp_hidden_size", self.mlp_hidden_size),
            ("TITANS MAC segment_length", self.segment_length),
            ("TITANS MAC persistent_tokens", self.persistent_tokens),
            ("TITANS MAC memory_expansion_factor", self.memory_expansion_factor),
            ("TITANS MAC projection_kernel_size", self.projection_kernel_size),
        ):
            _positive(name, value)
        for name, value in (
            ("alpha_initial", self.alpha_initial),
            ("eta_initial", self.eta_initial),
            ("theta_initial", self.theta_initial),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"TITANS MAC {name} must be in (0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("TITANS MAC dropout must be in [0, 1)")


type BlockSpec = StandardBlockSpec | TitansMACBlockSpec


def mixer_to_dict(spec: MixerSpec) -> dict[str, object]:
    return cast(dict[str, object], asdict(spec))


def mixer_from_dict(values: dict[str, object]) -> MixerSpec:
    kind = values.get("kind")
    if kind == "attention":
        return AttentionSpec(n_heads=_integer(values, "n_heads", 8))
    if kind == "hyena_se":
        return HyenaSESpec(kernel_size=_integer(values, "kernel_size", 7))
    if kind == "hyena_mr":
        return HyenaMRSpec(kernel_size=_integer(values, "kernel_size", 128))
    if kind == "hyena_li":
        return HyenaLISpec(filter_hidden_size=_integer(values, "filter_hidden_size", 64))
    if kind == "titans_memory":
        return TitansMemorySpec(
            expansion_factor=_integer(values, "expansion_factor", 4),
            projection_kernel_size=_integer(values, "projection_kernel_size", 4),
            alpha_initial=_number(values, "alpha_initial", 0.001),
            eta_initial=_number(values, "eta_initial", 0.9),
            theta_initial=_number(values, "theta_initial", 0.001),
        )
    raise ValueError(f"unknown mixer kind {kind!r}")


def block_to_dict(spec: BlockSpec) -> dict[str, object]:
    if isinstance(spec, StandardBlockSpec):
        return {
            "kind": spec.kind,
            "mixer": mixer_to_dict(spec.mixer),
            "mlp_hidden_size": spec.mlp_hidden_size,
            "dropout": spec.dropout,
        }
    return cast(dict[str, object], asdict(spec))


def block_from_dict(values: dict[str, object]) -> BlockSpec:
    kind = values.get("kind")
    if kind == "standard":
        raw_mixer = values.get("mixer")
        if not isinstance(raw_mixer, dict):
            raise TypeError("standard block mixer must be a mapping")
        return StandardBlockSpec(
            mixer=mixer_from_dict(cast(dict[str, object], raw_mixer)),
            mlp_hidden_size=_integer(values, "mlp_hidden_size"),
            dropout=_number(values, "dropout", 0.0),
        )
    if kind == "titans_mac":
        return TitansMACBlockSpec(
            n_heads=_integer(values, "n_heads"),
            mlp_hidden_size=_integer(values, "mlp_hidden_size"),
            segment_length=_integer(values, "segment_length", 32),
            persistent_tokens=_integer(values, "persistent_tokens", 4),
            memory_expansion_factor=_integer(values, "memory_expansion_factor", 4),
            projection_kernel_size=_integer(values, "projection_kernel_size", 4),
            alpha_initial=_number(values, "alpha_initial", 0.001),
            eta_initial=_number(values, "eta_initial", 0.9),
            theta_initial=_number(values, "theta_initial", 0.001),
            dropout=_number(values, "dropout", 0.0),
        )
    raise ValueError(f"unknown block kind {kind!r}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture of an autoregressive model with an explicit block schedule."""

    vocab_size: int
    d_model: int
    blocks: tuple[BlockSpec, ...]
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        _positive("vocab_size", self.vocab_size)
        _positive("d_model", self.d_model)
        _positive("max_seq_len", self.max_seq_len)
        if not self.blocks:
            raise ValueError("blocks must not be empty")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive")
        mac_segment_lengths: set[int] = set()
        for index, block in enumerate(self.blocks):
            n_heads = (
                block.mixer.n_heads
                if isinstance(block, StandardBlockSpec) and isinstance(block.mixer, AttentionSpec)
                else block.n_heads
                if isinstance(block, TitansMACBlockSpec)
                else None
            )
            if n_heads is not None:
                if self.d_model % n_heads != 0:
                    raise ValueError(f"block {index} d_model must be divisible by n_heads")
                if self.d_model // n_heads % 2 != 0 and isinstance(block, StandardBlockSpec):
                    raise ValueError(f"block {index} attention head dimension must be even")
            if isinstance(block, TitansMACBlockSpec) and block.segment_length > self.max_seq_len:
                raise ValueError(f"block {index} segment length exceeds max_seq_len")
            if isinstance(block, TitansMACBlockSpec):
                mac_segment_lengths.add(block.segment_length)
        if len(mac_segment_lengths) > 1:
            raise ValueError("all TITANS MAC blocks must use the same segment length")

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def is_stateful(self) -> bool:
        return any(
            isinstance(block, TitansMACBlockSpec) or isinstance(block.mixer, TitansMemorySpec)
            for block in self.blocks
        )

    @property
    def segment_length(self) -> int:
        """Required execution segment length for the model's block schedule."""

        for block in self.blocks:
            if isinstance(block, TitansMACBlockSpec):
                return block.segment_length
        return self.max_seq_len

    def to_dict(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "blocks": [block_to_dict(block) for block in self.blocks],
            "max_seq_len": self.max_seq_len,
            "norm_eps": self.norm_eps,
            "tie_embeddings": self.tie_embeddings,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ModelConfig:
        raw_blocks = values.get("blocks")
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise TypeError("blocks must be a non-empty list")
        blocks: list[BlockSpec] = []
        for raw_block in cast(list[object], raw_blocks):
            if not isinstance(raw_block, dict):
                raise TypeError("each block specification must be a mapping")
            blocks.append(block_from_dict(cast(dict[str, object], raw_block)))
        tie_embeddings = values.get("tie_embeddings", True)
        if not isinstance(tie_embeddings, bool):
            raise TypeError("tie_embeddings must be a boolean")
        return cls(
            vocab_size=_integer(values, "vocab_size"),
            d_model=_integer(values, "d_model"),
            blocks=tuple(blocks),
            max_seq_len=_integer(values, "max_seq_len", 4096),
            norm_eps=_number(values, "norm_eps", 1e-5),
            tie_embeddings=tie_embeddings,
        )


def _integer(values: dict[str, object], name: str, default: int | None = None) -> int:
    value = values.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(values: dict[str, object], name: str, default: float) -> float:
    value = values.get(name, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    return float(value)
