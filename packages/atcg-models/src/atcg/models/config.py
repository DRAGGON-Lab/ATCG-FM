"""Validated, serializable model configuration."""

from dataclasses import asdict, dataclass
from typing import Literal, Self, cast

MixerKind = Literal["attention", "hyena_se", "hyena_mr", "hyena_li"]


@dataclass(frozen=True, slots=True)
class MixerSpec:
    """Configuration for one family of causal sequence mixer."""

    kind: MixerKind
    kernel_size: int | None = None
    filter_hidden_size: int = 64

    def __post_init__(self) -> None:
        if self.kernel_size is not None and self.kernel_size < 1:
            raise ValueError("mixer kernel_size must be positive")
        if self.filter_hidden_size < 1:
            raise ValueError("filter_hidden_size must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> Self:
        kind = values.get("kind")
        if kind not in {"attention", "hyena_se", "hyena_mr", "hyena_li"}:
            raise ValueError(f"unknown mixer kind {kind!r}")
        kernel_size = values.get("kernel_size")
        hidden_size = values.get("filter_hidden_size", 64)
        if kernel_size is not None and not isinstance(kernel_size, int):
            raise TypeError("mixer kernel_size must be an integer or null")
        if not isinstance(hidden_size, int):
            raise TypeError("filter_hidden_size must be an integer")
        return cls(
            kind=cast(MixerKind, kind),
            kernel_size=kernel_size,
            filter_hidden_size=hidden_size,
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture of an autoregressive genomic language model."""

    vocab_size: int
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 8
    mlp_hidden_size: int = 768
    max_seq_len: int = 4096
    mixer_pattern: tuple[MixerSpec, ...] = (MixerSpec("attention"),)
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "mlp_hidden_size": self.mlp_hidden_size,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in positive_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if not self.mixer_pattern:
            raise ValueError("mixer_pattern must not be empty")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive")

    @property
    def layer_mixers(self) -> tuple[MixerSpec, ...]:
        return tuple(
            self.mixer_pattern[index % len(self.mixer_pattern)] for index in range(self.n_layers)
        )

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["mixer_pattern"] = [spec.to_dict() for spec in self.mixer_pattern]
        return values

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> Self:
        raw_pattern = values.get("mixer_pattern")
        if not isinstance(raw_pattern, list) or not raw_pattern:
            raise TypeError("mixer_pattern must be a non-empty list")
        pattern: list[MixerSpec] = []
        for raw_spec in cast(list[object], raw_pattern):
            if not isinstance(raw_spec, dict):
                raise TypeError("each mixer specification must be a mapping")
            pattern.append(MixerSpec.from_dict(cast(dict[str, object], raw_spec)))

        def integer(name: str, default: int | None = None) -> int:
            value = values.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            return value

        def number(name: str, default: float) -> float:
            value = values.get(name, default)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            return float(value)

        tie_embeddings = values.get("tie_embeddings", True)
        if not isinstance(tie_embeddings, bool):
            raise TypeError("tie_embeddings must be a boolean")

        return cls(
            vocab_size=integer("vocab_size"),
            d_model=integer("d_model", 256),
            n_layers=integer("n_layers", 12),
            n_heads=integer("n_heads", 8),
            mlp_hidden_size=integer("mlp_hidden_size", 768),
            max_seq_len=integer("max_seq_len", 4096),
            mixer_pattern=tuple(pattern),
            dropout=number("dropout", 0.0),
            norm_eps=number("norm_eps", 1e-5),
            tie_embeddings=tie_embeddings,
        )
