"""Sequence records with explicit provenance fields."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """One biological sequence and its source metadata.

    The sequence is kept as text rather than silently normalized. Tokenizers and
    transforms own normalization so experiments can record the exact policy used.
    """

    identifier: str
    sequence: str
    description: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        if not self.identifier or self.identifier.isspace():
            raise ValueError("sequence identifier must not be empty")
        if any(character.isspace() for character in self.identifier):
            raise ValueError("sequence identifier must not contain whitespace")
        if not self.sequence:
            raise ValueError(f"sequence {self.identifier!r} must not be empty")
        if any(character.isspace() for character in self.sequence):
            raise ValueError(f"sequence {self.identifier!r} must not contain whitespace")

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
