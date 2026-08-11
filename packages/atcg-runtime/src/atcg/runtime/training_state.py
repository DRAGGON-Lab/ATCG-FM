"""Minimal checkpointable trainer progress."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Progress counters whose meaning is independent of wall-clock timing."""

    step: int = 0
    tokens_seen: int = 0

    def __post_init__(self) -> None:
        if self.step < 0 or self.tokens_seen < 0:
            raise ValueError("training counters must not be negative")
