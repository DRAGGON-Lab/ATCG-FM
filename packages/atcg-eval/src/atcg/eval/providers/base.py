"""Shared GFMBench behavior for frozen runtime adapters."""

from collections.abc import Sequence

import numpy as np
from gfmbench_api.tasks.base import BaseGFMModel
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
NumericArray = NDArray[np.generic]


class FrozenGFMModel(BaseGFMModel):
    """A feature backbone with no task-specific prediction head."""

    def infer_sequence_to_labels_probs(
        self,
        sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> None:
        del sequences, conditional_input
        return None

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: list[str],
        ref_sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> None:
        del variant_sequences, ref_sequences, conditional_input
        return None

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: list[str],
        variant_pos: int,
        variant_letters: list[str],
        reference_letters: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[NumericArray | None, NumericArray | None]:
        del sequences, variant_pos, variant_letters, reference_letters, conditional_input
        return None, None


def normalize_sequences(
    sequences: Sequence[str],
    *,
    max_sequence_length: int,
    alphabet: frozenset[str],
    boundary_replacement: str,
) -> list[str]:
    """Normalize a non-empty equal-length batch and enforce a provider alphabet."""

    normalized = [
        str(sequence).upper().replace("P", boundary_replacement) for sequence in sequences
    ]
    if not normalized:
        raise ValueError("sequence batch must not be empty")
    if any(not sequence for sequence in normalized):
        raise ValueError("sequences must not be empty")
    lengths = {len(sequence) for sequence in normalized}
    if len(lengths) != 1:
        raise ValueError("GFMBench batches must contain equal-length sequences")
    sequence_length = len(normalized[0])
    if sequence_length > max_sequence_length:
        raise ValueError(
            f"sequence length {sequence_length} exceeds adapter maximum {max_sequence_length}"
        )
    for sequence in normalized:
        unexpected = sorted(set(sequence).difference(alphabet))
        if unexpected:
            raise ValueError(f"sequence contains unsupported symbols: {''.join(unexpected)}")
    return normalized


def base_position_map(sequences: Sequence[str], pos: int) -> IntArray:
    """Map a nucleotide position for base-aligned provider outputs."""

    return np.asarray(
        [pos if 0 <= pos < len(sequence) else -1 for sequence in sequences],
        dtype=np.int64,
    )
