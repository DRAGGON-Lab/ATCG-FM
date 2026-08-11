"""Framework-neutral datasets for autoregressive sequence modeling."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

from atcg.sequence.records import SequenceRecord
from atcg.sequence.tokenizers import Tokenizer
from atcg.sequence.transforms import reverse_complement


@dataclass(frozen=True, slots=True)
class LanguageModelExample:
    """One shifted causal language-model example."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    source_id: str
    offset: int
    reverse_complemented: bool = False

    def __post_init__(self) -> None:
        if not self.input_ids:
            raise ValueError("language-model example must contain at least one input token")
        if len(self.input_ids) != len(self.target_ids):
            raise ValueError("input and target lengths must match")


@dataclass(frozen=True, slots=True)
class _Window:
    record_index: int
    offset: int
    reverse_complemented: bool


class CausalWindowDataset(Sequence[LanguageModelExample]):
    """Map-style windows that never cross biological record boundaries."""

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        tokenizer: Tokenizer,
        *,
        context_length: int,
        stride: int | None = None,
        include_eos: bool = True,
        include_reverse_complements: bool = False,
        drop_incomplete: bool = False,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        resolved_stride = context_length if stride is None else stride
        if resolved_stride < 1:
            raise ValueError("stride must be positive")

        self._records = tuple(records)
        self._tokenizer = tokenizer
        self._context_length = context_length
        self._stride = resolved_stride
        self._include_eos = include_eos
        self._drop_incomplete = drop_incomplete
        self._windows = self._build_index(include_reverse_complements)

    def __len__(self) -> int:
        return len(self._windows)

    @overload
    def __getitem__(self, index: int) -> LanguageModelExample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[LanguageModelExample]: ...

    def __getitem__(
        self, index: int | slice
    ) -> LanguageModelExample | Sequence[LanguageModelExample]:
        if isinstance(index, slice):
            return [self[item_index] for item_index in range(*index.indices(len(self)))]

        window = self._windows[index]
        record = self._records[window.record_index]
        sequence = (
            reverse_complement(record.sequence) if window.reverse_complemented else record.sequence
        )
        token_ids = self._tokenizer.encode(sequence, add_eos=self._include_eos)
        segment = token_ids[window.offset : window.offset + self._context_length + 1]
        return LanguageModelExample(
            input_ids=tuple(segment[:-1]),
            target_ids=tuple(segment[1:]),
            source_id=record.identifier,
            offset=window.offset,
            reverse_complemented=window.reverse_complemented,
        )

    def _build_index(self, include_reverse_complements: bool) -> tuple[_Window, ...]:
        windows: list[_Window] = []
        orientations = (False, True) if include_reverse_complements else (False,)
        for record_index, record in enumerate(self._records):
            token_count = len(self._tokenizer.encode(record.sequence, add_eos=self._include_eos))
            for reverse_complemented in orientations:
                for offset in range(0, max(token_count - 1, 0), self._stride):
                    available = token_count - offset
                    if available < 2:
                        continue
                    if self._drop_incomplete and available < self._context_length + 1:
                        continue
                    windows.append(
                        _Window(
                            record_index=record_index,
                            offset=offset,
                            reverse_complemented=reverse_complemented,
                        )
                    )
        return tuple(windows)
