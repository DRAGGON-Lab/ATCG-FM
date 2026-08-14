"""Framework-neutral datasets for autoregressive sequence modeling."""

from collections.abc import Iterator, Sequence
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
    token_offset: int
    reverse_complemented: bool = False

    def __post_init__(self) -> None:
        if not self.input_ids:
            raise ValueError("language-model example must contain at least one input token")
        if len(self.input_ids) != len(self.target_ids):
            raise ValueError("input and target lengths must match")


@dataclass(frozen=True, slots=True)
class LanguageModelHorizon:
    """Consecutive segments from one logical stream for truncated backpropagation."""

    segments: tuple[LanguageModelExample, ...]
    stream_id: str
    stream_start: bool
    stream_end: bool

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("language-model horizon must contain at least one segment")
        if any(segment.source_id != self.stream_id for segment in self.segments):
            raise ValueError("all horizon segments must belong to its stream")
        expected_offset = self.segments[0].token_offset
        for segment in self.segments:
            if segment.token_offset != expected_offset:
                raise ValueError("horizon segments must be contiguous and ordered")
            expected_offset += len(segment.input_ids)


@dataclass(frozen=True, slots=True)
class _Window:
    record_index: int
    token_offset: int
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
        include_bos: bool = True,
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
        self._include_bos = include_bos
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
        token_ids = self._tokenizer.encode(
            sequence,
            add_bos=self._include_bos,
            add_eos=self._include_eos,
        )
        segment = token_ids[window.token_offset : window.token_offset + self._context_length + 1]
        return LanguageModelExample(
            input_ids=tuple(segment[:-1]),
            target_ids=tuple(segment[1:]),
            source_id=record.identifier,
            token_offset=window.token_offset,
            reverse_complemented=window.reverse_complemented,
        )

    def _build_index(self, include_reverse_complements: bool) -> tuple[_Window, ...]:
        windows: list[_Window] = []
        orientations = (False, True) if include_reverse_complements else (False,)
        for record_index, record in enumerate(self._records):
            token_count = len(
                self._tokenizer.encode(
                    record.sequence,
                    add_bos=self._include_bos,
                    add_eos=self._include_eos,
                )
            )
            for reverse_complemented in orientations:
                for token_offset in range(0, max(token_count - 1, 0), self._stride):
                    available = token_count - token_offset
                    if available < 2:
                        continue
                    if self._drop_incomplete and available < self._context_length + 1:
                        continue
                    windows.append(
                        _Window(
                            record_index=record_index,
                            token_offset=token_offset,
                            reverse_complemented=reverse_complemented,
                        )
                    )
        return tuple(windows)


class OrderedCausalStreamDataset(Sequence[LanguageModelHorizon]):
    """Non-overlapping, source-ordered segments grouped by gradient horizon."""

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        tokenizer: Tokenizer,
        *,
        segment_length: int,
        gradient_horizon: int,
        include_bos: bool = True,
        include_eos: bool = True,
    ) -> None:
        if segment_length < 1 or gradient_horizon < 1:
            raise ValueError("segment_length and gradient_horizon must be positive")
        self.segment_length = segment_length
        self.gradient_horizon = gradient_horizon
        self._streams = tuple(
            self._build_stream(
                record,
                tokenizer,
                include_bos=include_bos,
                include_eos=include_eos,
            )
            for record in records
        )
        self._streams = tuple(stream for stream in self._streams if stream)
        self._horizons = tuple(horizon for stream in self._streams for horizon in stream)

    def _build_stream(
        self,
        record: SequenceRecord,
        tokenizer: Tokenizer,
        *,
        include_bos: bool,
        include_eos: bool,
    ) -> tuple[LanguageModelHorizon, ...]:
        token_ids = tokenizer.encode(
            record.sequence,
            add_bos=include_bos,
            add_eos=include_eos,
        )
        segments: list[LanguageModelExample] = []
        for offset in range(0, max(len(token_ids) - 1, 0), self.segment_length):
            values = token_ids[offset : offset + self.segment_length + 1]
            if len(values) < 2:
                continue
            segments.append(
                LanguageModelExample(
                    input_ids=tuple(values[:-1]),
                    target_ids=tuple(values[1:]),
                    source_id=record.identifier,
                    token_offset=offset,
                )
            )
        horizons: list[LanguageModelHorizon] = []
        for start in range(0, len(segments), self.gradient_horizon):
            values = tuple(segments[start : start + self.gradient_horizon])
            horizons.append(
                LanguageModelHorizon(
                    segments=values,
                    stream_id=record.identifier,
                    stream_start=start == 0,
                    stream_end=start + len(values) == len(segments),
                )
            )
        return tuple(horizons)

    def __len__(self) -> int:
        return len(self._horizons)

    @overload
    def __getitem__(self, index: int) -> LanguageModelHorizon: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[LanguageModelHorizon]: ...

    def __getitem__(
        self, index: int | slice
    ) -> LanguageModelHorizon | Sequence[LanguageModelHorizon]:
        return self._horizons[index]

    def iter_batches(self, batch_size: int) -> Iterator[tuple[LanguageModelHorizon, ...]]:
        """Schedule at most one ordered horizon from each active stream per batch."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        next_stream = 0
        active: list[tuple[tuple[LanguageModelHorizon, ...], int]] = []
        while len(active) < batch_size and next_stream < len(self._streams):
            active.append((self._streams[next_stream], 0))
            next_stream += 1
        while active:
            yield tuple(stream[index] for stream, index in active)
            replacements: list[tuple[tuple[LanguageModelHorizon, ...], int]] = []
            for stream, index in active:
                if index + 1 < len(stream):
                    replacements.append((stream, index + 1))
                elif next_stream < len(self._streams):
                    replacements.append((self._streams[next_stream], 0))
                    next_stream += 1
            active = replacements
