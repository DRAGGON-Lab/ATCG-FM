"""Deterministic coordinate splits and streams for comparison experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from atcg.sequence.records import SequenceRecord


@dataclass(frozen=True, slots=True)
class CoordinateSplit:
    """One disjoint coordinate interval from a source sequence."""

    source_id: str
    split: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split {self.split!r}")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("coordinate split must be a non-empty positive interval")


@dataclass(frozen=True, slots=True)
class PreparedSequenceSplits:
    """Prepared streams plus the source-coordinate contract that produced them."""

    train: tuple[SequenceRecord, ...]
    validation: tuple[SequenceRecord, ...]
    test: tuple[SequenceRecord, ...]
    coordinates: tuple[CoordinateSplit, ...]

    def records(self, split: str) -> tuple[SequenceRecord, ...]:
        if split == "train":
            return self.train
        if split == "validation":
            return self.validation
        if split == "test":
            return self.test
        raise ValueError(f"unsupported split {split!r}")

    def fingerprint(self) -> str:
        """Hash identities, coordinates, and sequence bytes in stable order."""

        digest = hashlib.sha256()
        for split in ("train", "validation", "test"):
            for record in self.records(split):
                digest.update(split.encode())
                digest.update(b"\0")
                digest.update(record.identifier.encode())
                digest.update(b"\0")
                digest.update(record.sequence.encode())
                digest.update(b"\0")
        return digest.hexdigest()


def prepare_coordinate_splits(
    records: Sequence[SequenceRecord],
    *,
    stream_length: int = 16_384,
    minimum_stream_length: int = 129,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    guard_bases: int = 0,
) -> PreparedSequenceSplits:
    """Split every record by coordinate, then make bounded non-overlapping streams.

    ``guard_bases`` removes an interval between adjacent partitions.  A guard at least as
    large as the model context prevents an input window on one side of a split from
    sharing coordinates with the target context on the other side.
    """

    if stream_length < 1 or minimum_stream_length < 1:
        raise ValueError("stream lengths must be positive")
    if minimum_stream_length > stream_length:
        raise ValueError("minimum stream length cannot exceed stream length")
    if guard_bases < 0:
        raise ValueError("guard_bases must be non-negative")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test interval")

    prepared: dict[str, list[SequenceRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    coordinates: list[CoordinateSplit] = []
    for record in records:
        usable_length = len(record.sequence) - 2 * guard_bases
        if usable_length < 1:
            continue
        train_length = int(usable_length * train_fraction)
        validation_length = int(usable_length * validation_fraction)
        train_stop = train_length
        validation_start = train_stop + guard_bases
        validation_stop = validation_start + validation_length
        test_start = validation_stop + guard_bases
        intervals = (
            ("train", 0, train_stop),
            ("validation", validation_start, validation_stop),
            ("test", test_start, len(record.sequence)),
        )
        for split, start, stop in intervals:
            if stop <= start:
                continue
            coordinates.append(CoordinateSplit(record.identifier, split, start, stop))
            for chunk_start in range(start, stop, stream_length):
                chunk_stop = min(chunk_start + stream_length, stop)
                if chunk_stop - chunk_start < minimum_stream_length:
                    continue
                identifier = f"{record.identifier}__{split}__{chunk_start}_{chunk_stop}"
                metadata = {
                    **record.metadata,
                    "source_id": record.identifier,
                    "split": split,
                    "start": str(chunk_start),
                    "stop": str(chunk_stop),
                }
                prepared[split].append(
                    SequenceRecord(
                        identifier=identifier,
                        sequence=record.sequence[chunk_start:chunk_stop],
                        description=record.description,
                        metadata=metadata,
                    )
                )
    return PreparedSequenceSplits(
        train=tuple(prepared["train"]),
        validation=tuple(prepared["validation"]),
        test=tuple(prepared["test"]),
        coordinates=tuple(coordinates),
    )


def nested_stage_memberships(
    ordered_accessions: Sequence[str],
    *,
    stage_sizes: Sequence[int] = (1, 4, 16, 64),
) -> Mapping[str, tuple[str, ...]]:
    """Return nested accession prefixes named ``stage-<size>``."""

    if not stage_sizes or any(size < 1 for size in stage_sizes):
        raise ValueError("stage sizes must be positive")
    if tuple(sorted(set(stage_sizes))) != tuple(stage_sizes):
        raise ValueError("stage sizes must be strictly increasing")
    if len(ordered_accessions) < stage_sizes[-1]:
        raise ValueError("not enough accessions for the largest stage")
    if len(set(ordered_accessions)) != len(ordered_accessions):
        raise ValueError("ordered accessions must be unique")
    return {f"stage-{size}": tuple(ordered_accessions[:size]) for size in stage_sizes}


def write_fasta_lines(records: Iterable[SequenceRecord], *, width: int = 80) -> Iterator[str]:
    """Yield normalized FASTA lines without owning filesystem output."""

    if width < 1:
        raise ValueError("FASTA width must be positive")
    for record in records:
        yield f">{record.identifier}\n"
        for start in range(0, len(record.sequence), width):
            yield record.sequence[start : start + width] + "\n"
