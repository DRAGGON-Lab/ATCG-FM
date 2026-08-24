"""Strict FASTA parsing without a heavyweight bioinformatics dependency."""

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from atcg.sequence.records import SequenceRecord


def iter_fasta(
    lines: Iterable[str],
    *,
    source: str = "",
    record_filter: Callable[[str, str], bool] | None = None,
) -> Iterator[SequenceRecord]:
    """Stream selected FASTA records while rejecting ambiguous or lossy input."""

    seen_identifiers: set[str] = set()
    identifier: str | None = None
    description = ""
    sequence_parts: list[str] = []
    selected = False

    def finish_record() -> SequenceRecord | None:
        if identifier is None:
            return None
        if not selected:
            return None
        sequence = "".join(sequence_parts)
        metadata = {"source": source} if source else {}
        return SequenceRecord(
            identifier=identifier,
            description=description,
            sequence=sequence,
            metadata=metadata,
        )

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            completed = finish_record()
            if completed is not None:
                yield completed
            header = line[1:].strip()
            if not header:
                raise ValueError(f"empty FASTA header at line {line_number}")
            parts = header.split(maxsplit=1)
            identifier = parts[0]
            description = parts[1] if len(parts) == 2 else ""
            if identifier in seen_identifiers:
                raise ValueError(f"duplicate FASTA identifier {identifier!r}")
            seen_identifiers.add(identifier)
            selected = record_filter is None or record_filter(identifier, description)
            sequence_parts = []
        else:
            if identifier is None:
                raise ValueError(f"sequence data before first FASTA header at line {line_number}")
            if any(character.isspace() for character in line):
                raise ValueError(f"whitespace inside FASTA sequence at line {line_number}")
            if selected:
                sequence_parts.append(line)

    completed = finish_record()
    if completed is not None:
        yield completed


def parse_fasta(lines: Iterable[str], *, source: str = "") -> tuple[SequenceRecord, ...]:
    """Parse all FASTA records into memory."""

    return tuple(iter_fasta(lines, source=source))


def read_fasta(path: str | Path) -> tuple[SequenceRecord, ...]:
    """Read records from a UTF-8 FASTA file."""

    resolved = Path(path)
    with resolved.open(encoding="utf-8") as fasta_file:
        return parse_fasta(fasta_file, source=str(resolved))
