"""Strict FASTA parsing without a heavyweight bioinformatics dependency."""

from collections.abc import Iterable
from pathlib import Path

from atcg.sequence.records import SequenceRecord


def parse_fasta(lines: Iterable[str], *, source: str = "") -> tuple[SequenceRecord, ...]:
    """Parse FASTA text and reject ambiguous or lossy records."""

    records: list[SequenceRecord] = []
    seen_identifiers: set[str] = set()
    identifier: str | None = None
    description = ""
    sequence_parts: list[str] = []

    def finish_record() -> None:
        nonlocal identifier, description, sequence_parts
        if identifier is None:
            return
        sequence = "".join(sequence_parts)
        metadata = {"source": source} if source else {}
        records.append(
            SequenceRecord(
                identifier=identifier,
                description=description,
                sequence=sequence,
                metadata=metadata,
            )
        )
        identifier = None
        description = ""
        sequence_parts = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish_record()
            header = line[1:].strip()
            if not header:
                raise ValueError(f"empty FASTA header at line {line_number}")
            parts = header.split(maxsplit=1)
            identifier = parts[0]
            description = parts[1] if len(parts) == 2 else ""
            if identifier in seen_identifiers:
                raise ValueError(f"duplicate FASTA identifier {identifier!r}")
            seen_identifiers.add(identifier)
        else:
            if identifier is None:
                raise ValueError(f"sequence data before first FASTA header at line {line_number}")
            if any(character.isspace() for character in line):
                raise ValueError(f"whitespace inside FASTA sequence at line {line_number}")
            sequence_parts.append(line)

    finish_record()
    return tuple(records)


def read_fasta(path: str | Path) -> tuple[SequenceRecord, ...]:
    """Read records from a UTF-8 FASTA file."""

    resolved = Path(path)
    with resolved.open(encoding="utf-8") as fasta_file:
        return parse_fasta(fasta_file, source=str(resolved))
