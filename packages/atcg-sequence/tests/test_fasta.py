import pytest

from atcg.sequence import iter_fasta, parse_fasta


def test_parse_fasta_preserves_record_boundaries_and_provenance() -> None:
    records = parse_fasta(
        [">chr1 reference\n", "ACGT\n", "AC\n", ">chr2\n", "NNNN\n"],
        source="fixture.fa",
    )

    assert [record.identifier for record in records] == ["chr1", "chr2"]
    assert records[0].sequence == "ACGTAC"
    assert records[0].description == "reference"
    assert records[0].metadata == {"source": "fixture.fa"}


def test_parse_fasta_rejects_duplicate_identifiers() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_fasta([">same\n", "AC\n", ">same\n", "GT\n"])


def test_parse_fasta_rejects_sequence_before_header() -> None:
    with pytest.raises(ValueError, match="before first"):
        parse_fasta(["ACGT\n"])


def test_iter_fasta_is_lazy() -> None:
    consumed: list[str] = []

    def lines():
        for line in (">one\n", "AC\n", ">two\n", "GT\n"):
            consumed.append(line)
            yield line

    records = iter_fasta(lines())
    assert next(records).identifier == "one"
    assert consumed == [">one\n", "AC\n", ">two\n"]
    assert next(records).sequence == "GT"


def test_iter_fasta_filters_before_materializing_sequences() -> None:
    records = tuple(
        iter_fasta(
            [">skip first\n", "AAAA\n", ">keep second\n", "CCCC\n"],
            record_filter=lambda identifier, description: identifier == "keep",
        )
    )

    assert [(record.identifier, record.description, record.sequence) for record in records] == [
        ("keep", "second", "CCCC")
    ]
