import pytest
from atcg.sequence import parse_fasta


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
