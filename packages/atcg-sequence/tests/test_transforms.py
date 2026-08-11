import pytest
from atcg.sequence import reverse_complement


def test_reverse_complement_is_an_involution_for_iupac_dna() -> None:
    sequence = "ACGTRYSWKMBDHVN-"

    assert reverse_complement(reverse_complement(sequence)) == sequence


def test_reverse_complement_rejects_unknown_symbols() -> None:
    with pytest.raises(ValueError, match="'X'"):
        reverse_complement("ACX")
