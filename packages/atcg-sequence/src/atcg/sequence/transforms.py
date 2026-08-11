"""Biological sequence transforms."""

_IUPAC_DNA_ALPHABET = frozenset("ACGTRYSWKMBDHVN-")

_COMPLEMENT = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
        "B": "V",
        "V": "B",
        "D": "H",
        "H": "D",
        "N": "N",
        "-": "-",
    }
)


def reverse_complement(sequence: str) -> str:
    """Return the uppercase IUPAC DNA reverse complement.

    Unknown characters are rejected rather than passed through, because a silent
    non-biological transform can contaminate an augmentation experiment.
    """

    normalized = sequence.upper()
    invalid = sorted(set(normalized) - _IUPAC_DNA_ALPHABET)
    if invalid:
        rendered = ", ".join(repr(character) for character in invalid)
        raise ValueError(f"cannot complement non-IUPAC characters: {rendered}")
    return normalized.translate(_COMPLEMENT)[::-1]
