import pytest
from atcg.sequence import ByteTokenizer, FixedAlphabetTokenizer


def test_byte_tokenizer_round_trip_and_special_tokens() -> None:
    tokenizer = ByteTokenizer()

    encoded = tokenizer.encode("acgtn", add_bos=True, add_eos=True)

    assert encoded[0] == tokenizer.bos_id
    assert encoded[-1] == tokenizer.eos_id
    assert tokenizer.decode(encoded) == "ACGTN"


def test_byte_tokenizer_rejects_non_ascii_text() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        ByteTokenizer().encode("ACGµ")


def test_fixed_alphabet_tokenizer_is_strict_by_default() -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGTN")

    assert tokenizer.decode(tokenizer.encode("acgtn")) == "ACGTN"
    with pytest.raises(ValueError, match="offset 2"):
        tokenizer.encode("ACX")


def test_fixed_alphabet_can_represent_unknown_characters_explicitly() -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT", strict=False)

    assert tokenizer.decode(tokenizer.encode("ACX")) == "AC?"
