from atcg.sequence import (
    CausalWindowDataset,
    FixedAlphabetTokenizer,
    SequenceRecord,
)


def test_causal_windows_shift_targets_without_crossing_records() -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGTN")
    dataset = CausalWindowDataset(
        [
            SequenceRecord(identifier="first", sequence="ACGTA"),
            SequenceRecord(identifier="second", sequence="TT"),
        ],
        tokenizer,
        context_length=3,
        stride=3,
        include_eos=True,
    )

    examples = list(dataset)

    assert [example.source_id for example in examples] == ["first", "first", "second"]
    assert tokenizer.decode(examples[0].input_ids) == "ACG"
    assert tokenizer.decode(examples[0].target_ids) == "CGT"
    assert examples[1].target_ids[-1] == tokenizer.eos_id
    assert examples[2].target_ids[-1] == tokenizer.eos_id


def test_reverse_complement_windows_are_explicit_examples() -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT")
    dataset = CausalWindowDataset(
        [SequenceRecord(identifier="source", sequence="AACG")],
        tokenizer,
        context_length=3,
        include_eos=False,
        include_reverse_complements=True,
    )

    forward, reverse = dataset

    assert tokenizer.decode(forward.input_ids) == "AAC"
    assert tokenizer.decode(reverse.input_ids) == "CGT"
    assert not forward.reverse_complemented
    assert reverse.reverse_complemented
