from atcg.sequence import (
    CausalWindowDataset,
    FixedAlphabetTokenizer,
    OrderedCausalStreamDataset,
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
    assert examples[0].input_ids[0] == tokenizer.bos_id
    assert tokenizer.decode(examples[0].input_ids) == "AC"
    assert tokenizer.decode(examples[0].target_ids) == "ACG"
    assert examples[1].target_ids[-1] == tokenizer.eos_id
    assert examples[2].target_ids[-1] == tokenizer.eos_id


def test_reverse_complement_windows_are_explicit_examples() -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT")
    dataset = CausalWindowDataset(
        [SequenceRecord(identifier="source", sequence="AACG")],
        tokenizer,
        context_length=3,
        include_bos=False,
        include_eos=False,
        include_reverse_complements=True,
    )

    forward, reverse = dataset

    assert tokenizer.decode(forward.input_ids) == "AAC"
    assert tokenizer.decode(reverse.input_ids) == "CGT"
    assert not forward.reverse_complemented
    assert reverse.reverse_complemented


def test_ordered_stream_batches_preserve_identity_order_and_cross_segment_labels() -> None:
    tokenizer = FixedAlphabetTokenizer(alphabet="ACGT")
    dataset = OrderedCausalStreamDataset(
        [
            SequenceRecord(identifier="a", sequence="ACGTACGTACGT"),
            SequenceRecord(identifier="b", sequence="TGCATGCATGCA"),
            SequenceRecord(identifier="c", sequence="AAAACCCCGGGG"),
        ],
        tokenizer,
        segment_length=4,
        gradient_horizon=2,
        include_bos=False,
        include_eos=False,
    )

    batches = list(dataset.iter_batches(batch_size=2))

    assert [tuple(horizon.stream_id for horizon in batch) for batch in batches] == [
        ("a", "b"),
        ("a", "b"),
        ("c",),
        ("c",),
    ]
    first = batches[0][0]
    assert first.stream_start and not first.stream_end
    assert first.segments[0].target_ids[-1] == first.segments[1].input_ids[0]
    assert batches[1][0].stream_end
