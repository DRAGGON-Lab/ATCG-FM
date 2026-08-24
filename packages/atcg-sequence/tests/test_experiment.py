from atcg.sequence import SequenceRecord, nested_stage_memberships, prepare_coordinate_splits


def test_coordinate_splits_are_disjoint_and_stream_bounded() -> None:
    source = SequenceRecord(identifier="genome", sequence="ACGT" * 100)
    prepared = prepare_coordinate_splits(
        [source],
        stream_length=64,
        minimum_stream_length=8,
    )

    assert [(value.split, value.start, value.stop) for value in prepared.coordinates] == [
        ("train", 0, 320),
        ("validation", 320, 360),
        ("test", 360, 400),
    ]
    assert all(len(record.sequence) <= 64 for record in prepared.train)
    assert sum(len(record.sequence) for record in prepared.train) == 320
    assert prepared.fingerprint() == prepared.fingerprint()


def test_coordinate_splits_support_context_guards() -> None:
    source = SequenceRecord(identifier="genome", sequence="A" * 1000)
    prepared = prepare_coordinate_splits(
        [source],
        stream_length=512,
        minimum_stream_length=1,
        guard_bases=128,
    )

    train, validation, test = prepared.coordinates
    assert train.stop + 128 == validation.start
    assert validation.stop + 128 == test.start
    assert sum(value.stop - value.start for value in prepared.coordinates) == 744


def test_coordinate_split_rejects_negative_guard() -> None:
    source = SequenceRecord(identifier="genome", sequence="ACGT")

    try:
        prepare_coordinate_splits([source], guard_bases=-1)
    except ValueError as error:
        assert str(error) == "guard_bases must be non-negative"
    else:
        raise AssertionError("negative guards must be rejected")


def test_stage_memberships_are_nested_prefixes() -> None:
    accessions = tuple(f"GCF_{index:09d}.1" for index in range(64))
    stages = nested_stage_memberships(accessions)

    assert stages["stage-1"] == accessions[:1]
    assert stages["stage-4"] == accessions[:4]
    assert stages["stage-16"] == accessions[:16]
    assert stages["stage-64"] == accessions
