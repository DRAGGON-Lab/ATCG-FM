import torch

from atcg.runtime import collate_examples, collate_horizons
from atcg.runtime.batching import TARGET_IGNORE_ID
from atcg.sequence import LanguageModelExample, LanguageModelHorizon


def test_collate_right_pads_inputs_and_ignores_padding_targets() -> None:
    batch = collate_examples(
        [
            LanguageModelExample((1, 2, 3), (2, 3, 4), "long", 0),
            LanguageModelExample((5,), (6,), "short", 0),
        ],
        pad_id=9,
    )

    torch.testing.assert_close(batch.input_ids, torch.tensor([[1, 2, 3], [5, 9, 9]]))
    torch.testing.assert_close(
        batch.target_ids,
        torch.tensor([[2, 3, 4], [6, TARGET_IGNORE_ID, TARGET_IGNORE_ID]]),
    )
    assert batch.token_count == 4


def test_collate_horizons_pads_segments_without_losing_stream_metadata() -> None:
    batch = collate_horizons(
        [
            LanguageModelHorizon(
                segments=(LanguageModelExample((1, 2), (2, 3), "a", 0),),
                stream_id="a",
                stream_start=True,
                stream_end=False,
            ),
            LanguageModelHorizon(
                segments=(
                    LanguageModelExample((4, 5, 6), (5, 6, 7), "b", 0),
                    LanguageModelExample((7,), (8,), "b", 3),
                ),
                stream_id="b",
                stream_start=True,
                stream_end=True,
            ),
        ],
        pad_id=9,
        segment_length=3,
    )

    assert batch.input_ids.shape == (2, 2, 3)
    assert batch.stream_ids == ("a", "b")
    assert batch.stream_starts == (True, True)
    assert batch.stream_ends == (False, True)
    assert batch.token_count == 6
    assert not batch.valid_mask[0, 1].any()
