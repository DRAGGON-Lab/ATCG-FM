import torch

from atcg.runtime import collate_examples
from atcg.runtime.batching import TARGET_IGNORE_ID
from atcg.sequence import LanguageModelExample


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
