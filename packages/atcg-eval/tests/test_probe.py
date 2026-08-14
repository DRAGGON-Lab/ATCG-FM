from typing import cast

import numpy as np
from gfmbench_api.tasks.base import BaseGFMModel, BaseGFMTask
from numpy.typing import NDArray

from atcg.eval import fit_frozen_probe


class _Backbone(BaseGFMModel):
    def __init__(self, device: str = "cpu") -> None:
        del device

    def infer_sequence_to_labels_probs(
        self,
        sequences: list[str],
        conditional_input: object | None = None,
    ) -> None:
        del sequences, conditional_input
        return None

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: list[str],
        ref_sequences: list[str],
        conditional_input: object | None = None,
    ) -> None:
        del variant_sequences, ref_sequences, conditional_input
        return None

    def infer_sequence_to_sequence(
        self,
        sequences: list[str],
        conditional_input: object | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        del conditional_input
        representative = np.asarray(
            [[sequence.count("A"), sequence.count("C")] for sequence in sequences],
            dtype=np.float32,
        )
        probabilities = np.full((len(sequences), len(sequences[0])), 0.25, dtype=np.float32)
        embeddings = np.repeat(representative[:, None, :], len(sequences[0]), axis=1)
        return probabilities, embeddings, representative

    def sequence_pos_to_prob_pos(
        self,
        sequences: list[str],
        pos: int,
    ) -> NDArray[np.int64]:
        return np.asarray([pos if pos < len(sequence) else -1 for sequence in sequences])

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: list[str],
        variant_pos: int,
        variant_letters: list[str],
        reference_letters: list[str],
        conditional_input: object | None = None,
    ) -> tuple[None, None]:
        del sequences, variant_pos, variant_letters, reference_letters, conditional_input
        return None, None


class _SupervisedTask:
    def get_task_attributes(self) -> dict[str, object]:
        return {
            "classification_mode": "single_label",
            "is_variant_effect_prediction": False,
            "task_type": "classification",
        }

    def get_finetune_dataset(self) -> list[tuple[str, int, NDArray[np.float64]]]:
        return [
            ("AAAA", 0, np.array([])),
            ("AAAT", 0, np.array([])),
            ("CCCC", 1, np.array([])),
            ("CCCG", 1, np.array([])),
        ]


def test_frozen_probe_fits_gfmbench_training_representations() -> None:
    model = fit_frozen_probe(
        cast(BaseGFMTask, _SupervisedTask()),
        _Backbone(),
        batch_size=2,
        seed=67,
    )

    probabilities = model.infer_sequence_to_labels_probs(["AAAA", "CCCC"])

    assert probabilities is not None
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]
