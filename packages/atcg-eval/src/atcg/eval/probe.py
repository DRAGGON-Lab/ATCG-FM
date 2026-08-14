"""Frozen representation probes fitted from GFMBench task datasets."""

from collections.abc import Sequence

import numpy as np
import torch
from gfmbench_api.tasks.base import BaseGFMModel, BaseGFMTask
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

FloatArray = NDArray[np.float32]
NumericArray = NDArray[np.generic]


class FrozenProbeGFMModel(BaseGFMModel):
    """Compose a frozen GFM backbone with one fitted logistic classifier."""

    def __init__(
        self,
        backbone: BaseGFMModel,
        classifier: LogisticRegression,
        *,
        variant_pairs: bool,
        device: str = "cpu",
    ) -> None:
        del device
        self.backbone = backbone
        self.classifier = classifier
        self.variant_pairs = variant_pairs

    def eval(self) -> "FrozenProbeGFMModel":
        if hasattr(self.backbone, "eval"):
            self.backbone.eval()
        return self

    def infer_sequence_to_labels_probs(
        self,
        sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> FloatArray | None:
        if self.variant_pairs:
            return None
        return self.classifier.predict_proba(
            _single_sequence_features(self.backbone, sequences, conditional_input)
        ).astype(np.float32)

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: list[str],
        ref_sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> FloatArray | None:
        if not self.variant_pairs:
            return None
        return self.classifier.predict_proba(
            _variant_pair_features(
                self.backbone,
                variant_sequences,
                ref_sequences,
                conditional_input,
            )
        ).astype(np.float32)

    def infer_sequence_to_sequence(
        self,
        sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[NumericArray | None, NumericArray | None, NumericArray | None]:
        return self.backbone.infer_sequence_to_sequence(sequences, conditional_input)

    def sequence_pos_to_prob_pos(self, sequences: list[str], pos: int) -> NumericArray:
        return self.backbone.sequence_pos_to_prob_pos(sequences, pos)

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: list[str],
        variant_pos: int,
        variant_letters: list[str],
        reference_letters: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[NumericArray | None, NumericArray | None]:
        return self.backbone.infer_masked_sequence_to_token_probs(
            sequences,
            variant_pos,
            variant_letters,
            reference_letters,
            conditional_input,
        )


def fit_frozen_probe(
    task: BaseGFMTask,
    backbone: BaseGFMModel,
    *,
    batch_size: int,
    seed: int,
) -> FrozenProbeGFMModel:
    """Fit one deterministic logistic probe without changing backbone parameters."""

    attributes = task.get_task_attributes()
    if attributes.get("task_type") != "classification":
        raise ValueError("frozen probes currently support classification tasks only")
    if attributes.get("classification_mode") != "single_label":
        raise ValueError("frozen probes currently support single-label tasks only")
    dataset = task.get_finetune_dataset()
    if dataset is None:
        raise ValueError("supervised task does not expose a fine-tuning dataset")
    variant_pairs = bool(attributes.get("is_variant_effect_prediction"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feature_batches: list[FloatArray] = []
    label_batches: list[NDArray[np.int64]] = []
    for raw_batch in loader:
        batch = tuple(raw_batch)
        if variant_pairs:
            variant_sequences, reference_sequences, labels, conditional_input = batch
            features = _variant_pair_features(
                backbone,
                list(variant_sequences),
                list(reference_sequences),
                conditional_input,
            )
        else:
            sequences, labels, conditional_input = batch
            features = _single_sequence_features(
                backbone,
                list(sequences),
                conditional_input,
            )
        feature_batches.append(features)
        label_batches.append(_labels(labels))

    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    if np.unique(labels).size < 2:
        raise ValueError("probe training data must contain at least two classes")
    classifier = LogisticRegression(
        C=1.0,
        max_iter=5_000,
        random_state=seed,
        solver="lbfgs",
    )
    classifier.fit(features, labels)
    return FrozenProbeGFMModel(
        backbone,
        classifier,
        variant_pairs=variant_pairs,
    ).eval()


def _single_sequence_features(
    backbone: BaseGFMModel,
    sequences: list[str],
    conditional_input: object | None,
) -> FloatArray:
    _, _, representative = backbone.infer_sequence_to_sequence(sequences, conditional_input)
    if representative is None:
        raise ValueError("backbone does not provide sequence representations")
    return np.asarray(representative, dtype=np.float32)


def _variant_pair_features(
    backbone: BaseGFMModel,
    variant_sequences: list[str],
    reference_sequences: list[str],
    conditional_input: object | None,
) -> FloatArray:
    variant = _single_sequence_features(backbone, variant_sequences, conditional_input)
    reference = _single_sequence_features(backbone, reference_sequences, conditional_input)
    return np.concatenate((variant, reference, variant - reference), axis=1, dtype=np.float32)


def _labels(values: object) -> NDArray[np.int64]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    elif isinstance(values, Sequence):
        array = np.asarray(values)
    else:
        raise TypeError("probe labels must be a tensor or sequence")
    return np.asarray(array, dtype=np.int64).reshape(-1)
