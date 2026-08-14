"""JEPA-DNA target-encoder loading for the NTv3 benchmark backbone."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from atcg.eval.providers.base import FloatArray, FrozenGFMModel, NumericArray
from atcg.eval.providers.ntv3 import Ntv3GFMModel


@dataclass(frozen=True, slots=True)
class CheckpointLoadReport:
    """Auditable outcome of applying a JEPA-DNA target encoder."""

    matched_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


class JepaDnaGFMModel(FrozenGFMModel):
    """A JEPA-DNA target encoder with the capabilities of its NTv3 backbone."""

    def __init__(self, backbone: Ntv3GFMModel) -> None:
        self.backbone = backbone
        self.max_sequence_length = backbone.max_sequence_length

    def eval(self) -> "JepaDnaGFMModel":
        self.backbone.eval()
        return self

    def infer_sequence_to_sequence(
        self,
        sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[None, FloatArray, FloatArray]:
        return self.backbone.infer_sequence_to_sequence(sequences, conditional_input)

    def sequence_pos_to_prob_pos(
        self,
        sequences: list[str],
        pos: int,
    ) -> NDArray[np.int64]:
        return self.backbone.sequence_pos_to_prob_pos(sequences, pos)

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: list[str],
        variant_pos: int,
        variant_letters: list[str],
        reference_letters: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        return self.backbone.infer_masked_sequence_to_token_probs(
            sequences,
            variant_pos,
            variant_letters,
            reference_letters,
            conditional_input,
        )


def load_jepa_ntv3_checkpoint(
    adapter: Ntv3GFMModel,
    checkpoint: Path,
) -> CheckpointLoadReport:
    """Load a released JEPA-DNA NTv3 target state into its HF backbone."""

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not isinstance(adapter.model, torch.nn.Module):
        raise TypeError("JEPA-DNA checkpoint loading requires a torch module backbone")
    payload = cast(object, torch.load(checkpoint, map_location=adapter.device, weights_only=True))
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = cast(Mapping[object, object], payload)["model_state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError("JEPA-DNA checkpoint does not contain a state dictionary")

    state_dict: dict[str, torch.Tensor] = {}
    for raw_key, raw_value in cast(Mapping[object, object], payload).items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, torch.Tensor):
            continue
        key = raw_key.removeprefix("module.").removeprefix("model.")
        state_dict[key] = raw_value
    model_keys = set(adapter.model.state_dict())
    matched_keys = len(model_keys.intersection(state_dict))
    if matched_keys == 0:
        raise ValueError("JEPA-DNA checkpoint has no parameters matching the NTv3 backbone")

    incompatible = adapter.model.load_state_dict(state_dict, strict=False)
    adapter.model.eval()
    return CheckpointLoadReport(
        matched_keys=matched_keys,
        missing_keys=tuple(cast(list[str], incompatible.missing_keys)),
        unexpected_keys=tuple(cast(list[str], incompatible.unexpected_keys)),
    )
