"""GFMBench adapter for Hugging Face Biology's Carbon models."""

import math
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from atcg.eval.providers.base import (
    FloatArray,
    FrozenGFMModel,
    NumericArray,
    base_position_map,
    normalize_sequences,
)
from atcg.eval.providers.types import Pooling


class CarbonTokenizer(Protocol):
    """Tokenizer surface used by the Carbon adapter."""

    k: int

    def __call__(self, text: list[str], **kwargs: object) -> Mapping[str, torch.Tensor]: ...


class CarbonRuntime(Protocol):
    """FNS scoring and hidden-state surface used by the Carbon adapter."""

    def score_sequence(self, sequences: list[str]) -> tuple[object, object]: ...

    def __call__(self, **kwargs: object) -> object: ...


class CarbonGFMModel(FrozenGFMModel):
    """Expose Carbon FNS base probabilities and broadcast 6-mer representations."""

    def __init__(
        self,
        model: CarbonRuntime,
        tokenizer: CarbonTokenizer,
        *,
        max_sequence_length: int,
        device: str = "cuda",
        pooling: Pooling = "mean",
        boundary_replacement: str = "A",
    ) -> None:
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if tokenizer.k < 1:
            raise ValueError("Carbon tokenizer k must be positive")
        if boundary_replacement not in "ACGT":
            raise ValueError("Carbon boundary replacement must be one canonical base")
        self.model = model
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.device = torch.device(device)
        self.pooling = pooling
        self.boundary_replacement = boundary_replacement

    def eval(self) -> "CarbonGFMModel":
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
        return self

    def infer_sequence_to_sequence(
        self,
        sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        del conditional_input
        normalized = normalize_sequences(
            sequences,
            max_sequence_length=self.max_sequence_length,
            alphabet=frozenset("ACGT"),
            boundary_replacement=self.boundary_replacement,
        )
        sequence_length = len(normalized[0])
        with torch.inference_mode():
            _, raw_actual_probs = self.model.score_sequence(normalized)
            observed = _actual_probabilities(raw_actual_probs, len(normalized), sequence_length)
            token_embeddings = self._token_embeddings(normalized)

        base_embeddings = token_embeddings.repeat_interleave(self.tokenizer.k, dim=1)
        base_embeddings = base_embeddings[:, :sequence_length]
        representative = (
            token_embeddings[:, -1] if self.pooling == "last" else token_embeddings.mean(dim=1)
        )
        return (
            observed.cpu().numpy().astype(np.float32, copy=False),
            base_embeddings.float().cpu().numpy().astype(np.float32, copy=False),
            representative.float().cpu().numpy().astype(np.float32, copy=False),
        )

    def sequence_pos_to_prob_pos(self, sequences: list[str], pos: int) -> NDArray[np.int64]:
        return base_position_map(sequences, pos)

    def _token_embeddings(self, sequences: list[str]) -> torch.Tensor:
        prompts = [f"<dna>{sequence}" for sequence in sequences]
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden_states = _output_field(outputs, "hidden_states")
        if not isinstance(hidden_states, (list, tuple)) or not hidden_states:
            raise RuntimeError("Carbon did not return hidden states")
        last_hidden = cast(Sequence[object], hidden_states)[-1]
        if not isinstance(last_hidden, torch.Tensor) or last_hidden.ndim != 3:
            raise TypeError("Carbon final hidden state must be a rank-three tensor")
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Carbon tokenizer did not return an attention mask")

        token_count = math.ceil(len(sequences[0]) / self.tokenizer.k)
        rows: list[torch.Tensor] = []
        for index in range(len(sequences)):
            valid_length = int(attention_mask[index].sum().item())
            start = valid_length - token_count
            if start < 0:
                raise RuntimeError("Carbon tokenization returned fewer DNA tokens than expected")
            row = last_hidden[index, start:valid_length]
            if row.shape[0] != token_count:
                raise RuntimeError("Carbon hidden-state alignment failed")
            rows.append(row)
        return torch.stack(rows)


def _actual_probabilities(value: object, batch_size: int, length: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.ndim == 1 and batch_size == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] != batch_size:
            raise ValueError("Carbon observed probabilities must have shape [batch, length]")
        if tensor.shape[1] < length:
            raise ValueError("Carbon returned fewer base probabilities than requested")
        return tensor[:, :length].float()
    if not isinstance(value, (list, tuple)) or len(value) != batch_size:
        raise TypeError("Carbon observed probabilities must be a tensor or tensor sequence")
    rows: list[torch.Tensor] = []
    for row in cast(Sequence[object], value):
        if not isinstance(row, torch.Tensor) or row.ndim != 1 or row.shape[0] < length:
            raise ValueError("Carbon observed probability rows must cover every input base")
        rows.append(row[:length].float())
    return torch.stack(rows)


def _output_field(outputs: object, field: str) -> object:
    if isinstance(outputs, Mapping):
        return cast(Mapping[object, object], outputs).get(field)
    return getattr(outputs, field, None)
