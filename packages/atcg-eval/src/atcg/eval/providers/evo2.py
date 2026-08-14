"""GFMBench adapter for the official Arc Institute Evo 2 runtime."""

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


class Evo2Tokenizer(Protocol):
    """The tokenizer surface exposed by ``evo2.Evo2``."""

    def tokenize(self, sequence: str) -> Sequence[int]: ...


class Evo2Runtime(Protocol):
    """The inference surface exposed by ``evo2.Evo2``."""

    tokenizer: Evo2Tokenizer

    def __call__(
        self,
        input_ids: torch.Tensor,
        return_embeddings: bool = False,
        layer_names: list[str] | None = None,
    ) -> tuple[object, Mapping[str, torch.Tensor] | None]: ...


class Evo2GFMModel(FrozenGFMModel):
    """Expose Evo 2 likelihoods and an explicit intermediate layer to GFMBench."""

    def __init__(
        self,
        runtime: Evo2Runtime,
        *,
        embedding_layer: str,
        max_sequence_length: int,
        device: str = "cuda:0",
        pooling: Pooling = "mean",
    ) -> None:
        if not embedding_layer:
            raise ValueError("Evo2 requires an explicit embedding layer")
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least two")
        self.runtime = runtime
        self.embedding_layer = embedding_layer
        self.max_sequence_length = max_sequence_length
        self.device = torch.device(device)
        self.pooling = pooling

    def eval(self) -> "Evo2GFMModel":
        model = getattr(self.runtime, "model", None)
        if isinstance(model, torch.nn.Module):
            model.eval()
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
            alphabet=frozenset("ACGTN"),
            boundary_replacement="N",
        )
        token_rows = [list(self.runtime.tokenizer.tokenize(sequence)) for sequence in normalized]
        sequence_length = len(normalized[0])
        if any(len(row) != sequence_length for row in token_rows):
            raise ValueError("Evo2 tokenizer is not base aligned for the supplied sequence")
        input_ids = torch.tensor(token_rows, dtype=torch.long, device=self.device)
        raw_logits, raw_embeddings = self.runtime(
            input_ids,
            return_embeddings=True,
            layer_names=[self.embedding_layer],
        )
        logits = _batch_sequence_tensor(raw_logits, len(normalized), sequence_length, "logits")
        if raw_embeddings is None or self.embedding_layer not in raw_embeddings:
            raise RuntimeError(f"Evo2 did not return embedding layer {self.embedding_layer!r}")
        embeddings = _batch_sequence_tensor(
            raw_embeddings[self.embedding_layer],
            len(normalized),
            sequence_length,
            "embeddings",
        )

        vocabulary_size = logits.shape[-1]
        observed = torch.full(
            (len(normalized), sequence_length),
            1.0 / vocabulary_size,
            device=logits.device,
            dtype=torch.float32,
        )
        if sequence_length > 1:
            observed[:, 1:] = (
                logits[:, :-1]
                .float()
                .softmax(dim=-1)
                .gather(-1, input_ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )
        representative = embeddings[:, -1] if self.pooling == "last" else embeddings.mean(dim=1)
        return (
            observed.cpu().numpy().astype(np.float32, copy=False),
            embeddings.float().cpu().numpy().astype(np.float32, copy=False),
            representative.float().cpu().numpy().astype(np.float32, copy=False),
        )

    def sequence_pos_to_prob_pos(self, sequences: list[str], pos: int) -> NDArray[np.int64]:
        return base_position_map(sequences, pos)


def _batch_sequence_tensor(
    value: object,
    batch_size: int,
    sequence_length: int,
    name: str,
) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        values = cast(Sequence[object], value)
        if len(values) != 1:
            raise ValueError(f"Evo2 {name} returned {len(values)} tensors; expected one")
        value = values[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Evo2 {name} must be a torch tensor")
    tensor = value
    if tensor.ndim == 2 and batch_size == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Evo2 {name} must have rank three")
    if tensor.shape[:2] == (batch_size, sequence_length):
        return tensor
    if tensor.shape[:2] == (sequence_length, batch_size):
        return tensor.transpose(0, 1)
    raise ValueError(
        f"Evo2 {name} shape {tuple(tensor.shape)} does not match "
        f"batch={batch_size}, length={sequence_length}"
    )
