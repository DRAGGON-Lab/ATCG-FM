"""GFMBench adapter for InstaDeep Nucleotide Transformer v3."""

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
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

NTV3_SEQUENCE_MULTIPLE = 128


class Ntv3Tokenizer(Protocol):
    """Tokenizer surface used by the NTv3 adapter."""

    pad_token_id: int | None
    eos_token_id: int | None
    mask_token_id: int | None
    unk_token_id: int | None

    def __call__(self, text: list[str], **kwargs: object) -> Mapping[str, torch.Tensor]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


class Ntv3Runtime(Protocol):
    """Masked-LM surface used by the NTv3 adapter."""

    def __call__(self, **kwargs: object) -> object: ...


class Ntv3GFMModel(FrozenGFMModel):
    """Expose NTv3 base embeddings and masked-nucleotide probabilities."""

    def __init__(
        self,
        model: Ntv3Runtime,
        tokenizer: Ntv3Tokenizer,
        *,
        max_sequence_length: int,
        device: str = "cuda",
        pooling: Pooling = "mean",
        use_autocast: bool = False,
    ) -> None:
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.device = torch.device(device)
        self.pooling = pooling
        self.use_autocast = use_autocast and self.device.type == "cuda"

    def eval(self) -> "Ntv3GFMModel":
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
        return self

    def infer_sequence_to_sequence(
        self,
        sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[None, FloatArray, FloatArray]:
        del conditional_input
        normalized = self._normalize(sequences)
        input_ids, attention_mask = self._encode(normalized)
        with torch.inference_mode(), self._autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = _last_hidden(outputs)
        sequence_length = len(normalized[0])
        base_embeddings = hidden_states[:, :sequence_length]
        representative = (
            base_embeddings[:, -1] if self.pooling == "last" else base_embeddings.mean(dim=1)
        )
        return (
            None,
            base_embeddings.float().cpu().numpy().astype(np.float32, copy=False),
            representative.float().cpu().numpy().astype(np.float32, copy=False),
        )

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: list[str],
        variant_pos: int,
        variant_letters: list[str],
        reference_letters: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        del conditional_input
        normalized = self._normalize(sequences)
        if len(variant_letters) != len(normalized) or len(reference_letters) != len(normalized):
            raise ValueError("variant and reference letters must match the sequence batch")
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise RuntimeError("NTv3 tokenizer does not define a mask token")
        input_ids, attention_mask = self._encode(normalized)
        valid = np.asarray(
            [0 <= variant_pos < len(sequence) for sequence in normalized],
            dtype=np.bool_,
        )
        masked = input_ids.clone()
        for index, is_valid in enumerate(valid):
            if is_valid:
                masked[index, variant_pos] = mask_token_id
        with torch.inference_mode(), self._autocast():
            outputs = self.model(
                input_ids=masked,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
        logits = _output_field(outputs, "logits")
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("NTv3 logits must be a rank-three tensor")
        probabilities = logits.float().softmax(dim=-1)
        variant = np.zeros(len(normalized), dtype=np.float32)
        reference = np.zeros(len(normalized), dtype=np.float32)
        for index, is_valid in enumerate(valid):
            if not is_valid:
                continue
            variant_id = self._base_id(variant_letters[index])
            reference_id = self._base_id(reference_letters[index])
            variant[index] = probabilities[index, variant_pos, variant_id].item()
            reference[index] = probabilities[index, variant_pos, reference_id].item()
        return variant, reference

    def sequence_pos_to_prob_pos(self, sequences: list[str], pos: int) -> NDArray[np.int64]:
        return base_position_map(sequences, pos)

    def _normalize(self, sequences: list[str]) -> list[str]:
        return normalize_sequences(
            sequences,
            max_sequence_length=self.max_sequence_length,
            alphabet=frozenset("ACGTN"),
            boundary_replacement="N",
        )

    def _encode(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=NTV3_SEQUENCE_MULTIPLE,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_sequence_length,
        )
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("NTv3 tokenizer did not return input_ids")
        attention_mask = encoded.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.ones_like(input_ids)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return _pad_to_multiple(
            input_ids,
            attention_mask,
            multiple=NTV3_SEQUENCE_MULTIPLE,
            pad_token_id=self._pad_token_id(),
        )

    def _pad_token_id(self) -> int:
        token_id = self.tokenizer.pad_token_id
        if token_id is None:
            token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise RuntimeError("NTv3 tokenizer does not define a pad or EOS token")
        return token_id

    def _base_id(self, base: str) -> int:
        normalized = base.upper().replace("P", "N")
        if len(normalized) != 1 or normalized not in "ACGTN":
            raise ValueError(f"unsupported NTv3 base {base!r}")
        token_id = self.tokenizer.convert_tokens_to_ids(normalized)
        if self.tokenizer.unk_token_id is not None and token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"NTv3 tokenizer does not encode base {base!r}")
        return token_id

    def _autocast(self) -> AbstractContextManager[object]:
        if self.use_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()


def _pad_to_multiple(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    multiple: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    remainder = input_ids.shape[1] % multiple
    if remainder == 0:
        return input_ids, attention_mask
    padding = multiple - remainder
    pad_ids = torch.full(
        (input_ids.shape[0], padding),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    pad_mask = torch.zeros(
        (attention_mask.shape[0], padding),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat((input_ids, pad_ids), dim=1), torch.cat((attention_mask, pad_mask), dim=1)


def _last_hidden(outputs: object) -> torch.Tensor:
    hidden_states = _output_field(outputs, "hidden_states")
    if not isinstance(hidden_states, (list, tuple)) or not hidden_states:
        raise RuntimeError("NTv3 did not return hidden states")
    last_hidden = cast(tuple[object, ...] | list[object], hidden_states)[-1]
    if not isinstance(last_hidden, torch.Tensor) or last_hidden.ndim != 3:
        raise TypeError("NTv3 final hidden state must be a rank-three tensor")
    return last_hidden


def _output_field(outputs: object, field: str) -> object:
    if isinstance(outputs, Mapping):
        return cast(Mapping[object, object], outputs).get(field)
    return getattr(outputs, field, None)
