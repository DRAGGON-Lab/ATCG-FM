from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pytest
import torch

from atcg.eval.providers.carbon import CarbonGFMModel, CarbonRuntime, CarbonTokenizer
from atcg.eval.providers.evo2 import Evo2GFMModel, Evo2Runtime
from atcg.eval.providers.jepa_dna import JepaDnaGFMModel, load_jepa_ntv3_checkpoint
from atcg.eval.providers.ntv3 import Ntv3GFMModel, Ntv3Runtime, Ntv3Tokenizer


class _EvoTokenizer:
    def tokenize(self, sequence: str) -> list[int]:
        return [{"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}[base] for base in sequence]


class _EvoRuntime:
    tokenizer = _EvoTokenizer()

    def __call__(
        self,
        input_ids: torch.Tensor,
        return_embeddings: bool = False,
        layer_names: list[str] | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor] | None]:
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 5)
        if length > 1:
            logits[:, :-1].scatter_(2, input_ids[:, 1:].unsqueeze(-1).cpu(), 8.0)
        embeddings = torch.arange(batch * length * 3, dtype=torch.float32).reshape(batch, length, 3)
        return logits, {"layer": embeddings} if return_embeddings and layer_names else None


class _CarbonTokenizer:
    k = 6

    def __call__(self, text: list[str], **kwargs: object) -> Mapping[str, torch.Tensor]:
        del kwargs
        lengths = [1 + (len(value.removeprefix("<dna>")) + self.k - 1) // self.k for value in text]
        width = max(lengths)
        input_ids = torch.zeros(len(text), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, length in enumerate(lengths):
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _CarbonRuntime:
    def score_sequence(self, sequences: list[str]) -> tuple[None, list[torch.Tensor]]:
        return None, [torch.full((len(sequence),), 0.75) for sequence in sequences]

    def __call__(self, **kwargs: object) -> Mapping[str, tuple[torch.Tensor]]:
        input_ids = cast(torch.Tensor, kwargs["input_ids"])
        batch, length = input_ids.shape
        hidden = torch.arange(batch * length * 2, dtype=torch.float32).reshape(batch, length, 2)
        return {"hidden_states": (hidden,)}


class _Ntv3Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    mask_token_id = 6
    unk_token_id = 7
    ids: ClassVar[dict[str, int]] = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}

    def __call__(self, text: list[str], **kwargs: object) -> Mapping[str, torch.Tensor]:
        del kwargs
        width = max(len(sequence) for sequence in text)
        input_ids = torch.zeros(len(text), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(text):
            values = torch.tensor([self.ids[base] for base in sequence])
            input_ids[row, : len(sequence)] = values
            attention_mask[row, : len(sequence)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids.get(token, self.unk_token_id)


class _Ntv3Runtime(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs: object) -> Mapping[str, object]:
        input_ids = cast(torch.Tensor, kwargs["input_ids"])
        hidden = torch.stack((input_ids.float(), input_ids.float() * self.scale), dim=-1)
        logits = torch.zeros(*input_ids.shape, 8, device=input_ids.device)
        logits[..., 1] = 1.0
        logits[..., 2] = 2.0
        return {"hidden_states": (hidden,), "logits": logits}


def test_evo2_adapter_returns_base_aligned_likelihoods_and_embeddings() -> None:
    adapter = Evo2GFMModel(
        cast(Evo2Runtime, _EvoRuntime()),
        embedding_layer="layer",
        max_sequence_length=16,
        device="cpu",
        pooling="last",
    )

    probs, embeddings, representative = adapter.infer_sequence_to_sequence(["ACGT", "TGCA"])

    assert probs.shape == (2, 4)
    assert probs[0, 0] == pytest.approx(0.2)
    assert probs[0, 1] > 0.99
    assert embeddings.shape == (2, 4, 3)
    np.testing.assert_array_equal(representative, embeddings[:, -1])


def test_carbon_adapter_uses_fns_probs_and_broadcasts_kmer_embeddings() -> None:
    adapter = CarbonGFMModel(
        cast(CarbonRuntime, _CarbonRuntime()),
        cast(CarbonTokenizer, _CarbonTokenizer()),
        max_sequence_length=32,
        device="cpu",
    )

    probs, embeddings, representative = adapter.infer_sequence_to_sequence(["ACGTACGTACGT"])

    np.testing.assert_allclose(probs, 0.75)
    assert embeddings.shape == (1, 12, 2)
    np.testing.assert_array_equal(embeddings[:, 0], embeddings[:, 5])
    assert not np.array_equal(embeddings[:, 5], embeddings[:, 6])
    assert representative.shape == (1, 2)


def test_ntv3_adapter_trims_alignment_padding_and_supports_masked_probs() -> None:
    adapter = Ntv3GFMModel(
        cast(Ntv3Runtime, _Ntv3Runtime()),
        cast(Ntv3Tokenizer, _Ntv3Tokenizer()),
        max_sequence_length=256,
        device="cpu",
    )

    probs, embeddings, representative = adapter.infer_sequence_to_sequence(["ACGT"])
    variant, reference = adapter.infer_masked_sequence_to_token_probs(["ACGT"], 1, ["C"], ["A"])

    assert probs is None
    assert embeddings.shape == (1, 4, 2)
    assert representative.shape == (1, 2)
    assert variant[0] > reference[0]
    np.testing.assert_array_equal(adapter.sequence_pos_to_prob_pos(["ACGT"], 3), [3])


def test_jepa_ntv3_loader_requires_and_reports_matching_parameters(tmp_path: Path) -> None:
    runtime = _Ntv3Runtime()
    adapter = Ntv3GFMModel(
        cast(Ntv3Runtime, runtime),
        cast(Ntv3Tokenizer, _Ntv3Tokenizer()),
        max_sequence_length=256,
        device="cpu",
    )
    checkpoint = tmp_path / "jepa.pt"
    torch.save({"model_state_dict": {"model.scale": torch.tensor(3.0)}}, checkpoint)

    report = load_jepa_ntv3_checkpoint(adapter, checkpoint)
    jepa = JepaDnaGFMModel(adapter)

    assert report.matched_keys == 1
    assert runtime.scale.item() == pytest.approx(3.0)
    assert jepa.infer_sequence_to_sequence(["ACGT"])[1].shape == (1, 4, 2)


def test_carbon_rejects_noncanonical_ambiguity() -> None:
    adapter = CarbonGFMModel(
        cast(CarbonRuntime, _CarbonRuntime()),
        cast(CarbonTokenizer, _CarbonTokenizer()),
        max_sequence_length=32,
        device="cpu",
    )

    with pytest.raises(ValueError, match="unsupported symbols"):
        adapter.infer_sequence_to_sequence(["ACGN"])
