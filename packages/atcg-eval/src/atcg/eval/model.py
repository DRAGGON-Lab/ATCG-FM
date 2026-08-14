"""GFMBench adapter for native ATCG autoregressive models."""

from typing import Literal

import numpy as np
import torch
from gfmbench_api.tasks.base import BaseGFMModel
from numpy.typing import NDArray

from atcg.models import GenomicLanguageModel
from atcg.runtime import StatefulInferencePolicy, forward_independent_sequences
from atcg.sequence import Tokenizer

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
NumericArray = NDArray[np.generic]
Pooling = Literal["last", "mean"]
BoundaryPadding = Literal["error", "unknown"]


class AtcgGFMModel(BaseGFMModel):
    """Expose an ATCG character-level causal model through GFMBench's model API."""

    def __init__(
        self,
        model: GenomicLanguageModel,
        tokenizer: Tokenizer,
        *,
        device: str = "cpu",
        pooling: Pooling = "last",
        boundary_padding: BoundaryPadding = "unknown",
        stateful_policy: StatefulInferencePolicy | None = None,
    ) -> None:
        if model.config.max_seq_len < 2:
            raise ValueError("GFMBench evaluation requires a model context of at least two tokens")
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError("tokenizer vocabulary does not match the model configuration")
        if model.config.is_stateful and stateful_policy is None:
            raise ValueError("stateful ATCG models require an explicit inference policy")
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.pooling = pooling
        self.boundary_padding = boundary_padding
        self.stateful_policy = stateful_policy

    @property
    def max_sequence_length(self) -> int:
        """Maximum nucleotide length after reserving one position for BOS."""

        if (
            self.stateful_policy is not None
            and self.stateful_policy.max_sequence_length is not None
        ):
            return self.stateful_policy.max_sequence_length
        return self.model.config.max_seq_len - 1

    def eval(self) -> "AtcgGFMModel":
        """Match the model method expected by GFMBench runners."""

        self.model.eval()
        return self

    def infer_sequence_to_labels_probs(
        self,
        sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> FloatArray | None:
        """Return no predictions until a supervised head has been fitted."""

        return None

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: list[str],
        ref_sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> FloatArray | None:
        """Return no predictions until a paired-sequence head has been fitted."""

        return None

    def infer_sequence_to_sequence(
        self,
        sequences: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return base-aligned probabilities, embeddings, and sequence representations."""

        normalized = [self._normalize_sequence(sequence) for sequence in sequences]
        if not normalized:
            raise ValueError("sequence batch must not be empty")
        lengths = {len(sequence) for sequence in normalized}
        if len(lengths) != 1:
            raise ValueError("GFMBench batches must contain equal-length sequences")
        sequence_length = len(normalized[0])
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds adapter maximum "
                f"{self.max_sequence_length}"
            )

        target_rows = [self.tokenizer.encode(sequence) for sequence in normalized]
        input_rows = [[self.tokenizer.bos_id, *token_ids] for token_ids in target_rows]
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        targets = torch.tensor(target_rows, dtype=torch.long, device=self.device)

        self.model.eval()
        logits, hidden_states = self._forward_inputs(input_ids)
        prediction_logits = logits[:, :-1]
        observed_probs = prediction_logits.softmax(dim=-1).gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        )
        base_embeddings = hidden_states[:, 1:]
        if self.pooling == "last":
            representative = base_embeddings[:, -1]
        else:
            representative = base_embeddings.mean(dim=1)

        return (
            observed_probs.squeeze(-1).float().cpu().numpy(),
            base_embeddings.float().cpu().numpy(),
            representative.float().cpu().numpy(),
        )

    def _forward_inputs(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = forward_independent_sequences(
            self.model,
            input_ids,
            pad_id=self.tokenizer.pad_id,
            stateful_policy=self.stateful_policy,
        )
        return output.logits, output.hidden_states

    def sequence_pos_to_prob_pos(self, sequences: list[str], pos: int) -> IntArray:
        """Map a nucleotide position to the base-aligned arrays returned above."""

        mapped = np.full(len(sequences), -1, dtype=np.int64)
        for index, sequence in enumerate(sequences):
            normalized = self._normalize_sequence(sequence)
            if 0 <= pos < len(normalized):
                mapped[index] = pos
        return mapped

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: list[str],
        variant_pos: int,
        variant_letters: list[str],
        reference_letters: list[str],
        conditional_input: NumericArray | None = None,
    ) -> tuple[None, None]:
        """Causal ATCG models do not provide masked-token probabilities."""

        return None, None

    def _normalize_sequence(self, sequence: str) -> str:
        normalized = sequence.upper()
        if self.boundary_padding == "unknown":
            normalized = normalized.replace("P", "N")
        elif "P" in normalized:
            raise ValueError("GFMBench boundary padding P is not accepted by this adapter")
        if not normalized:
            raise ValueError("sequences must not be empty")
        self.tokenizer.encode(normalized)
        return normalized
