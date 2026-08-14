"""Validated manifests for mixer-level and whole-block comparative experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal, cast

from atcg.models import GenomicLanguageModel, ModelConfig, StandardBlockSpec

type SubstitutionUnit = Literal["mixer", "block"]


@dataclass(frozen=True, slots=True)
class ComparisonInvariants:
    """Experimental quantities that must remain fixed across candidates."""

    tokenizer_id: str
    dataset_split: str
    training_tokens: int
    optimizer_protocol: str
    segment_length: int
    gradient_horizon: int

    def __post_init__(self) -> None:
        if self.training_tokens < 1 or self.segment_length < 1 or self.gradient_horizon < 1:
            raise ValueError("comparison token and segment quantities must be positive")
        for name, value in (
            ("tokenizer_id", self.tokenizer_id),
            ("dataset_split", self.dataset_split),
            ("optimizer_protocol", self.optimizer_protocol),
        ):
            if not value:
                raise ValueError(f"comparison {name} must not be empty")


@dataclass(frozen=True, slots=True)
class ComparisonCandidate:
    """Named native architecture participating in one controlled comparison."""

    candidate_id: str
    config: ModelConfig

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")


@dataclass(frozen=True, slots=True)
class ComparisonPlan:
    """A validated claim boundary and its architecture candidates."""

    comparison_id: str
    substitution_unit: SubstitutionUnit
    invariants: ComparisonInvariants
    candidates: tuple[ComparisonCandidate, ...]
    max_parameter_delta_fraction: float | None = None

    def __post_init__(self) -> None:
        if not self.comparison_id:
            raise ValueError("comparison_id must not be empty")
        if len(self.candidates) < 2:
            raise ValueError("a comparison requires at least two candidates")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("comparison candidate ids must be unique")
        if self.max_parameter_delta_fraction is not None and not (
            0.0 <= self.max_parameter_delta_fraction < 1.0
        ):
            raise ValueError("max_parameter_delta_fraction must be in [0, 1)")
        self._validate_architectures()

    def _validate_architectures(self) -> None:
        reference = self.candidates[0].config
        for candidate in self.candidates[1:]:
            config = candidate.config
            common = (
                config.vocab_size,
                config.d_model,
                config.max_seq_len,
                config.norm_eps,
                config.tie_embeddings,
                config.n_layers,
            )
            reference_common = (
                reference.vocab_size,
                reference.d_model,
                reference.max_seq_len,
                reference.norm_eps,
                reference.tie_embeddings,
                reference.n_layers,
            )
            if common != reference_common:
                raise ValueError("comparison candidates differ outside the block schedule")
        if self.substitution_unit == "mixer":
            shells = []
            for candidate in self.candidates:
                if not all(
                    isinstance(block, StandardBlockSpec) for block in candidate.config.blocks
                ):
                    raise ValueError("mixer comparisons require only standard blocks")
                standard_blocks = cast(tuple[StandardBlockSpec, ...], candidate.config.blocks)
                shells.append(
                    tuple((block.mlp_hidden_size, block.dropout) for block in standard_blocks)
                )
            if any(shell != shells[0] for shell in shells[1:]):
                raise ValueError("mixer comparison candidates must share one standard block shell")

    def manifest(
        self,
        models: Mapping[str, GenomicLanguageModel],
    ) -> dict[str, object]:
        """Materialize architecture and capacity accounting for a concrete run set."""

        expected_ids = {candidate.candidate_id for candidate in self.candidates}
        if set(models) != expected_ids:
            raise ValueError("comparison models must exactly match candidate ids")
        rows: list[dict[str, object]] = []
        parameter_counts: list[int] = []
        for candidate in self.candidates:
            model = models[candidate.candidate_id]
            if model.config != candidate.config:
                raise ValueError(
                    f"model {candidate.candidate_id!r} does not match its candidate configuration"
                )
            parameters = model.parameter_count()
            parameter_counts.append(parameters)
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "model_config": candidate.config.to_dict(),
                    "static_trainable_parameters": parameters,
                    "recurrent_state_elements_per_stream": model.recurrent_state_elements(),
                }
            )
        if self.max_parameter_delta_fraction is not None:
            largest = max(parameter_counts)
            delta = (largest - min(parameter_counts)) / largest
            if delta > self.max_parameter_delta_fraction:
                raise ValueError(
                    f"candidate parameter delta {delta:.4f} exceeds configured maximum "
                    f"{self.max_parameter_delta_fraction:.4f}"
                )
        return {
            "schema_version": 1,
            "comparison_id": self.comparison_id,
            "substitution_unit": self.substitution_unit,
            "invariants": asdict(self.invariants),
            "max_parameter_delta_fraction": self.max_parameter_delta_fraction,
            "candidates": rows,
        }
