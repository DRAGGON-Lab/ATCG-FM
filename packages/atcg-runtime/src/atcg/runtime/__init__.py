"""Training, checkpointing, and inference for ATCG-FM models."""

from atcg.runtime.batching import (
    CausalBatch,
    StatefulCausalBatch,
    collate_examples,
    collate_horizons,
)
from atcg.runtime.checkpoint import (
    LoadedCheckpoint,
    load_checkpoint,
    load_model_checkpoint,
    save_checkpoint,
)
from atcg.runtime.comparison import (
    ComparisonCandidate,
    ComparisonInvariants,
    ComparisonPlan,
    SubstitutionUnit,
)
from atcg.runtime.inference import (
    GenerationResult,
    IndependentSequenceOutput,
    SequenceScore,
    StatefulInferencePolicy,
    forward_independent_sequences,
    generate,
    score_sequence,
)
from atcg.runtime.stateful import StatefulTrainer, StreamStateStore, fit_stateful
from atcg.runtime.training import (
    StepMetrics,
    Trainer,
    TrainingConfig,
    TrainingRun,
    fit,
)
from atcg.runtime.training_state import TrainingState
from atcg.runtime.validation import CausalValidationMetrics, validate_causal_language_model

__all__ = [
    "CausalBatch",
    "CausalValidationMetrics",
    "ComparisonCandidate",
    "ComparisonInvariants",
    "ComparisonPlan",
    "GenerationResult",
    "IndependentSequenceOutput",
    "LoadedCheckpoint",
    "SequenceScore",
    "StatefulCausalBatch",
    "StatefulInferencePolicy",
    "StatefulTrainer",
    "StepMetrics",
    "StreamStateStore",
    "SubstitutionUnit",
    "Trainer",
    "TrainingConfig",
    "TrainingRun",
    "TrainingState",
    "collate_examples",
    "collate_horizons",
    "fit",
    "fit_stateful",
    "forward_independent_sequences",
    "generate",
    "load_checkpoint",
    "load_model_checkpoint",
    "save_checkpoint",
    "score_sequence",
    "validate_causal_language_model",
]
