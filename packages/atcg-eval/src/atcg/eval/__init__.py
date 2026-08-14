"""Strict GFMBench-based evaluation for genomic foundation models."""

from atcg.eval.cli import main
from atcg.eval.model import AtcgGFMModel
from atcg.eval.probe import FrozenProbeGFMModel, fit_frozen_probe
from atcg.eval.provenance import ModelProvenance
from atcg.eval.providers import (
    MODEL_PROVIDERS,
    CarbonGFMModel,
    Evo2GFMModel,
    JepaDnaGFMModel,
    LoadedProvider,
    Ntv3GFMModel,
    ProviderRequest,
    load_provider,
)
from atcg.eval.registry import MODERN_V1, TaskSpec, task_spec
from atcg.eval.runner import (
    BenchmarkConfig,
    BenchmarkExecutionError,
    BenchmarkRun,
    StrictBenchmarkRunner,
)
from atcg.runtime import StatefulInferencePolicy

__all__ = [
    "MODEL_PROVIDERS",
    "MODERN_V1",
    "AtcgGFMModel",
    "BenchmarkConfig",
    "BenchmarkExecutionError",
    "BenchmarkRun",
    "CarbonGFMModel",
    "Evo2GFMModel",
    "FrozenProbeGFMModel",
    "JepaDnaGFMModel",
    "LoadedProvider",
    "ModelProvenance",
    "Ntv3GFMModel",
    "ProviderRequest",
    "StatefulInferencePolicy",
    "StrictBenchmarkRunner",
    "TaskSpec",
    "fit_frozen_probe",
    "load_provider",
    "main",
    "task_spec",
]
