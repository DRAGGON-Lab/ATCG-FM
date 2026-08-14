"""Lazy runtime providers for benchmarked genomic foundation models."""

from atcg.eval.providers.carbon import CarbonGFMModel
from atcg.eval.providers.evo2 import Evo2GFMModel
from atcg.eval.providers.jepa_dna import CheckpointLoadReport, JepaDnaGFMModel
from atcg.eval.providers.ntv3 import Ntv3GFMModel
from atcg.eval.providers.registry import (
    MODEL_PROVIDERS,
    LoadedProvider,
    ProviderRequest,
    load_provider,
)

__all__ = [
    "MODEL_PROVIDERS",
    "CarbonGFMModel",
    "CheckpointLoadReport",
    "Evo2GFMModel",
    "JepaDnaGFMModel",
    "LoadedProvider",
    "Ntv3GFMModel",
    "ProviderRequest",
    "load_provider",
]
