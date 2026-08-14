"""Provider configuration types kept free of optional runtime imports."""

from typing import Literal

Pooling = Literal["last", "mean"]
ProviderName = Literal["atcg", "carbon", "evo2", "jepa-dna", "ntv3"]
