"""Lazy construction and provenance for supported model runtimes."""

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from gfmbench_api.tasks.base import BaseGFMModel

from atcg.eval.model import AtcgGFMModel
from atcg.eval.provenance import ModelProvenance
from atcg.eval.providers.carbon import CarbonGFMModel, CarbonRuntime, CarbonTokenizer
from atcg.eval.providers.evo2 import Evo2GFMModel, Evo2Runtime
from atcg.eval.providers.jepa_dna import JepaDnaGFMModel, load_jepa_ntv3_checkpoint
from atcg.eval.providers.ntv3 import Ntv3GFMModel, Ntv3Runtime, Ntv3Tokenizer
from atcg.eval.providers.types import Pooling, ProviderName
from atcg.models import MemoryMode
from atcg.runtime import StatefulInferencePolicy, load_model_checkpoint
from atcg.sequence import FixedAlphabetTokenizer

EVO2_RUNTIME_REVISION = "53f195997257c56c00e5ef8d33a54f5baad143a6"
CARBON_FNS_REVISION = "bf6f6bec000ea6ced8cb656d02f3120a24795c91"
NTV3_RUNTIME_REVISION = "2dc37b86e16a6970fbc731751f7719d9f676f7f9"
JEPA_DNA_RUNTIME_REVISION = "99e443341646e7cce3663f6a5a333cf421f24061"
JEPA_DNA_NTV3_CHECKPOINT_REVISION = "be526df5438b375223017051b1a36d8b9dee2f59"
REPOSITORY_ROOT = Path(__file__).resolve().parents[6]


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Discoverable defaults for one runtime provider."""

    default_model_ref: str
    default_revision: str | None
    default_max_sequence_length: int
    runtime_revision: str | None
    environment: str


MODEL_PROVIDERS: Mapping[ProviderName, ProviderSpec] = {
    "atcg": ProviderSpec("atcg", None, 0, None, "workspace"),
    "carbon": ProviderSpec(
        "HuggingFaceBio/Carbon-3B",
        CARBON_FNS_REVISION,
        196_602,
        CARBON_FNS_REVISION,
        "environments/models/carbon",
    ),
    "evo2": ProviderSpec(
        "evo2_7b",
        None,
        1_000_000,
        EVO2_RUNTIME_REVISION,
        "environments/models/evo2",
    ),
    "jepa-dna": ProviderSpec(
        "nvidia/NV-JEPA-DNA-NTv3",
        JEPA_DNA_NTV3_CHECKPOINT_REVISION,
        8_192,
        JEPA_DNA_RUNTIME_REVISION,
        "environments/models/jepa-dna",
    ),
    "ntv3": ProviderSpec(
        "InstaDeepAI/NTv3_100M_pre",
        None,
        8_192,
        NTV3_RUNTIME_REVISION,
        "environments/models/ntv3",
    ),
}


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """Runtime-neutral request parsed from the benchmark CLI."""

    provider: ProviderName
    model_ref: str | None = None
    model_id: str | None = None
    revision: str | None = None
    checkpoint: Path | None = None
    backbone_ref: str | None = None
    backbone_revision: str | None = None
    max_sequence_length: int | None = None
    device: str = "cpu"
    pooling: Pooling = "mean"
    dtype: str = "auto"
    embedding_layer: str | None = None
    local_files_only: bool = False
    use_kernels: bool = False
    memory_mode: MemoryMode = "frozen"


@dataclass(frozen=True, slots=True)
class LoadedProvider:
    """A constructed GFMBench model and the exact identity used to construct it."""

    model: BaseGFMModel
    provenance: ModelProvenance
    max_sequence_length: int


class _FromPretrained(Protocol):
    @classmethod
    def from_pretrained(cls, model_ref: str, **kwargs: object) -> object: ...


def load_provider(request: ProviderRequest) -> LoadedProvider:
    """Construct one provider while importing optional runtimes only on demand."""

    loaders: Mapping[ProviderName, Callable[[ProviderRequest], LoadedProvider]] = {
        "atcg": _load_atcg,
        "carbon": _load_carbon,
        "evo2": _load_evo2,
        "jepa-dna": _load_jepa_dna,
        "ntv3": _load_ntv3,
    }
    return loaders[request.provider](request)


def _load_atcg(request: ProviderRequest) -> LoadedProvider:
    if request.revision is not None:
        raise ValueError("ATCG checkpoints are identified by content hash, not --revision")
    if request.dtype != "auto":
        raise ValueError("ATCG checkpoint dtype is fixed by the native runtime")
    checkpoint = _required_checkpoint(request, file_only=True)
    model, _ = load_model_checkpoint(checkpoint, device=request.device)
    tokenizer = FixedAlphabetTokenizer()
    stateful_policy = (
        StatefulInferencePolicy(
            memory_mode=request.memory_mode,
            max_sequence_length=request.max_sequence_length,
        )
        if model.config.is_stateful
        else None
    )
    adapter = AtcgGFMModel(
        model,
        tokenizer,
        device=request.device,
        pooling=request.pooling,
        stateful_policy=stateful_policy,
    )
    model_ref = request.model_ref or checkpoint.stem
    max_length = adapter.max_sequence_length
    if request.max_sequence_length is not None:
        max_length = min(max_length, request.max_sequence_length)
    return LoadedProvider(
        model=adapter,
        max_sequence_length=max_length,
        provenance=ModelProvenance(
            provider=request.provider,
            model_id=_model_id(request, model_ref),
            model_ref=model_ref,
            checkpoint=checkpoint,
            metadata={
                "adapter": type(adapter).__name__,
                "alphabet": tokenizer.alphabet,
                "boundary_padding": "unknown",
                "context_tokens": model.config.max_seq_len,
                "parameters": model.parameter_count(),
                "pooling": request.pooling,
                "recurrent_state_elements": model.recurrent_state_elements(),
                "memory_mode": request.memory_mode if model.config.is_stateful else None,
                "state_reset": "per_sequence" if model.config.is_stateful else None,
                "tokenizer": "fixed_iupac_v1",
            },
        ),
    )


def _load_evo2(request: ProviderRequest) -> LoadedProvider:
    if request.revision is not None:
        raise ValueError(
            "the Evo2 Vortex loader cannot select a model revision; "
            "use a pinned local snapshot with --checkpoint"
        )
    if request.dtype != "auto":
        raise ValueError("Evo2 precision is fixed by its checkpoint configuration")
    spec = MODEL_PROVIDERS["evo2"]
    model_ref = request.model_ref or spec.default_model_ref
    max_length = request.max_sequence_length or _evo2_context(model_ref)
    embedding_layer = request.embedding_layer or _evo2_embedding_layer(model_ref)
    try:
        module = importlib.import_module("evo2")
    except ImportError as error:
        raise _optional_runtime_error("evo2", spec.environment) from error
    runtime_class = cast(Callable[..., Evo2Runtime], module.Evo2)
    kwargs: dict[str, object] = {
        "model_name": model_ref,
        "use_kernels": request.use_kernels,
    }
    if request.checkpoint is not None:
        kwargs["local_path"] = str(request.checkpoint)
    runtime = runtime_class(**kwargs)
    adapter = Evo2GFMModel(
        runtime,
        embedding_layer=embedding_layer,
        max_sequence_length=max_length,
        device=request.device,
        pooling=request.pooling,
    )
    return LoadedProvider(
        model=adapter,
        max_sequence_length=max_length,
        provenance=ModelProvenance(
            provider=request.provider,
            model_id=_model_id(request, model_ref),
            model_ref=model_ref,
            revision=request.revision,
            runtime_revision=spec.runtime_revision,
            runtime_lock=_runtime_lock(spec),
            checkpoint=request.checkpoint,
            metadata={
                "adapter": type(adapter).__name__,
                "boundary_padding": "unknown",
                "embedding_layer": embedding_layer,
                "pooling": request.pooling,
                "use_kernels": request.use_kernels,
            },
        ),
    )


def _load_carbon(request: ProviderRequest) -> LoadedProvider:
    spec = MODEL_PROVIDERS["carbon"]
    model_ref = request.model_ref or spec.default_model_ref
    checkpoint = (
        _required_checkpoint(request, file_only=False) if request.checkpoint is not None else None
    )
    if checkpoint is not None and not checkpoint.is_dir():
        raise NotADirectoryError(checkpoint)
    load_ref = str(checkpoint) if checkpoint is not None else model_ref
    revision = (
        request.revision if checkpoint is not None else request.revision or spec.default_revision
    )
    tokenizer, model = _load_huggingface_masked_or_causal(
        auto_model_name="AutoModelForCausalLM",
        model_ref=load_ref,
        revision=revision if checkpoint is None else None,
        request=request,
        environment=spec.environment,
    )
    carbon_tokenizer = cast(CarbonTokenizer, tokenizer)
    adapter = CarbonGFMModel(
        cast(CarbonRuntime, model),
        carbon_tokenizer,
        max_sequence_length=request.max_sequence_length or spec.default_max_sequence_length,
        device=request.device,
        pooling=request.pooling,
    )
    return LoadedProvider(
        model=adapter,
        max_sequence_length=adapter.max_sequence_length,
        provenance=ModelProvenance(
            provider=request.provider,
            model_id=_model_id(request, model_ref),
            model_ref=model_ref,
            revision=revision,
            runtime_revision=spec.runtime_revision,
            runtime_lock=_runtime_lock(spec),
            checkpoint=checkpoint,
            metadata={
                "adapter": type(adapter).__name__,
                "boundary_padding": "adenine",
                "dtype": request.dtype,
                "kmer_width": carbon_tokenizer.k,
                "pooling": request.pooling,
                "probabilities": "factorized_nucleotide_supervision",
            },
        ),
    )


def _load_ntv3(request: ProviderRequest) -> LoadedProvider:
    spec = MODEL_PROVIDERS["ntv3"]
    model_ref = request.model_ref or spec.default_model_ref
    checkpoint = (
        _required_checkpoint(request, file_only=False) if request.checkpoint is not None else None
    )
    if checkpoint is not None and not checkpoint.is_dir():
        raise NotADirectoryError(checkpoint)
    load_ref = str(checkpoint) if checkpoint is not None else model_ref
    tokenizer, model = _load_huggingface_masked_or_causal(
        auto_model_name="AutoModelForMaskedLM",
        model_ref=load_ref,
        revision=request.revision if checkpoint is None else None,
        request=request,
        environment=spec.environment,
    )
    adapter = Ntv3GFMModel(
        cast(Ntv3Runtime, model),
        cast(Ntv3Tokenizer, tokenizer),
        max_sequence_length=request.max_sequence_length or spec.default_max_sequence_length,
        device=request.device,
        pooling=request.pooling,
        use_autocast=request.dtype == "bfloat16",
    )
    return LoadedProvider(
        model=adapter,
        max_sequence_length=adapter.max_sequence_length,
        provenance=ModelProvenance(
            provider=request.provider,
            model_id=_model_id(request, model_ref),
            model_ref=model_ref,
            revision=request.revision,
            runtime_revision=spec.runtime_revision,
            runtime_lock=_runtime_lock(spec),
            checkpoint=checkpoint,
            metadata={
                "adapter": type(adapter).__name__,
                "boundary_padding": "unknown",
                "dtype": request.dtype,
                "pooling": request.pooling,
                "probabilities": "masked_nucleotide_only",
                "tokenizer": "single_base",
            },
        ),
    )


def _load_jepa_dna(request: ProviderRequest) -> LoadedProvider:
    spec = MODEL_PROVIDERS["jepa-dna"]
    checkpoint = _required_checkpoint(request, file_only=True)
    model_ref = request.model_ref or spec.default_model_ref
    backbone_ref = request.backbone_ref or "InstaDeepAI/NTv3_100M_pre"
    backbone_request = ProviderRequest(
        provider="ntv3",
        model_ref=backbone_ref,
        revision=request.backbone_revision,
        max_sequence_length=request.max_sequence_length,
        device=request.device,
        pooling=request.pooling,
        dtype=request.dtype,
        local_files_only=request.local_files_only,
    )
    loaded_backbone = _load_ntv3(backbone_request)
    if not isinstance(loaded_backbone.model, Ntv3GFMModel):
        raise TypeError("JEPA-DNA NTv3 provider constructed an incompatible backbone")
    report = load_jepa_ntv3_checkpoint(loaded_backbone.model, checkpoint)
    adapter = JepaDnaGFMModel(loaded_backbone.model)
    metadata = dict(loaded_backbone.provenance.metadata)
    metadata.update(
        {
            "adapter": type(adapter).__name__,
            "backbone_ref": backbone_ref,
            "backbone_revision": request.backbone_revision,
            "checkpoint_matched_keys": report.matched_keys,
            "checkpoint_missing_keys": list(report.missing_keys),
            "checkpoint_unexpected_keys": list(report.unexpected_keys),
            "encoder": "target",
        }
    )
    return LoadedProvider(
        model=adapter,
        max_sequence_length=adapter.max_sequence_length,
        provenance=ModelProvenance(
            provider=request.provider,
            model_id=_model_id(request, model_ref),
            model_ref=model_ref,
            revision=request.revision or spec.default_revision,
            runtime_revision=spec.runtime_revision,
            runtime_lock=_runtime_lock(spec),
            checkpoint=checkpoint,
            metadata=metadata,
        ),
    )


def _load_huggingface_masked_or_causal(
    *,
    auto_model_name: str,
    model_ref: str,
    revision: str | None,
    request: ProviderRequest,
    environment: str,
) -> tuple[object, object]:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise _optional_runtime_error(request.provider, environment) from error
    tokenizer_factory = cast(_FromPretrained, transformers.AutoTokenizer)
    model_factory = cast(_FromPretrained, getattr(transformers, auto_model_name))
    load_kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "local_files_only": request.local_files_only,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    tokenizer = tokenizer_factory.from_pretrained(model_ref, **load_kwargs)
    dtype = _torch_dtype(request.dtype)
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    model = model_factory.from_pretrained(model_ref, **load_kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{request.provider} runtime did not return a torch module")
    model.to(request.device).eval()
    return tokenizer, model


def _torch_dtype(name: str) -> torch.dtype | None:
    values = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}") from error


def _required_checkpoint(request: ProviderRequest, *, file_only: bool) -> Path:
    if request.checkpoint is None:
        raise ValueError(f"provider {request.provider!r} requires --checkpoint")
    if file_only and not request.checkpoint.is_file():
        raise FileNotFoundError(request.checkpoint)
    if not file_only and not request.checkpoint.exists():
        raise FileNotFoundError(request.checkpoint)
    return request.checkpoint


def _model_id(request: ProviderRequest, model_ref: str) -> str:
    return request.model_id or f"{request.provider}:{model_ref}"


def _evo2_context(model_ref: str) -> int:
    if model_ref.endswith("_base") or model_ref == "evo2_1b_base":
        return 8_192
    if model_ref == "evo2_7b_262k":
        return 262_144
    return 1_000_000


def _evo2_embedding_layer(model_ref: str) -> str:
    if "7b" in model_ref:
        return "blocks.28.mlp.l3"
    raise ValueError("--embedding-layer is required for non-7B Evo2 checkpoints")


def _optional_runtime_error(provider: str, environment: str) -> ImportError:
    return ImportError(
        f"provider {provider!r} is not installed in this environment; "
        f"run through the isolated uv project at {environment}"
    )


def _runtime_lock(spec: ProviderSpec) -> Path:
    return REPOSITORY_ROOT / spec.environment / "uv.lock"
