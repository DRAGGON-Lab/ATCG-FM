"""Command line for strict modern genomic foundation-model evaluation."""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from atcg.eval.providers import MODEL_PROVIDERS, ProviderRequest, load_provider
from atcg.eval.providers.types import Pooling, ProviderName
from atcg.eval.registry import MODERN_V1, task_spec
from atcg.eval.runner import BenchmarkConfig, StrictBenchmarkRunner
from atcg.models import MemoryMode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atcg-eval",
        description="Run strict GFMBench-based ATCG evaluations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list-tasks", help="list the modern-v1 task protocol")
    listing.set_defaults(handler=_list_tasks)

    providers = commands.add_parser("list-providers", help="list supported model runtimes")
    providers.set_defaults(handler=_list_providers)

    run = commands.add_parser("run", help="evaluate one model runtime with GFMBench")
    run.add_argument("--provider", choices=tuple(MODEL_PROVIDERS), required=True)
    run.add_argument("--model-ref", help="provider model name or remote repository")
    run.add_argument("--model-id", help="label written to result records")
    run.add_argument("--revision", help="immutable remote model revision")
    run.add_argument(
        "--checkpoint",
        type=Path,
        help="native/JEPA weight file or provider-local snapshot directory",
    )
    run.add_argument("--backbone-ref", help="JEPA-DNA base encoder repository")
    run.add_argument("--backbone-revision", help="immutable JEPA-DNA backbone revision")
    run.add_argument("--max-sequence-length", type=int, help="evaluation context cap in bases")
    run.add_argument("--embedding-layer", help="Evo2 intermediate layer name")
    run.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    run.add_argument("--local-files-only", action="store_true", help="disable HF downloads")
    run.add_argument("--use-kernels", action="store_true", help="enable Evo2 Vortex kernels")
    run.add_argument(
        "--memory-mode",
        choices=("adaptive", "frozen", "disabled"),
        default="frozen",
        help="ATCG recurrent-memory policy; state always resets per benchmark sequence",
    )
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--task", action="append", required=True)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--max-samples", type=int)
    run.add_argument("--cache-size-gb", type=float, default=4.0)
    run.add_argument("--pooling", choices=("last", "mean"), default="mean")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cpu")
    run.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one evaluation command."""

    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    handler = cast(Callable[[argparse.Namespace], int], arguments.handler)
    return handler(arguments)


def _list_tasks(arguments: argparse.Namespace) -> int:
    del arguments
    _print_json(
        {
            "protocol": "modern-v1",
            "tasks": [{"family": task.family, "name": task.name} for task in MODERN_V1],
        }
    )
    return 0


def _list_providers(arguments: argparse.Namespace) -> int:
    del arguments
    _print_json(
        {
            "providers": {
                name: {
                    "default_max_sequence_length": spec.default_max_sequence_length or None,
                    "default_model_ref": spec.default_model_ref,
                    "default_revision": spec.default_revision,
                    "environment": spec.environment,
                    "runtime_revision": spec.runtime_revision,
                }
                for name, spec in MODEL_PROVIDERS.items()
            }
        }
    )
    return 0


def _run(arguments: argparse.Namespace) -> int:
    data_dir = cast(Path, arguments.data_dir)
    output_dir = cast(Path, arguments.output_dir)
    names = cast(list[str], arguments.task)
    loaded = load_provider(
        ProviderRequest(
            provider=cast(ProviderName, arguments.provider),
            model_ref=cast(str | None, arguments.model_ref),
            model_id=cast(str | None, arguments.model_id),
            revision=cast(str | None, arguments.revision),
            checkpoint=cast(Path | None, arguments.checkpoint),
            backbone_ref=cast(str | None, arguments.backbone_ref),
            backbone_revision=cast(str | None, arguments.backbone_revision),
            max_sequence_length=cast(int | None, arguments.max_sequence_length),
            device=cast(str, arguments.device),
            pooling=cast(Pooling, arguments.pooling),
            dtype=cast(str, arguments.dtype),
            embedding_layer=cast(str | None, arguments.embedding_layer),
            local_files_only=cast(bool, arguments.local_files_only),
            use_kernels=cast(bool, arguments.use_kernels),
            memory_mode=cast(MemoryMode, arguments.memory_mode),
        )
    )
    runner = StrictBenchmarkRunner(
        loaded.model,
        BenchmarkConfig(
            protocol_id="modern-v1",
            model=loaded.provenance,
            data_root=data_dir,
            output_dir=output_dir,
            max_sequence_length=loaded.max_sequence_length,
            batch_size=cast(int, arguments.batch_size),
            max_num_samples=cast(int | None, arguments.max_samples),
            cache_size_gb=cast(float, arguments.cache_size_gb),
            seed=cast(int, arguments.seed),
        ),
        [task_spec(name) for name in names],
    )
    result = runner.run()
    _print_json(
        {
            "output_dir": str(result.output_dir),
            "model_id": loaded.provenance.model_id,
            "provider": loaded.provenance.provider,
            "records": len(result.records),
            "run_id": result.run_id,
        }
    )
    return 0


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))
