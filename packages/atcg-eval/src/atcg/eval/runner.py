"""Strict execution of curated GFMBench task protocols."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from gfmbench_api.tasks.base import BaseGFMModel, BaseGFMTask

from atcg.eval.probe import fit_frozen_probe
from atcg.eval.provenance import ModelProvenance, benchmark_manifest, write_json
from atcg.eval.registry import TaskSpec
from atcg.eval.results import MetricRecord, records_from_scores, write_records, write_summary

TaskFactory = Callable[[TaskSpec, Path, Mapping[str, object]], BaseGFMTask]


class BenchmarkExecutionError(RuntimeError):
    """A task failed after its failure record was persisted."""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Resolved controls common to every task in one benchmark run."""

    protocol_id: str
    model: ModelProvenance
    data_root: Path
    output_dir: Path
    max_sequence_length: int
    batch_size: int = 8
    max_num_samples: int | None = None
    cache_size_gb: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_num_samples is not None and self.max_num_samples < 1:
            raise ValueError("max_num_samples must be positive or null")
        if self.cache_size_gb < 0:
            raise ValueError("cache_size_gb must not be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Materialized result of a complete strict task run."""

    run_id: str
    output_dir: Path
    records: tuple[MetricRecord, ...]


class StrictBenchmarkRunner:
    """Run GFMBench tasks with explicit errors and ATCG-owned artifacts."""

    def __init__(
        self,
        model: BaseGFMModel,
        config: BenchmarkConfig,
        tasks: Sequence[TaskSpec],
        *,
        task_factory: TaskFactory | None = None,
        workspace: Path | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("benchmark run must include at least one task")
        self.model = model
        self.config = config
        self.tasks = tuple(tasks)
        self.task_factory = task_factory or _create_task
        self.workspace = Path.cwd() if workspace is None else workspace

    def run(self) -> BenchmarkRun:
        """Execute tasks sequentially, persisting every completed or failed result."""

        destination = self.config.output_dir
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"benchmark output directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        run_id = uuid4().hex
        task_config = self._task_config()
        write_json(
            destination / "manifest.json",
            benchmark_manifest(
                run_id=run_id,
                protocol_id=self.config.protocol_id,
                model=self.config.model,
                data_root=self.config.data_root,
                task_names=[task.name for task in self.tasks],
                task_config=task_config,
                workspace=self.workspace,
            ),
        )

        all_records: list[MetricRecord] = []
        for spec in self.tasks:
            try:
                task = self.task_factory(spec, self.config.data_root, task_config)
                task_model = self.model
                if task.get_task_attributes().get("has_finetuning_data"):
                    task_model = fit_frozen_probe(
                        task,
                        self.model,
                        batch_size=self.config.batch_size,
                        seed=self.config.seed,
                    )
                raw_scores = task.eval_test_set(task_model)
                scores = cast(Mapping[str, float | None], raw_scores)
                all_records.extend(
                    records_from_scores(
                        run_id=run_id,
                        protocol_id=self.config.protocol_id,
                        model_id=self.config.model.model_id,
                        task_id=spec.name,
                        seed=self.config.seed,
                        scores=scores,
                    )
                )
            except Exception as error:
                all_records.append(
                    MetricRecord(
                        run_id=run_id,
                        protocol_id=self.config.protocol_id,
                        model_id=self.config.model.model_id,
                        task_id=spec.name,
                        split="test",
                        seed=self.config.seed,
                        metric_name="__task__",
                        metric_value=None,
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                self._write_results(destination, all_records)
                raise BenchmarkExecutionError(f"GFMBench task {spec.name!r} failed") from error
            self._write_results(destination, all_records)

        return BenchmarkRun(
            run_id=run_id,
            output_dir=destination,
            records=tuple(all_records),
        )

    def _task_config(self) -> dict[str, object]:
        return {
            "batch_size": self.config.batch_size,
            "cache_size": self.config.cache_size_gb,
            "disable_safe_model_call": True,
            "max_num_samples": self.config.max_num_samples,
            "max_sequence_length": self.config.max_sequence_length,
            "num_workers": 0,
            "supervised_probe": "logistic_regression_c1_lbfgs",
        }

    @staticmethod
    def _write_results(destination: Path, records: Sequence[MetricRecord]) -> None:
        write_records(destination / "records.jsonl", records)
        write_summary(destination / "summary.csv", records)


def _create_task(
    spec: TaskSpec,
    data_root: Path,
    task_config: Mapping[str, object],
) -> BaseGFMTask:
    return spec.load()(str(data_root), dict(task_config))
