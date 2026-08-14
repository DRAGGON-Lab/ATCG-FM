import json
from pathlib import Path
from typing import cast

import pytest
from gfmbench_api.tasks.base import BaseGFMModel, BaseGFMTask

from atcg.eval.provenance import ModelProvenance
from atcg.eval.registry import TaskSpec
from atcg.eval.runner import (
    BenchmarkConfig,
    BenchmarkExecutionError,
    StrictBenchmarkRunner,
)


class _Model:
    pass


class _Task:
    def __init__(self, scores: dict[str, float | None] | None = None) -> None:
        self.scores = scores

    def eval_test_set(self, model: object) -> dict[str, float | None]:
        del model
        if self.scores is None:
            raise RuntimeError("intentional failure")
        return self.scores

    def get_task_attributes(self) -> dict[str, object]:
        return {"has_finetuning_data": False}


def _config(tmp_path: Path, output_name: str) -> BenchmarkConfig:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"trusted fixture")
    return BenchmarkConfig(
        protocol_id="modern-v1",
        model=ModelProvenance(
            provider="atcg",
            model_id="fixture-model",
            model_ref="fixture-model",
            checkpoint=checkpoint,
            metadata={"pooling": "last"},
        ),
        data_root=tmp_path / "data",
        output_dir=tmp_path / output_name,
        max_sequence_length=128,
        max_num_samples=100,
    )


def _spec() -> TaskSpec:
    return TaskSpec("fixture", "unused", "Unused", "zero_shot_likelihood")


def test_strict_runner_writes_normalized_success_and_unsupported_records(
    tmp_path: Path,
) -> None:
    def factory(spec: TaskSpec, root: Path, config: object) -> BaseGFMTask:
        del spec, root, config
        return cast(BaseGFMTask, _Task({"auroc": 0.75, "masked_auroc": None}))

    result = StrictBenchmarkRunner(
        cast(BaseGFMModel, _Model()),
        _config(tmp_path, "success"),
        [_spec()],
        task_factory=factory,
        workspace=tmp_path,
    ).run()

    assert {record.status for record in result.records} == {"succeeded", "unsupported"}
    assert (result.output_dir / "manifest.json").is_file()
    assert (result.output_dir / "summary.csv").is_file()
    records = [
        json.loads(line) for line in (result.output_dir / "records.jsonl").read_text().splitlines()
    ]
    assert [record["metric_name"] for record in records] == ["auroc", "masked_auroc"]
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["task_config"]["disable_safe_model_call"] is True
    assert manifest["task_config"]["num_workers"] == 0
    assert manifest["model"]["evaluation"]["pooling"] == "last"
    assert manifest["model"]["checkpoint"]["sha256"]


def test_strict_runner_accepts_remote_model_identity(tmp_path: Path) -> None:
    def factory(spec: TaskSpec, root: Path, config: object) -> BaseGFMTask:
        del spec, root, config
        return cast(BaseGFMTask, _Task({"auroc": 0.75}))

    runtime_lock = tmp_path / "model-runtime.lock"
    runtime_lock.write_text("resolved runtime")
    config = BenchmarkConfig(
        protocol_id="modern-v1",
        model=ModelProvenance(
            provider="carbon",
            model_id="carbon-fixture",
            model_ref="HuggingFaceBio/Carbon-3B",
            revision="immutable-revision",
            runtime_lock=runtime_lock,
        ),
        data_root=tmp_path / "data",
        output_dir=tmp_path / "remote",
        max_sequence_length=128,
    )
    result = StrictBenchmarkRunner(
        cast(BaseGFMModel, _Model()),
        config,
        [_spec()],
        task_factory=factory,
        workspace=tmp_path,
    ).run()

    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["model"]["checkpoint"] is None
    assert manifest["model"]["revision"] == "immutable-revision"
    assert manifest["model"]["runtime_lock"]["sha256"]


def test_strict_runner_persists_failure_before_raising(tmp_path: Path) -> None:
    def factory(spec: TaskSpec, root: Path, config: object) -> BaseGFMTask:
        del spec, root, config
        return cast(BaseGFMTask, _Task())

    config = _config(tmp_path, "failure")
    runner = StrictBenchmarkRunner(
        cast(BaseGFMModel, _Model()),
        config,
        [_spec()],
        task_factory=factory,
        workspace=tmp_path,
    )

    with pytest.raises(BenchmarkExecutionError, match="fixture"):
        runner.run()

    record = json.loads((config.output_dir / "records.jsonl").read_text().strip())
    assert record["status"] == "failed"
    assert "intentional failure" in record["error"]
