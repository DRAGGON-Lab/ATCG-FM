"""Normalized result records independent of GFMBench's presentation CSV."""

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MetricStatus = Literal["succeeded", "unsupported", "failed"]


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One model, task, split, seed, and metric observation."""

    run_id: str
    protocol_id: str
    model_id: str
    task_id: str
    split: str
    seed: int
    metric_name: str
    metric_value: float | None
    status: MetricStatus
    error: str | None = None


def records_from_scores(
    *,
    run_id: str,
    protocol_id: str,
    model_id: str,
    task_id: str,
    seed: int,
    scores: Mapping[str, float | None],
) -> list[MetricRecord]:
    """Normalize GFMBench scores and reject non-finite numerical results."""

    records: list[MetricRecord] = []
    for name, value in sorted(scores.items()):
        if value is None:
            records.append(
                MetricRecord(
                    run_id=run_id,
                    protocol_id=protocol_id,
                    model_id=model_id,
                    task_id=task_id,
                    split="test",
                    seed=seed,
                    metric_name=name,
                    metric_value=None,
                    status="unsupported",
                )
            )
        else:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"metric {task_id}/{name} is not finite: {numeric}")
            records.append(
                MetricRecord(
                    run_id=run_id,
                    protocol_id=protocol_id,
                    model_id=model_id,
                    task_id=task_id,
                    split="test",
                    seed=seed,
                    metric_name=name,
                    metric_value=numeric,
                    status="succeeded",
                )
            )
    if not records:
        records.append(
            MetricRecord(
                run_id=run_id,
                protocol_id=protocol_id,
                model_id=model_id,
                task_id=task_id,
                split="test",
                seed=seed,
                metric_name="__task__",
                metric_value=None,
                status="unsupported",
                error="task returned no metrics",
            )
        )
    return records


def write_records(path: Path, records: Iterable[MetricRecord]) -> None:
    """Atomically write long-form JSONL records."""

    values = tuple(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for record in values:
                output.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_summary(path: Path, records: Iterable[MetricRecord]) -> None:
    """Write a compact CSV view without making it the canonical result store."""

    values = tuple(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = (
            csv.DictWriter(output, fieldnames=tuple(asdict(values[0]).keys())) if values else None
        )
        if writer is not None:
            writer.writeheader()
            for record in values:
                writer.writerow(asdict(record))
