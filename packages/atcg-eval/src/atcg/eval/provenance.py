"""Research provenance captured around GFMBench execution."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

GFMBENCH_REVISION = "ce1be5d98c37b9f9ceedc32b254fe38668f989d8"


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Identity of a local or externally hosted benchmark model."""

    provider: str
    model_id: str
    model_ref: str
    revision: str | None = None
    runtime_revision: str | None = None
    runtime_lock: Path | None = None
    checkpoint: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        for field_name in ("provider", "model_id", "model_ref"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.checkpoint is not None and not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)
        if self.runtime_lock is not None and not self.runtime_lock.is_file():
            raise FileNotFoundError(self.runtime_lock)

    def as_manifest(self) -> dict[str, object]:
        """Return a JSON-compatible model identity record."""

        checkpoint: dict[str, object] | None = None
        if self.checkpoint is not None:
            resolved = self.checkpoint.resolve()
            checkpoint = {
                "path": str(resolved),
                "sha256": sha256_path(resolved),
                "type": "directory" if resolved.is_dir() else "file",
            }
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "model_ref": self.model_ref,
            "revision": self.revision,
            "runtime_revision": self.runtime_revision,
            "runtime_lock": (
                {
                    "path": str(self.runtime_lock.resolve()),
                    "sha256": sha256_file(self.runtime_lock),
                }
                if self.runtime_lock is not None
                else None
            ),
            "checkpoint": checkpoint,
            "evaluation": dict(self.metadata),
        }


def sha256_file(path: Path) -> str:
    """Hash a file without reading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file or a directory tree including relative paths."""

    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def source_state(workspace: Path) -> dict[str, object]:
    """Capture Git and lockfile state without requiring a repository."""

    state: dict[str, object] = {
        "git_commit": None,
        "git_dirty": None,
        "uv_lock_sha256": None,
    }
    try:
        state["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        state["git_dirty"] = bool(status.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    lock_path = workspace / "uv.lock"
    if lock_path.is_file():
        state["uv_lock_sha256"] = sha256_file(lock_path)
    return state


def benchmark_manifest(
    *,
    run_id: str,
    protocol_id: str,
    model: ModelProvenance,
    data_root: Path,
    task_names: Sequence[str],
    task_config: Mapping[str, object],
    workspace: Path,
) -> dict[str, object]:
    """Build the immutable manifest written before task execution begins."""

    dataset_manifest = data_root / "dataset-manifest.json"
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "protocol_id": protocol_id,
        "model": model.as_manifest(),
        "data": {
            "root": str(data_root.resolve()),
            "manifest": str(dataset_manifest.resolve()) if dataset_manifest.is_file() else None,
            "manifest_sha256": (
                sha256_file(dataset_manifest) if dataset_manifest.is_file() else None
            ),
        },
        "tasks": list(task_names),
        "task_config": dict(task_config),
        "software": {
            "gfmbench_api_version": importlib.metadata.version("gfmbench-api"),
            "gfmbench_api_revision": GFMBENCH_REVISION,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "source": source_state(workspace),
    }


def write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write one formatted JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
