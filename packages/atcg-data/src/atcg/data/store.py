"""Filesystem-neutral access to the ATCG-FM object store."""

import json
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import pyarrow.dataset as ds  # pyright: ignore[reportMissingTypeStubs]
import pyarrow.fs as fs  # pyright: ignore[reportMissingTypeStubs]
from pyarrow import NativeFile  # pyright: ignore[reportMissingTypeStubs]

DEFAULT_BUCKET = "draggon-lab-data"


@dataclass(frozen=True, slots=True)
class DataStore:
    """A read-only view of a bucket-shaped dataset directory.

    Paths passed to this class are relative to ``root``. The same interface works
    against Google Cloud Storage in production and a local directory in tests.
    """

    root: str
    filesystem: fs.FileSystem

    def __post_init__(self) -> None:
        if not self.root or self.root.isspace():
            raise ValueError("data store root must not be empty")

    @classmethod
    def gcs(cls, bucket: str = DEFAULT_BUCKET, *, prefix: str = "") -> "DataStore":
        """Connect to GCS using Google Application Default Credentials."""

        root = _join_root(bucket, prefix)
        return cls(root=root, filesystem=fs.GcsFileSystem())

    @classmethod
    def local(cls, root: str | Path) -> "DataStore":
        """Open a local directory with the same layout as the GCS bucket."""

        return cls(root=str(Path(root).resolve()), filesystem=fs.LocalFileSystem())

    def object_path(self, relative_path: str) -> str:
        """Return a filesystem path for an object below this store's root."""

        relative = _validate_relative_path(relative_path)
        return posixpath.join(self.root.rstrip("/"), relative)

    def open_input_file(self, relative_path: str) -> NativeFile:
        """Open an object for random-access binary reading."""

        return cast(NativeFile, self.filesystem.open_input_file(self.object_path(relative_path)))

    def read_json(self, relative_path: str) -> Mapping[str, object]:
        """Read a JSON object from the store."""

        with self.open_input_file(relative_path) as stream:
            value = cast(object, json.loads(stream.read().decode("utf-8")))
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"{relative_path!r} does not contain a JSON object")
        return cast(dict[str, object], value)

    def read_manifest(self) -> Mapping[str, object]:
        """Read the root dataset manifest."""

        return self.read_json("manifest.json")

    def parquet_dataset(self, relative_path: str) -> ds.Dataset:
        """Open a Parquet file or shard directory as a lazy Arrow dataset."""

        return cast(
            ds.Dataset,
            ds.dataset(
                self.object_path(relative_path),
                filesystem=self.filesystem,
                format="parquet",
            ),
        )


def _join_root(bucket: str, prefix: str) -> str:
    if not bucket or bucket.isspace() or "/" in bucket or bucket.startswith("gs://"):
        raise ValueError("bucket must be a bare GCS bucket name")
    if not prefix:
        return bucket
    return posixpath.join(bucket, _validate_relative_path(prefix))


def _validate_relative_path(value: str) -> str:
    if not value or value.isspace():
        raise ValueError("object path must not be empty")
    if value.startswith("gs://") or value.startswith("/"):
        raise ValueError("object path must be relative to the data store root")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("object path must not contain empty, current, or parent components")
    return PurePosixPath(*parts).as_posix()
