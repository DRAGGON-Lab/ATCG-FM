import json
from pathlib import Path

import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]
import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]
import pytest

from atcg.data import DEFAULT_BUCKET, DataStore


def test_local_store_reads_manifest_and_parquet_shards(tmp_path: Path) -> None:
    (tmp_path / "curated/sequences").mkdir(parents=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"dataset": {"name": "fixture"}}),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({"sequence": [b"ACGT", b"TGCA"], "length": [4, 4]}),
        tmp_path / "curated/sequences/part-00000.parquet",
    )

    store = DataStore.local(tmp_path)

    assert store.read_manifest()["dataset"] == {"name": "fixture"}
    table = store.parquet_dataset("curated/sequences").to_table(columns=["sequence"])
    assert table.to_pydict() == {"sequence": [b"ACGT", b"TGCA"]}


def test_open_input_file_is_root_relative(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/source.txt").write_bytes(b"source")
    store = DataStore.local(tmp_path)

    with store.open_input_file("raw/source.txt") as stream:
        assert stream.read() == b"source"


@pytest.mark.parametrize(
    "path",
    ["", "/raw/file", "gs://other/file", "../file", "raw/../file", "raw//file", "raw/./file"],
)
def test_store_rejects_paths_outside_its_root(tmp_path: Path, path: str) -> None:
    store = DataStore.local(tmp_path)

    with pytest.raises(ValueError):
        store.object_path(path)


def test_gcs_defaults_name_the_canonical_bucket() -> None:
    assert DEFAULT_BUCKET == "draggon-lab-data"
