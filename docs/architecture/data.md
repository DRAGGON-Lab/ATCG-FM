# Dataset storage

The canonical dataset root is `gs://draggon-lab-data`. The bucket holds upstream source
artifacts, analysis-friendly Parquet tables, and their provenance manifests. The repository
does not wrap object storage in a project-specific CLI. Publishers normally maintain a
bucket-shaped local directory with `gcloud storage rsync`; `rclone` remains a portable
alternative for moving and comparing the prepared directory tree.

## Storage layout

The prepared directory is the bucket root. Keep the shared layout shallow:

```text
raw/
  ncbi/
    packages/
    batch-manifests/
  gtdb/
    r220/
curated/
  gtdb/
    metadata.parquet
    taxonomy.parquet
  ncbi/
    assemblies.parquet
  sequences/
    part-<number>.parquet
representations/
  nonoverlap_6mer_v1/
    manifest.json
    tokenizer_spec.json
    index.jsonl
    stream_metadata.parquet
    tokens_<number>.npy
    base_counts_<number>.npy
manifest.json
```

`raw/` is organized by upstream provider and release. `curated/` is organized by data kind
and contains reproducible, model-neutral derivatives of the raw inputs. `representations/`
is organized by a named encoding contract and contains training-ready derivatives. If the
project later needs multiple independent source datasets, add clearly named top-level
directories only when that need exists; do not add release and alias layers preemptively.

The layers have different responsibilities:

- `raw/` contains byte-preserved upstream inputs: the NCBI package ZIPs and their original
  batch manifests, plus the compressed GTDB R220 tables. These are provenance artifacts,
  not efficient training inputs.
- `curated/gtdb/` contains normalized GTDB metadata and taxonomy tables, and
  `curated/ncbi/assemblies.parquet` reconciles requested assemblies with package contents
  and status.
- `curated/sequences/` contains sharded Parquet rows with binary DNA and sequence-level
  provenance. This is the portable, directly queryable sequence table from which training
  encodings can be built.
- `representations/nonoverlap_6mer_v1/` contains the fixed-vocabulary, ordered six-mer token
  streams used by stateful long-context training. It is named for its encoding contract,
  not for a particular model architecture, so other consumers can reuse it.

The six-mer representation contains 303,080 contig or replicon streams, 2,517,365,589
tokens, and exact spans covering 15,010,066,854 represented bases. Its 51 token and 51
base-count NumPy shards are memory-mapped by `synbio-torch`. `index.jsonl` preserves stream
boundaries and ANI-aware train, validation, and test assignments; `stream_metadata.parquet`
retains accession, contig, ANI clade, GC fraction, and split metadata. The representation's
own manifest binds these files to the `nonoverlap_6mer_v1` tokenizer checksum and the
curated source fingerprint.

The current source snapshot comprises 163 checksum-verified NCBI package ZIPs requesting
3,250 unique assemblies, with 3,249 FASTA files. `GCA_902363335.1` is suppressed upstream
and must remain an explicit missing/suppressed record rather than being dropped silently.
The source also includes two compressed GTDB R220 metadata tables. The manifest stored with
the data is the authority for the snapshot's full inventory and checksums.

## Maintain a local publisher mirror

A publisher can keep the complete bucket-shaped tree in any stable local directory. Set a
shell variable once so the source is visible in every command below:

```console
ATCG_DATA_ROOT=/path/to/data
```

To create a local mirror from an existing bucket, synchronize in the download direction:

```console
mkdir -p "$ATCG_DATA_ROOT"
gcloud storage rsync \
  gs://draggon-lab-data \
  "$ATCG_DATA_ROOT" \
  --recursive \
  --checksums-only
```

This is a one-way synchronization, not bidirectional conflict resolution. The first argument
is always the source and the second is always the destination. Finish or discard local edits
before pulling over the same paths.

### Publish additive updates

Before publishing, confirm that the variable resolves to the intended, populated dataset
root. These checks are deliberately simple guards against an empty path, an unmounted disk,
or selecting the wrong directory:

```console
test -s "$ATCG_DATA_ROOT/manifest.json"
test -d "$ATCG_DATA_ROOT/raw"
test -d "$ATCG_DATA_ROOT/curated"
test -d "$ATCG_DATA_ROOT/representations"
```

Preview the upload first:

```console
gcloud storage rsync \
  "$ATCG_DATA_ROOT" \
  gs://draggon-lab-data \
  --recursive \
  --checksums-only \
  --dry-run
```

Review the source and destination printed by the command, then publish without deleting
unmatched remote objects:

```console
gcloud storage rsync \
  "$ATCG_DATA_ROOT" \
  gs://draggon-lab-data \
  --recursive \
  --checksums-only
```

This is the normal update command. It uploads new and changed objects while preserving any
remote objects absent from the local directory.

### Publish an exact mirror

Adding `--delete-unmatched-destination-objects` makes the local directory authoritative for
the entire bucket and can delete remote data quickly. Use it only for an intentional prune,
never as the routine upload command. First confirm that the bucket has the expected soft
delete recovery policy:

```console
gcloud storage buckets describe gs://draggon-lab-data \
  --format="default(soft_delete_policy)"
```

Then preview the exact operation, including every proposed deletion:

```console
gcloud storage rsync \
  "$ATCG_DATA_ROOT" \
  gs://draggon-lab-data \
  --recursive \
  --checksums-only \
  --delete-unmatched-destination-objects \
  --dry-run
```

Only after reviewing the complete dry-run output should a publisher execute the exact-mirror
publication:

```console
gcloud storage rsync \
  "$ATCG_DATA_ROOT" \
  gs://draggon-lab-data \
  --recursive \
  --checksums-only \
  --delete-unmatched-destination-objects
```

Run the destructive dry-run once more afterward; a fully matched local directory and bucket
should produce no copy or delete operations.

Never run the destructive form when the local source is new, empty, partially downloaded,
temporarily unavailable, or mid-rebuild. Do not reverse the source and destination. Only one
publisher should perform an exact-mirror operation at a time: a stale local mirror can erase
objects another publisher added. Google documents the deletion behavior in the
[`gcloud storage rsync` reference](https://cloud.google.com/sdk/gcloud/reference/storage/rsync)
and recovery policy in the
[Cloud Storage soft-delete guide](https://cloud.google.com/storage/docs/use-soft-delete).

## Transfer with rclone

Install `rclone`, then create a remote named `gcs`. Select **Google Cloud Storage** when the
interactive configuration asks for a storage backend and complete browser authentication:

```console
rclone config
```

Upload the contents of the prepared data directory to the bucket:

```console
rclone copy /path/to/data gcs:draggon-lab-data --progress --checksum
```

`copy` is safe to repeat: it skips matching objects and does not delete destination
objects. After the upload, compare the complete local and remote trees without changing
either one:

```console
rclone check /path/to/data gcs:draggon-lab-data
```

Use `rclone sync` only when deletion is intentional; it makes the destination match the
source and can remove destination objects. Preview any such operation with `--dry-run`.

## Access from Python

The `atcg-data` workspace package provides read-only access through PyArrow. It uses Google
Application Default Credentials rather than storing credentials or service-account keys in
the repository. For local development, initialize those credentials once:

```console
gcloud auth application-default login
```

On GCP, attach a read-only service account to the workload instead. Library code then uses
the same API in local scripts, notebooks, training jobs, and managed compute:

```python
from atcg.data import DataStore

store = DataStore.gcs()
manifest = store.read_manifest()

sequences = store.parquet_dataset("curated/sequences")
for batch in sequences.to_batches(columns=["sequence"], batch_size=1_024):
    train_on(batch)
```

PyArrow discovers the Parquet shards lazily and supports projection and filter pushdown;
callers do not need to download or materialize the complete dataset. Tests and offline
workflows can address an identically shaped local directory with `DataStore.local(path)`.

The NumPy training representation is different: NumPy memory mapping requires ordinary
local files. Copy that prefix to node-local storage before opening it with `synbio-torch`:

```console
gcloud storage rsync \
  gs://draggon-lab-data/representations/nonoverlap_6mer_v1 \
  /path/to/nonoverlap_6mer_v1 \
  --recursive
```

```python
from synbiotorch.datasets import TokenStreamStore

streams = TokenStreamStore("/path/to/nonoverlap_6mer_v1")
```

## Cloud Storage policy

Keep the bucket in the same region as the expected primary GPU or TPU workloads. Start with
Standard storage and change storage policy only after observing access patterns.

IAM should separate responsibilities:

- publisher identities may write dataset objects;
- training and analysis identities receive read-only access;
- ordinary jobs do not receive delete permission.

For repeated training epochs, copy shards to node-local SSD or use a bounded Cloud Storage
FUSE cache. Record the input object paths and manifest fingerprint in every run even when
bytes are cached locally.

## Scientific constraints

Splits are assigned at the source-genome or other justified biological grouping before
overlapping windows are extracted. Derived tables and training artifacts retain accession
versions, sequence checksums, upstream releases, transformation versions, and split
assignments. Numerical convenience must not erase suppressed inputs, parse failures,
duplicate identities, or other discrepancies: each is reconciled or declared in the
manifest.
