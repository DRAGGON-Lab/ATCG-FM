# ATCG-FM

ATCG-FM is a native-PyTorch research workspace for genomic foundation models. It is
organized around explicit mixer-level and whole-block substitution boundaries so attention,
convolution, recurrence, composite attention-memory blocks, and neural fast weights can be
studied without hiding their mathematics behind a training framework.

The native model supports attention and Evo 2-inspired Hyena controls, TITANS neural memory
inside the shared standard block shell, and complete TITANS Memory-as-Context blocks with
explicit per-stream state.

## Workspace

The uv workspace contains six packages:

- `atcg-cli`: command-line training, scoring, and generation.
- `atcg-data`: read-only access to the shared GCS bucket and its Parquet datasets.
- `atcg-sequence`: genomic sequence representation, tokenization, and datasets.
- `atcg-models`: PyTorch model components and sequence mixers.
- `atcg-runtime`: training, validation, checkpointing, scoring, and generation.
- `atcg-eval`: strict, provenance-rich evaluation built on GFMBench-API.

The core workspace supports Python 3.12 and 3.13.

Install all development dependencies and run the test suite with:

```console
uv sync
uv run pytest
```

Evaluation is a separate dependency group because GFMBench brings dataset, Transformers,
and scientific Python dependencies that model development does not otherwise need:

```console
uv sync --group eval
uv run --group eval pytest packages/atcg-eval/tests
uv run --group eval atcg-eval list-tasks
uv run --group eval atcg-eval list-providers
```

The initial Linux resolution deliberately uses PyTorch's CPU wheel index so correctness CI
does not install unused CUDA runtimes. A CUDA or ROCm profile has not yet been selected;
that choice will be pinned and validated against the first training cluster rather than
guessed in the base environment.

Raw datasets, checkpoints, and local run artifacts are deliberately excluded from Git.
Tracked dataset manifests and finalized research reports will record their provenance.
See the [dataset storage notes](docs/architecture/data.md) for the bucket layout and
portable transfer commands.

## Tiny end-to-end run

The checked-in FASTA fixture is synthetic and is only intended to validate the software
path:

```console
uv run atcg train \
  --fasta tests/fixtures/tiny.fa \
  --run-dir runs/tiny \
  --architecture hybrid \
  --steps 5 \
  --context-length 16 \
  --d-model 32 \
  --layers 4 \
  --heads 4

uv run atcg generate \
  --checkpoint runs/tiny/checkpoints/last.pt \
  --prompt ACGT \
  --tokens 8
```

Each training run writes its resolved model and training settings, environment metadata,
Git and lockfile provenance, per-step JSONL metrics, and an atomic final checkpoint. The
current CLI intentionally uses the fixed IUPAC nucleotide tokenizer; byte-level research
is available through the Python API. Dataset-level language-model validation belongs to
`atcg-runtime`; downstream scientific evaluation belongs to `atcg-eval`.

Stateful local candidates use ordered segments and an explicit gradient horizon:

```console
uv run atcg train \
  --fasta tests/fixtures/tiny.fa \
  --run-dir runs/titans-memory-tiny \
  --architecture titans-memory \
  --steps 5 \
  --context-length 16 \
  --gradient-horizon 2 \
  --d-model 32 \
  --layers 1
```

Use `--architecture titans-mac` for the complete Memory-as-Context block. These paths reject
shuffled independent windows and checkpoint active per-stream fast state.

Scoring and generation use the same per-sequence reset and fixed-width stateful segmentation
as GFMBench. Stateful inference defaults to fixed memory retrieval; select online writes
explicitly when required:

```console
uv run atcg score \
  --checkpoint runs/titans-memory-tiny/checkpoints/last.pt \
  --sequence ACGTACGT \
  --memory-mode adaptive

uv run atcg generate \
  --checkpoint runs/titans-memory-tiny/checkpoints/last.pt \
  --prompt ACGT \
  --tokens 8 \
  --memory-mode frozen
```

## Modern foundation-model evaluation

`atcg-eval` uses [GFMBench-API](https://github.com/NVIDIA/GFMBench-api)'s model and task
contracts directly. ATCG adds a strict runner,
an ATCG checkpoint adapter, a versioned task protocol, and long-form result artifacts.
GFMBench is pinned to an exact upstream revision in `uv.lock`.

The first protocol can be inspected without downloading data:

```console
uv run --group eval atcg-eval list-tasks
```

To run a bounded zero-shot task, provide an unused output directory. GFMBench downloads
missing task data beneath the requested data directory:

```console
uv run --group eval atcg-eval run \
  --provider atcg \
  --checkpoint runs/tiny/checkpoints/last.pt \
  --model-id atcg-hybrid-tiny \
  --task vepeval_clinvar \
  --data-dir data/gfmbench \
  --output-dir runs/eval/vepeval-smoke \
  --max-samples 100
```

Evo 2, Carbon, NTv3, and the NTv3-backed JEPA-DNA target encoder use the same command and
artifact contract from isolated, locked uv projects under
[`environments/models`](environments/models/README.md). Their CUDA and Transformers
dependencies do not enter the native training environment.

See [the evaluation architecture](docs/architecture/evaluation.md) for the scientific and
software boundaries.

## Status

ATCG-FM is experimental research software. Model outputs are predictions or generated
sequences, not validated biological findings.
