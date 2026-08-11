# ATCG-FM

ATCG-FM is a native-PyTorch research workspace for genomic foundation models. It is
organized around explicit, interchangeable sequence-mixing operators so attention,
convolution, recurrence, and memory mechanisms can be studied without hiding their
mathematics behind a training framework.

The repository is in its initial implementation phase. Its first target is a small,
autoregressive DNA language model with a conventional attention baseline and an
Evo 2-inspired hybrid mixer schedule.

## Workspace

The uv workspace contains five packages:

- `atcg-cli`: command-line training, evaluation, scoring, and generation.
- `atcg-sequence`: genomic sequence representation, tokenization, and datasets.
- `atcg-models`: PyTorch model components and sequence mixers.
- `atcg-runtime`: training, checkpointing, scoring, and generation.
- `atcg-eval`: scientific evaluation tasks and metrics.

Install all development dependencies and run the test suite with:

```console
uv sync
uv run pytest
```

The initial Linux resolution deliberately uses PyTorch's CPU wheel index so correctness CI
does not install unused CUDA runtimes. A CUDA or ROCm profile has not yet been selected;
that choice will be pinned and validated against the first training cluster rather than
guessed in the base environment.

Raw datasets, checkpoints, and local run artifacts are deliberately excluded from Git.
Tracked dataset manifests and finalized research reports will record their provenance.

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

uv run atcg evaluate \
  --checkpoint runs/tiny/checkpoints/last.pt \
  --fasta tests/fixtures/tiny.fa

uv run atcg generate \
  --checkpoint runs/tiny/checkpoints/last.pt \
  --prompt ACGT \
  --tokens 8
```

Each training run writes its resolved model and training settings, environment metadata,
Git and lockfile provenance, per-step JSONL metrics, and an atomic final checkpoint. The
current CLI intentionally uses the fixed IUPAC nucleotide tokenizer; byte-level research
is available through the Python API.

## Status

ATCG-FM is experimental research software. Model outputs are predictions or generated
sequences, not validated biological findings.
