# ATCG-FM

ATCG-FM is a native-PyTorch research workspace for genomic foundation models. It is
organized around explicit, interchangeable sequence-mixing operators so attention,
convolution, recurrence, and memory mechanisms can be studied without hiding their
mathematics behind a training framework.

The repository is in its initial implementation phase. Its first target is a small,
autoregressive DNA language model with a conventional attention baseline and an
Evo 2-inspired hybrid mixer schedule.

## Workspace

The uv workspace contains four packages:

- `atcg-sequence`: genomic sequence representation, tokenization, and datasets.
- `atcg-models`: PyTorch model components and sequence mixers.
- `atcg-runtime`: training, checkpointing, scoring, and generation.
- `atcg-eval`: scientific evaluation tasks and metrics.

Install all development dependencies and run the test suite with:

```console
uv sync
uv run pytest
```

Raw datasets, checkpoints, and local run artifacts are deliberately excluded from Git.
Tracked dataset manifests and finalized research reports will record their provenance.

## Status

ATCG-FM is experimental research software. Model outputs are predictions or generated
sequences, not validated biological findings.

