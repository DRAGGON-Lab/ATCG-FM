# Contributing

ATCG-FM favors small, testable changes that preserve the distinction between reference
mathematics, optimized implementations, and experimental conclusions.

## Development checks

```console
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Research changes

Changes to models, datasets, or evaluations should state:

1. The scientific question or implementation claim.
2. The baseline and controlled variables.
3. The validation performed and its limitations.
4. Any change to checkpoint, tokenizer, or dataset semantics.

Optimized operators must retain a readable reference implementation and numerical parity
tests. Notebooks may demonstrate an experiment but must not contain the only copy of
canonical model or data logic.

