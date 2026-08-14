# ADR 0003: GFMBench as the evaluation substrate

## Status

Accepted

## Context

ATCG-FM needs modern downstream and zero-shot evaluation across native checkpoints and
external genomic foundation models. Maintaining a separate generic task and model API would
duplicate GFMBench-API and make comparisons depend on two evolving protocol layers.

Training validation has different responsibilities from downstream benchmarking. It needs
to remain lightweight and available without Transformers, dataset download clients, or
external model runtimes.

## Decision

Build `atcg-eval` directly on GFMBench's abstract model and task classes. Pin GFMBench to an
exact Git revision. Use upstream concrete task implementations directly and add future ATCG
tasks by subclassing the same base classes.

Keep language-model validation in `atcg-runtime`. Install downstream evaluation through the
workspace's `eval` dependency group. Keep external model runtimes in separate environments.

ATCG owns strict failure semantics and long-form artifacts. It does not use GFMBench's wide
CSV report as the canonical scientific record.

## Consequences

GFMBench protocol changes are explicit dependency upgrades rather than transparent changes
from a floating branch. The evaluation environment is larger than the training environment,
but model development and correctness CI can remain lean.

ATCG does not own duplicate benchmark abstractions. It does own adapter correctness,
dataset and checkpoint provenance, explicit unsupported capabilities, and extensions for
likelihood, long-context, and efficiency studies.
