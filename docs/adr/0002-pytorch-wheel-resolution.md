# ADR 0002: PyTorch wheel resolution

## Status

Accepted for the initial CPU correctness phase

## Context

PyPI's Linux PyTorch distribution resolves CUDA runtime packages by default. CPU-only CI
would otherwise download several gigabytes of accelerator libraries that it cannot use.
PyTorch accelerator wheels also depend on the target driver and cluster environment.

## Decision

Resolve PyTorch from its explicit CPU index on Linux and from PyPI on macOS. This keeps the
shared lockfile and CI correctness gate portable. Do not select a CUDA or ROCm build until
the first training hardware target is known.

## Consequences

The current workspace is a CPU reference environment on Linux. Enabling GPU training is a
deliberate follow-up that will add a mutually exclusive, explicitly versioned accelerator
profile and validate it on the target hardware. Model code contains no CPU-only assumption.

