# ADR 0001: Native PyTorch workspace

## Status

Accepted

## Context

ATCG-FM is intended to compare sequence-mixing and memory mechanisms. Framework-owned
model structure would make it harder to inspect mathematical behavior and to distinguish
an architectural result from a framework optimization.

## Decision

Use a uv workspace with separate sequence, model, runtime, and evaluation packages. Model
implementations use `torch.nn.Module` directly. The initial training loop uses native
PyTorch APIs and keeps compilation, distributed sharding, and external experiment tracking
optional.

Reference implementations are the semantic baseline. Optimized kernels must be selectable
backends with forward and gradient parity tests.

## Consequences

The repository owns more of its training and checkpoint code. In return, experimental
operators remain inspectable, independently testable, and usable outside one trainer.

