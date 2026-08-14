# ADR 0004: Separate mixer and whole-block substitutions

## Status

Accepted.

## Decision

ATCG-FM represents a model as a typed schedule of `SequenceBlock` specifications. The
controlled `StandardMixerBlock` owns a fixed normalization, residual, and feed-forward
shell and delegates only token interaction to a `SequenceMixer`. Architectures that couple
multiple token-interaction systems or own their residual topology implement
`SequenceBlock` directly.

All blocks use explicit per-stream state input and replacement-state output. The runtime,
not the model, owns stream identity, state lifetime, detachment, and serialization.

## Consequences

TITANS neural memory can be evaluated as a mixer without changing the standard shell, while
TITANS Memory-as-Context can be evaluated as a complete composite block without accidental
double normalization or residuals. Experiment manifests identify the substitution unit and
validate its invariants. Stateful execution requires ordered stream batching and cannot use
the ordinary shuffled-window trainer.
