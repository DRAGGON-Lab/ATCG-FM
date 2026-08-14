# TITANS substitution study

This study separates two claims that must not share one undifferentiated leaderboard.

## Mixer track

Attention, Hyena, and `TitansMemorySpec` are placed inside the same
`StandardMixerBlock`. The estimand is the effect of the token-mixing mechanism while the
normalization, residual, SwiGLU, tokenizer, data order, training tokens, and optimizer
remain fixed.

Run each TITANS memory candidate in `adaptive`, `frozen`, and `disabled` modes. The primary
contrasts are adaptive minus frozen for online writing and frozen minus disabled for fixed
memory retrieval.

## Block track

`TitansMACBlockSpec` replaces the whole standard block. The total MAC effect compares it
with a parameter- and compute-reported standard attention block. Within-MAC adaptive,
frozen, and disabled runs decompose online writes, retrieval, and the persistent-token MAC
layout.

## Required reporting

Construct a `ComparisonPlan` with `substitution_unit` set to `mixer` or `block`. Its
manifest records static trainable parameters and recurrent-state elements per stream. Also
report token budget, wall time, tokens per second, peak memory, gradient horizon, seed, and
GFMBench memory/reset policy. Development runs may use three seeds; a comparative claim
should use the predeclared confirmatory seed count and uncertainty analysis.

GFMBench evaluation resets state for every independent example. An adaptive-memory frozen
probe freezes the outer model parameters but still performs declared within-example fast
weight updates; this must never be described simply as a frozen encoder without the memory
mode qualifier.
