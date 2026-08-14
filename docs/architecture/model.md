# Model architecture

ATCG-FM has two explicit substitution boundaries. A mixer comparison changes only the
causal token-mixing operator inside `StandardMixerBlock`; RMSNorm, residual routing,
SwiGLU, dropout, embeddings, and the language-model head remain fixed. A block comparison
replaces the complete `SequenceBlock`, allowing a candidate such as TITANS
Memory-as-Context to own its internal attention, persistent tokens, neural-memory reads and
writes, output integration, normalization, residuals, and MLP.

`ModelConfig.blocks` is a typed block schedule:

- `StandardBlockSpec(mixer=...)` is the controlled mixer shell.
- `TitansMACBlockSpec(...)` is a complete composite-block substitution.

The native mixer specifications are causal attention, Hyena-SE, Hyena-MR, Hyena-LI, and
`TitansMemorySpec`. The latter isolates the TITANS neural long-term memory inside the
standard shell. `hybrid_tiny` cycles through `SE -> MR -> LI -> attention`; it is inspired
by Evo 2 but is not checkpoint-compatible with StripedHyena 2, Savanna, or Vortex.

## Explicit state

Every block implements a segment-level state-in/state-out contract. Stateless attention
and Hyena return `None`; TITANS components return `NeuralMemoryState`. One `ModelState`
contains the state aligned with every block for one logical biological stream.

`GenomicLanguageModel.forward_segment` accepts one `ModelState` per batch row and returns
replacement states in `CausalLMOutput`. The model never owns a mutable stream cache.
`atcg-runtime.StreamStateStore` owns the `stream_id -> ModelState` mapping, resets at stream
starts, removes completed streams, and detaches state at gradient-horizon boundaries.

TITANS neural memory stores:

- functional MLP fast weights;
- surprise momentum for each fast parameter;
- independent causal histories for context queries, output queries, and memory writes.

Adaptive updates use input-dependent decay (`alpha`), surprise momentum (`eta`), and
learning-rate (`theta`) gates. `adaptive`, `frozen`, and `disabled` modes distinguish online
writes, fixed memory retrieval, and a bypassed memory branch.

`TitansMACBlock` reads incoming memory, applies causal attention to
`[persistent; retrieval; current segment]`, writes attention-selected information, and
causally retrieves from the post-update memory at each position through a learned output
gate. Its manual attention implementation is intentional: PyTorch's CPU flash-attention
backward does not currently provide the second-order derivative required to meta-learn
through the fast-weight update.

## Ordered training

`OrderedCausalStreamDataset` emits non-overlapping, source-ordered segments grouped into a
truncated-gradient horizon. `StatefulTrainer` processes those segments in order, carries
state between horizons, and rejects duplicated stream identities within a batch. Ordinary
`Trainer` rejects stateful architectures so shuffled independent windows cannot silently
invalidate a memory experiment.

Checkpoint schema 3 binds checkpoints to the `explicit_block_state_v1` model interface and
the `ordered_segment_causal_v1` execution format. A checkpoint may contain active stream
states in addition to static model and optimizer parameters. Dataset cursor ownership is
still a caller concern; an exact interrupted-run continuation must restore the matching
ordered-data cursor alongside the checkpoint.

Independent-sequence inference is owned by `atcg-runtime`. Its shared forward path resets
state per call, pads each stateful segment with an explicit validity mask, carries detached
state between segments, and enables local gradients only for adaptive memory writes. Native
scoring, full-prefix generation, and the GFMBench adapter all call this same path. Stateful
scoring and generation default to `frozen`; `adaptive` and `disabled` must be declared.

## Comparative claims

`ComparisonPlan` validates the claim boundary before a manifest is written:

- mixer comparisons require standard blocks and identical shells;
- block comparisons may vary complete blocks while keeping the model scaffold fixed;
- concrete manifests report static trainable parameters and dynamic recurrent-state
  elements per stream separately;
- an optional parameter-delta tolerance can reject unmatched candidates.

Reference implementations define semantics. A future scan, fused kernel, or MLX backend
must demonstrate forward, causal, gradient, state-transition, and resume parity before it
can replace them in a scientific comparison.
