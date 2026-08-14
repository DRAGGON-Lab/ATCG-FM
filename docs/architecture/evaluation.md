# Evaluation architecture

## Boundary

`atcg-eval` is an opinionated extension of
[GFMBench-API](https://github.com/NVIDIA/GFMBench-api) rather than an independent
benchmark middleware. GFMBench owns its model contract, task base classes, concrete task
implementations, dataset acquisition, and task metrics. ATCG owns model adapters, protocol
selection, strict execution, provenance, and normalized research artifacts.

GFMBench is a Git dependency pinned to revision
`ce1be5d98c37b9f9ceedc32b254fe38668f989d8`. Updating that revision is a scientific
protocol change and requires adapter, task-registry, and result-contract validation.

Training validation is deliberately outside this package. Dataset-level next-token loss,
bits per token, perplexity, and token accuracy live in `atcg-runtime.validation` because
they are part of the training substrate. `atcg-eval` is for downstream and comparative
studies.

## Model providers

The CLI resolves a provider request into three objects: a GFMBench model, a maximum sequence
length, and `ModelProvenance`. Provider imports are lazy. The workspace can list and test
provider contracts without importing Transformers, Vortex, or downloading weights.

`ModelProvenance` separates the reported model ID from its runtime reference, immutable
revision, runtime-source revision, and optional local checkpoint. A local file is hashed; a
local checkpoint directory receives a deterministic tree hash. Remote models are valid
without a local checkpoint, but a missing immutable revision remains visible as `null` rather
than being presented as reproducible. External providers also hash their model-specific uv
lockfile into the model record.

### Native ATCG

`AtcgGFMModel` subclasses GFMBench's `BaseGFMModel`. For the current character-level causal
model it returns:

- the probability assigned to every observed nucleotide, aligned to base coordinates;
- final normalized hidden states aligned to the same bases;
- either the causal last-base state or the mean base state as the sequence representative.

The adapter reserves one position for BOS. Stateless models use
`model.config.max_seq_len - 1` as their nucleotide limit. Stateful models require an
explicit `StatefulInferencePolicy`; they are processed in ordered fixed-width segments and
may set a larger evaluation cap. State always resets for every independent GFMBench
sequence. The manifest distinguishes `adaptive`, `frozen`, and `disabled` memory so a
"frozen backbone" cannot conceal test-time fast-weight updates. GFMBench boundary padding
`P` is mapped explicitly to `N`. Ragged batches are rejected rather than ambiguously padded.
The policy and segmented forward implementation live in `atcg-runtime` and are shared with
the native score and generation commands; the benchmark adapter adds only base-coordinate
alignment and evaluation-specific length limits.

Masked-token prediction and supervised labels are unsupported by a bare causal checkpoint.
Those methods return `None`, which GFMBench exposes as unsupported metrics. For supervised
classification, the strict runner fits a deterministic logistic probe on GFMBench's training
split and wraps the unchanged backbone in `FrozenProbeGFMModel`. Single-sequence features use
the sequence representative. Variant-pair features concatenate the variant representation,
reference representation, and their difference. Full backbone fine-tuning is not part of
`modern-v1`.

Non-overlapping six-mer checkpoints must not use the character adapter. They require a
separate adapter with a truthful token-to-base mapping. Likelihood-based nucleotide tasks
remain unsupported until such a model exposes factorized per-base probabilities.

### Evo 2

`Evo2GFMModel` uses the official Arc Institute Vortex runtime. It returns shifted
autoregressive observed-base probabilities and embeddings from an explicit intermediate
layer. The first base receives the vocabulary-uniform probability because the official
runtime path does not prepend BOS. Base alignment is verified at runtime. The provider
defaults the 7B family to `blocks.28.mlp.l3`, following the official example, and requires an
explicit layer for non-7B checkpoints.

### Carbon

`CarbonGFMModel` requires the FNS revision and uses `score_sequence()` rather than deriving
single-base probabilities from 6-mer token logits. Hidden states remain 6-mer states in the
model; the adapter broadcasts each state over its six represented bases so probabilities,
embeddings, and position maps share GFMBench's base axis. Carbon only accepts canonical
uppercase DNA. Boundary `P` becomes `A`, matching Carbon's documented right-padding base;
other ambiguity symbols are errors.

### NTv3 and JEPA-DNA

`Ntv3GFMModel` uses the official single-base masked-LM checkpoints through Transformers remote
code. Inputs are padded to a multiple of 128 for the U-Net, while outputs are trimmed back to
the biological sequence. It provides embeddings and masked alternate/reference probabilities,
but no unmasked pseudo-likelihood.

The initial `jepa-dna` provider composes the released NTv3-100M target-encoder state dict with
that NTv3 runtime. It checks that checkpoint keys actually match the backbone and records the
complete non-strict load report. DNABERT-2 and HyenaDNA JEPA checkpoints remain separate future
adapters because their token-to-base mappings differ.

## Strict execution

The ATCG runner always supplies these task controls:

- `disable_safe_model_call = true`, so model exceptions cannot become missing scores;
- `num_workers = 0`, so data iteration is deterministic;
- a finite inference-cache size;
- an explicit model context limit;
- an optional bounded sample count for smoke runs.

The runner refuses to reuse a non-empty output directory. It writes `manifest.json` before
task execution, then updates `records.jsonl` and `summary.csv` after every task. A task
exception is persisted as a failed `__task__` record before the runner raises.

GFMBench's wide `BenchmarkReport` CSV is not the canonical research artifact. ATCG records
one row per run, model, task, split, seed, and metric with explicit `succeeded`,
`unsupported`, or `failed` status. The manifest records provider/model revisions, optional
checkpoint and dataset manifest hashes, GFMBench version and revision, Git state, and lockfile
identity.

## Versioned protocol

`modern-v1` currently names a capability-diverse subset of upstream tasks:

- GUE promoter and splice-site classification;
- VariantBenchmarks non-coding prediction;
- Vepeval ClinVar embedding-distance evaluation;
- SongLab ClinVar and BRCA1 likelihood evaluation;
- TraitGym Mendelian evaluation.

The task registry imports the upstream classes directly. It does not copy their datasets or
metrics. The runner fits the protocol's frozen probe independently for each supervised task.

Evo 2, NTv3, Carbon, and JEPA-DNA are evaluated from model-specific uv projects under
`environments/models`. They launch the in-tree evaluator and emit the same ATCG run artifacts,
but their incompatible runtime dependencies and GPU wheel policies do not enter the core
training environment.
