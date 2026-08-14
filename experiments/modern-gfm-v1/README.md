# Modern GFM v1

## Question

How do ATCG attention and hybrid architectures compare with modern genomic foundation-model
checkpoints across frozen representation quality, zero-shot variant-effect prediction, and
computational efficiency?

## Comparison boundaries

External pretrained checkpoints are capability references, not controlled architecture
ablations. Architecture claims are restricted to ATCG models trained with matched corpus,
tokenizer, context, parameter or compute budget, training tokens, and seed policy.

The initial external panel is Evo2-7B, NTv3-100M, Carbon-3B FNS, and JEPA-DNA. HyenaDNA and
DNABERT2 may be retained as calibration controls but are not the headline modern baselines.

## Protocol

The canonical task list is the `modern-v1` registry in `atcg.eval.registry`. Initial
integration uses `--max-samples 100`; those runs test software and adapter semantics and do
not support scientific claims. Full supervised probes use at least three seeds. Deterministic
zero-shot runs report sample-level uncertainty rather than redundant model seeds.

Every run uses a new output directory containing a manifest, long-form JSONL records, and a
CSV summary. Missing model capabilities remain explicit `unsupported` records. Exceptions
are `failed` records and stop the run.

## Stages

1. Validate ATCG character-level probability, embedding, pooling, and position mapping.
2. Run bounded zero-shot GFMBench tasks with an ATCG checkpoint.
3. Validate the frozen-probe path on bounded supervised tasks.
4. Validate the locked Evo2, Carbon, NTv3, and NTv3-backed JEPA-DNA runtime adapters on GPU.
5. Run a common 100-sample smoke matrix and retain its manifests and failure records.
6. Run the full protocol and analyze capability and efficiency separately.
7. Train matched ATCG attention and hybrid baselines for architecture comparisons.

## Runtime matrix

| Provider | Zero-shot likelihood | Masked SNV | Frozen representation | Initial checkpoint |
| --- | --- | --- | --- | --- |
| ATCG | base-aligned causal | no | last or mean | native schema 3 |
| Evo 2 | base-aligned causal | no | explicit intermediate layer | Evo2-7B |
| Carbon | FNS base probability | no | broadcast 6-mer state | Carbon-3B FNS |
| NTv3 | no | yes | final single-base state | NTv3-100M pre |
| JEPA-DNA | no | yes | final target-encoder state | NTv3-100M target |

This is a capability matrix, not a license to compare every metric across every model. A model
that does not expose a scientifically valid output receives `unsupported`; the runner never
synthesizes a substitute score.
