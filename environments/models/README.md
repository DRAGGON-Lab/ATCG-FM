# External model runtimes

Each directory is an independent uv project with its own lockfile and Linux PyTorch wheel
index. The launcher imports the current ATCG packages directly from this checkout; it avoids
installing them as path dependencies because that would merge the model environment with the
workspace's CPU-only dependency policy.

The common command shape is:

```console
uv sync --project environments/models/<provider> --locked
uv run --project environments/models/<provider> --locked \
  python environments/models/atcg_eval.py <atcg-eval arguments>
```

`list-providers` shows the runtime and model defaults without loading a checkpoint:

```console
uv run --group eval atcg-eval list-providers
```

## Evo 2

The Evo 2 lock pins the official Arc Institute repository at
`53f195997257c56c00e5ef8d33a54f5baad143a6`, PyTorch 2.7.1, and CUDA 12.8 Linux wheels.
Flash Attention 2.8.0.post2 is built against that same PyTorch runtime, so `uv sync` requires
a compatible CUDA development toolkit. The runtime is Linux/CUDA-only. The default is
`evo2_7b` with the paper-recommended
intermediate representation `blocks.28.mlp.l3`; pass `--embedding-layer` when evaluating a
different architecture. Evo 2's 1B, 20B, and 40B checkpoints require FP8/Transformer Engine
and Hopper hardware; the 7B variants can run in bfloat16 without Transformer Engine.

```console
uv run --project environments/models/evo2 --locked \
  python environments/models/atcg_eval.py run \
  --provider evo2 \
  --model-ref evo2_7b \
  --device cuda:0 \
  --task songlab_clinvar \
  --data-dir data/gfmbench \
  --output-dir runs/eval/evo2-7b-songlab \
  --max-samples 100
```

Passing a local Evo 2 snapshot directory through `--checkpoint` adds a deterministic tree
hash to the run manifest. Without it, the manifest records the requested model name and
runtime revision but cannot prove the immutable Hugging Face snapshot selected by Evo 2.

## Carbon

The Carbon adapter defaults to `HuggingFaceBio/Carbon-3B` at the exact FNS revision
`bf6f6bec000ea6ced8cb656d02f3120a24795c91`. It uses `score_sequence()` for observed-base
probabilities and broadcasts each 6-mer hidden state over its six bases for GFMBench's
base-aligned embedding contract. Boundary `P` is replaced by `A`, matching Carbon's own
right-padding convention; other non-ACGT symbols fail explicitly.

```console
uv run --project environments/models/carbon --locked \
  python environments/models/atcg_eval.py run \
  --provider carbon \
  --device cuda \
  --dtype bfloat16 \
  --task traitgym_mendelian \
  --data-dir data/gfmbench \
  --output-dir runs/eval/carbon-3b-traitgym \
  --max-samples 100
```

## NTv3

NTv3 checkpoints are gated on Hugging Face and carry custom non-commercial terms. Review and
accept the model license, then authenticate before syncing weights. The adapter uses the
official single-base Transformers remote code, pads the
U-Net input to a multiple of 128, removes padding from returned embeddings, and exposes masked
nucleotide probabilities. It deliberately returns no unmasked sequence likelihood.

```console
uv run --project environments/models/ntv3 --locked \
  python environments/models/atcg_eval.py run \
  --provider ntv3 \
  --model-ref InstaDeepAI/NTv3_100M_pre \
  --revision <accepted-model-commit> \
  --device cuda \
  --dtype bfloat16 \
  --task gue_splice_site \
  --data-dir data/gfmbench \
  --output-dir runs/eval/ntv3-100m-splice \
  --max-samples 100
```

Use an immutable `--revision`; the gated repository's commit cannot be resolved anonymously
and therefore has no baked-in default.

## JEPA-DNA

The first JEPA-DNA track is the released NTv3-100M target encoder. Download its state dict at
the pinned collection revision, then supply the local file. The loader strips the official
wrapper prefix, requires at least one matching backbone parameter, and records matched,
missing, and unexpected keys in the manifest.

```console
hf download nvidia/NV-JEPA-DNA-NTv3 \
  jepa_dna_ntv3_target.pt \
  --revision be526df5438b375223017051b1a36d8b9dee2f59 \
  --local-dir checkpoints/jepa-dna-ntv3

uv run --project environments/models/jepa-dna --locked \
  python environments/models/atcg_eval.py run \
  --provider jepa-dna \
  --checkpoint checkpoints/jepa-dna-ntv3/jepa_dna_ntv3_target.pt \
  --backbone-ref InstaDeepAI/NTv3_100M_pre \
  --backbone-revision <accepted-model-commit> \
  --device cuda \
  --dtype bfloat16 \
  --task gue_splice_site \
  --data-dir data/gfmbench \
  --output-dir runs/eval/jepa-dna-ntv3-splice \
  --max-samples 100
```

DNABERT-2 and HyenaDNA JEPA target encoders are not silently routed through this adapter;
they need their own token-to-base contracts before joining the benchmark panel.
