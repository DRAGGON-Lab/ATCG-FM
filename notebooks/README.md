# Notebooks

Notebooks are for exploratory analysis and figures. They should import ATCG-FM packages
and must not contain the only implementation of a tokenizer, model, metric, or transform.

## Colab Transformer–TITANS hybrid pilot

Dataset creation is now owned by [Ecole](https://github.com/DRAGGON-Lab/Ecole). Use a published, checksum-verified Ecole release and record its manifest fingerprint before model training. The original `00_prepare_ecoli_dataset.ipynb` was migrated to Ecole as an archived historical reference; its maintained implementation is the Ecole CLI.\n\nThe numbered training workflow reuses the shared `bacteria_titan_v1_ecoli_related_15gbp` corpus:

1. [Ecole](https://github.com/DRAGGON-Lab/Ecole) selects metadata-defined ANI-99 groups and writes one 16M-base training set plus four 2M-base evaluation sets. Training genomes contribute guarded within-genome validation/test coordinates; separate source validation/test genomes retain their ANI-clade holdout.
2. `01_train_transformer.ipynb` benchmarks the T4, freezes the shared profile and ordered
   schedule, trains the Transformer control, and evaluates both generalization regimes.
3. `02_train_titans.ipynb` consumes the immutable configuration and evaluates adaptive
   memory plus a segment-reset ablation.
4. `03_compare_results.ipynb` checks the comparison contract, reports stream-offset quality,
   and bootstraps paired stream-level Transformer–TITANS differences.

The two test sets answer different questions. `test_within` measures unseen coordinates
from known genomes; `test_clade` measures transfer to ANI-99-separated genomes. Both must
be reported. GPU notebooks require an NVIDIA T4 and retain Colab's CUDA-enabled PyTorch.

Set the optional `ECOLE_DATASET_ROOT` environment variable in Colab to the parent of `stage-small/` for an Ecole release. Without it, the historical Drive path is used. Each notebook checks the schema and split checksums before training or comparison.
