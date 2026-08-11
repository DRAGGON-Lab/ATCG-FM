# Data

Raw and processed genomic datasets are not stored in Git. Dataset manifests belong in
`data/manifests/` and should record source URLs, licenses, checksums, preprocessing, split
construction, and sequence-level provenance.

Dataset splits must be constructed before extracting overlapping windows. A random split
of windows from the same source genome is not an acceptable held-out evaluation.

