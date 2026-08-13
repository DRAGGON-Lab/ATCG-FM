# Data

Raw and derived genomic data are not stored in Git. The canonical data bucket is
`gs://draggon-lab-data`; this directory contains only small, reviewed manifest records.

See [the dataset storage notes](../docs/architecture/data.md) for the bucket layout and
portable `rclone` transfer commands.

Dataset splits must be constructed before extracting overlapping windows. A random split
of windows from the same source genome is not an acceptable held-out evaluation.
