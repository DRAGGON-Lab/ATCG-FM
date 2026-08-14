# Experiments

Each study gets its own directory with a short question, fixed baselines, controlled
variables, configuration files, and an analysis script. Canonical model and dataset code
belongs in packages rather than an experiment directory.

Development results from one seed are diagnostic. Comparative claims should report at
least three seeds, uncertainty, training tokens, parameter counts, and whether the study
matched data, compute, or elapsed time.

Native architecture comparisons must declare `substitution_unit: mixer` or
`substitution_unit: block`. Mixer comparisons retain `StandardMixerBlock` and vary only its
mixer. Block comparisons may replace the complete computational block. Report static
trainable parameters and recurrent state per active stream separately.
