# Model architecture

ATCG-FM uses a decoder-only residual skeleton with an explicit sequence-mixer schedule.
Normalization, residuals, and the feed-forward network are owned by `MixerBlock`; a mixer
only maps `(batch, sequence, channels)` hidden states to an equal-shaped causal result.

The reference operators are:

- PyTorch scaled dot-product causal self-attention with rotary positions.
- Hyena-SE: short explicit gated convolution.
- Hyena-MR: medium explicit gated convolution with a learned exponential envelope.
- Hyena-LI: a long implicit filter evaluated with zero-padded FFT convolution.

`hybrid_tiny` cycles through `SE -> MR -> LI -> attention`. This is inspired by the
operator layout reported for Evo 2, but it is not a port of StripedHyena 2. It is not
checkpoint-compatible with Evo 2, Savanna, or Vortex, and the reference PyTorch operators
do not claim their optimized throughput.

Reference implementations define semantics. A future optimized backend must demonstrate
forward, causal, and gradient parity over multiple sequence lengths and precisions before
it can be used in a scientific comparison.

