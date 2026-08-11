"""Autoregressive genomic language model."""

from torch import Tensor, nn
from torch.nn import functional as F

from atcg.models.block import MixerBlock
from atcg.models.config import ModelConfig


class GenomicLanguageModel(nn.Module):
    """Decoder-only language model with an explicit sequence-mixer schedule."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    mixer_spec=mixer_spec,
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_hidden_size=config.mlp_hidden_size,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                    norm_eps=config.norm_eps,
                )
                for mixer_spec in config.layer_mixers
            ]
        )
        self.final_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_projection: nn.Linear | None = (
            None
            if config.tie_embeddings
            else nn.Linear(config.d_model, config.vocab_size, bias=False)
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds maximum {self.config.max_seq_len}"
            )

        hidden_states = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        if self.output_projection is not None:
            return self.output_projection(hidden_states)
        return F.linear(hidden_states, self.token_embedding.weight)

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
