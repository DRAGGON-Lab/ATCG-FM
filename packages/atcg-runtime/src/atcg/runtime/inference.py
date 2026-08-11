"""Framework-light scoring and autoregressive generation."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from atcg.models import GenomicLanguageModel
from atcg.sequence import Tokenizer
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SequenceScore:
    """Autoregressive likelihood summary in explicit units."""

    total_nll: float
    mean_nll: float
    bits_per_token: float
    token_count: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Generated token and text representation with its conditioning prompt."""

    prompt: str
    sequence: str
    token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]


def _model_device(model: GenomicLanguageModel) -> torch.device:
    return next(model.parameters()).device


def score_sequence(
    model: GenomicLanguageModel,
    tokenizer: Tokenizer,
    sequence: str,
) -> SequenceScore:
    """Score one complete sequence including its EOS transition."""

    token_ids = tokenizer.encode(sequence, add_bos=True, add_eos=True)
    if len(token_ids) - 1 > model.config.max_seq_len:
        raise ValueError("tokenized sequence exceeds the model context")
    device = _model_device(model)
    inputs = torch.tensor(token_ids[:-1], dtype=torch.long, device=device)[None, :]
    targets = torch.tensor(token_ids[1:], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        logits = model(inputs)[0]
        losses = F.cross_entropy(logits, targets, reduction="none")
    total_nll = float(losses.sum().item())
    token_count = targets.numel()
    mean_nll = total_nll / token_count
    return SequenceScore(
        total_nll=total_nll,
        mean_nll=mean_nll,
        bits_per_token=mean_nll / math.log(2.0),
        token_count=token_count,
    )


def generate(
    model: GenomicLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    allowed_token_ids: Sequence[int] | None = None,
    seed: int = 0,
) -> GenerationResult:
    """Generate by full-prefix decoding from the readable reference model."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")
    if temperature < 0.0:
        raise ValueError("temperature must not be negative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive or null")

    token_ids = tokenizer.encode(prompt, add_bos=True)
    if len(token_ids) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("requested generation exceeds the model context")
    if allowed_token_ids is not None:
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids must not be empty")
        invalid = [
            token_id for token_id in allowed_token_ids if not 0 <= token_id < tokenizer.vocab_size
        ]
        if invalid:
            raise ValueError(f"allowed token ids are outside vocabulary: {invalid}")

    device = _model_device(model)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated: list[int] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            inputs = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
            next_logits = model(inputs)[0, -1].float()
            next_logits = _restrict_logits(next_logits, allowed_token_ids)
            if temperature == 0.0:
                next_token = int(next_logits.argmax().item())
            else:
                scaled = next_logits / temperature
                if top_k is not None and top_k < scaled.numel():
                    threshold = torch.topk(scaled, top_k).values[-1]
                    scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
                probabilities = F.softmax(scaled, dim=-1)
                next_token = int(torch.multinomial(probabilities, 1, generator=generator).item())
            token_ids.append(next_token)
            generated.append(next_token)
            if next_token == tokenizer.eos_id:
                break

    return GenerationResult(
        prompt=prompt,
        sequence=tokenizer.decode(token_ids),
        token_ids=tuple(token_ids),
        generated_token_ids=tuple(generated),
    )


def _restrict_logits(logits: Tensor, allowed_token_ids: Sequence[int] | None) -> Tensor:
    if allowed_token_ids is None:
        return logits
    restricted = torch.full_like(logits, -torch.inf)
    indices = torch.tensor(tuple(allowed_token_ids), dtype=torch.long, device=logits.device)
    restricted[indices] = logits[indices]
    return restricted
