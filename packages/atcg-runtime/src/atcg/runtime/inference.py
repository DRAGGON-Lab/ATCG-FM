"""Framework-light scoring and autoregressive generation."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from atcg.models import GenomicLanguageModel, MemoryMode, ModelState
from atcg.sequence import Tokenizer


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


@dataclass(frozen=True, slots=True)
class StatefulInferencePolicy:
    """Leakage-safe policy for independent sequences with recurrent memory."""

    memory_mode: MemoryMode = "frozen"
    reset: Literal["per_sequence"] = "per_sequence"
    max_sequence_length: int | None = None

    def __post_init__(self) -> None:
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError("stateful max_sequence_length must be positive or null")


@dataclass(frozen=True, slots=True)
class IndependentSequenceOutput:
    """Concatenated outputs from one independently reset sequence batch."""

    logits: Tensor
    hidden_states: Tensor


def _model_device(model: GenomicLanguageModel) -> torch.device:
    return next(model.parameters()).device


def forward_independent_sequences(
    model: GenomicLanguageModel,
    input_ids: Tensor,
    *,
    pad_id: int,
    stateful_policy: StatefulInferencePolicy | None = None,
) -> IndependentSequenceOutput:
    """Run an equal-length sequence batch with fresh state and causal segmentation."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence)")
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must contain at least one non-empty sequence")

    model.eval()
    if not model.config.is_stateful:
        if input_ids.shape[1] > model.config.max_seq_len:
            raise ValueError("tokenized sequence exceeds the model context")
        with torch.inference_mode():
            output = model(input_ids)
        return IndependentSequenceOutput(output.logits, output.hidden_states)

    if stateful_policy is None:
        raise ValueError("stateful models require an explicit inference policy")

    segment_length = model.config.segment_length
    states: tuple[ModelState, ...] | None = None
    logits: list[Tensor] = []
    hidden_states: list[Tensor] = []
    for start in range(0, input_ids.shape[1], segment_length):
        stop = min(start + segment_length, input_ids.shape[1])
        valid_length = stop - start
        segment = torch.full(
            (input_ids.shape[0], segment_length),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        segment[:, :valid_length] = input_ids[:, start:stop]
        valid_mask = torch.zeros_like(segment, dtype=torch.bool)
        valid_mask[:, :valid_length] = True
        if stateful_policy.memory_mode == "adaptive":
            with torch.enable_grad():
                output = model.forward_segment(
                    segment,
                    states,
                    valid_mask=valid_mask,
                    memory_mode="adaptive",
                )
            states = tuple(state.detach() for state in output.states)
        else:
            with torch.inference_mode():
                output = model.forward_segment(
                    segment,
                    states,
                    valid_mask=valid_mask,
                    memory_mode=stateful_policy.memory_mode,
                )
            states = output.states
        logits.append(output.logits[:, :valid_length].detach())
        hidden_states.append(output.hidden_states[:, :valid_length].detach())

    return IndependentSequenceOutput(
        logits=torch.cat(logits, dim=1),
        hidden_states=torch.cat(hidden_states, dim=1),
    )


def score_sequence(
    model: GenomicLanguageModel,
    tokenizer: Tokenizer,
    sequence: str,
    *,
    memory_mode: MemoryMode = "frozen",
) -> SequenceScore:
    """Score one complete sequence including its EOS transition."""

    token_ids = tokenizer.encode(sequence, add_bos=True, add_eos=True)
    device = _model_device(model)
    inputs = torch.tensor(token_ids[:-1], dtype=torch.long, device=device)[None, :]
    targets = torch.tensor(token_ids[1:], dtype=torch.long, device=device)
    policy = StatefulInferencePolicy(memory_mode=memory_mode) if model.config.is_stateful else None
    output = forward_independent_sequences(
        model,
        inputs,
        pad_id=tokenizer.pad_id,
        stateful_policy=policy,
    )
    with torch.inference_mode():
        logits = output.logits[0]
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
    memory_mode: MemoryMode = "frozen",
) -> GenerationResult:
    """Generate by full-prefix decoding from the readable reference model."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")
    if temperature < 0.0:
        raise ValueError("temperature must not be negative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive or null")

    token_ids = tokenizer.encode(prompt, add_bos=True)
    if not model.config.is_stateful and len(token_ids) + max_new_tokens > model.config.max_seq_len:
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
    policy = StatefulInferencePolicy(memory_mode=memory_mode) if model.config.is_stateful else None
    for _ in range(max_new_tokens):
        inputs = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
        output = forward_independent_sequences(
            model,
            inputs,
            pad_id=tokenizer.pad_id,
            stateful_policy=policy,
        )
        next_logits = output.logits[0, -1].float()
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
