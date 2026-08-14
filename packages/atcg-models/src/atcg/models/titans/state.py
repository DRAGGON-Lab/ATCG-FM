"""Explicit fast-weight state for TITANS neural memory."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

TensorMap = OrderedDict[str, Tensor]


@dataclass(frozen=True, slots=True)
class NeuralMemoryState:
    """Fast weights, surprise momentum, and independent causal histories."""

    fast_weights: Mapping[str, Tensor]
    surprise: Mapping[str, Tensor]
    context_query_history: Tensor
    output_query_history: Tensor
    write_history: Tensor

    def __post_init__(self) -> None:
        if tuple(self.fast_weights) != tuple(self.surprise):
            raise ValueError("fast weights and surprise must have identical keys")
        for name, value in self.fast_weights.items():
            if value.shape != self.surprise[name].shape:
                raise ValueError(f"surprise[{name!r}] must match its fast weight")
        histories = (
            self.context_query_history,
            self.output_query_history,
            self.write_history,
        )
        if any(history.ndim != 2 for history in histories):
            raise ValueError("TITANS projection histories must be two-dimensional")
        if len({tuple(history.shape) for history in histories}) != 1:
            raise ValueError("TITANS projection histories must have equal shapes")

    def detach(self) -> NeuralMemoryState:
        return NeuralMemoryState(
            fast_weights=OrderedDict(
                (name, value.detach().requires_grad_(True))
                for name, value in self.fast_weights.items()
            ),
            surprise=OrderedDict((name, value.detach()) for name, value in self.surprise.items()),
            context_query_history=self.context_query_history.detach(),
            output_query_history=self.output_query_history.detach(),
            write_history=self.write_history.detach(),
        )

    def to(self, device: torch.device | str) -> NeuralMemoryState:
        target = torch.device(device)
        return NeuralMemoryState(
            fast_weights=OrderedDict(
                (name, value.to(target)) for name, value in self.fast_weights.items()
            ),
            surprise=OrderedDict((name, value.to(target)) for name, value in self.surprise.items()),
            context_query_history=self.context_query_history.to(target),
            output_query_history=self.output_query_history.to(target),
            write_history=self.write_history.to(target),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": "titans_neural_memory",
            "format_version": 1,
            "fast_weights": OrderedDict(
                (name, value.detach().cpu().clone()) for name, value in self.fast_weights.items()
            ),
            "surprise": OrderedDict(
                (name, value.detach().cpu().clone()) for name, value in self.surprise.items()
            ),
            "context_query_history": self.context_query_history.detach().cpu().clone(),
            "output_query_history": self.output_query_history.detach().cpu().clone(),
            "write_history": self.write_history.detach().cpu().clone(),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> NeuralMemoryState:
        if payload.get("kind") != "titans_neural_memory" or payload.get("format_version") != 1:
            raise ValueError("unsupported TITANS neural-memory state")
        raw_fast = payload.get("fast_weights")
        raw_surprise = payload.get("surprise")
        context = payload.get("context_query_history")
        output = payload.get("output_query_history")
        write = payload.get("write_history")
        if not isinstance(raw_fast, Mapping) or not isinstance(raw_surprise, Mapping):
            raise TypeError("invalid TITANS fast-weight mappings")
        if (
            not isinstance(context, Tensor)
            or not isinstance(output, Tensor)
            or not isinstance(write, Tensor)
        ):
            raise TypeError("invalid TITANS projection histories")
        fast = _tensor_map(cast(Mapping[object, object], raw_fast), requires_grad=True)
        surprise = _tensor_map(cast(Mapping[object, object], raw_surprise), requires_grad=False)
        return cls(
            fast_weights=fast,
            surprise=surprise,
            context_query_history=context.detach().clone(),
            output_query_history=output.detach().clone(),
            write_history=write.detach().clone(),
        )


def _tensor_map(values: Mapping[object, object], *, requires_grad: bool) -> TensorMap:
    result: TensorMap = OrderedDict()
    for name, value in values.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("TITANS state mappings must contain string-to-tensor entries")
        result[name] = value.detach().clone().requires_grad_(requires_grad)
    return result
