"""TITANS neural long-term memory with exact sequential fast-weight updates."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

from atcg.models.titans.state import NeuralMemoryState, TensorMap


class PaperResidualMemory(nn.Module):
    """Two-layer expanded residual MLP used as functional fast memory."""

    def __init__(self, d_model: int, expansion_factor: int) -> None:
        super().__init__()
        hidden_size = d_model * expansion_factor
        self.in_projection = nn.Linear(d_model, hidden_size)
        self.out_projection = nn.Linear(hidden_size, d_model)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        transformed = self.out_projection(F.gelu(self.in_projection(inputs)))
        return inputs + self.normalization(transformed)


@dataclass(frozen=True, slots=True)
class ParameterGates:
    alpha: Mapping[str, Tensor]
    eta: Mapping[str, Tensor]
    theta: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class MemoryStepOutput:
    retrieval: Tensor
    state: NeuralMemoryState
    mean_alpha: Tensor
    mean_eta: Tensor
    mean_theta: Tensor


class ChannelUpdateGates(nn.Module):
    """Input-dependent decay, momentum, and learning-rate gates by output channel."""

    _LAYERS: ClassVar[dict[str, str]] = {
        "in_projection.weight": "in_projection",
        "in_projection.bias": "in_projection",
        "out_projection.weight": "out_projection",
        "out_projection.bias": "out_projection",
        "normalization.weight": "out_projection",
        "normalization.bias": "out_projection",
    }

    def __init__(
        self,
        d_model: int,
        expansion_factor: int,
        *,
        alpha_initial: float,
        eta_initial: float,
        theta_initial: float,
    ) -> None:
        super().__init__()
        widths = {
            "in_projection": d_model * expansion_factor,
            "out_projection": d_model,
        }
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d_model, 3 * width) for name, width in widths.items()}
        )
        self.initials = (alpha_initial, eta_initial, theta_initial)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initials = self.initials
        with torch.no_grad():
            for module in self.heads.values():
                head = cast(nn.Linear, module)
                width = head.out_features // 3
                head.weight.zero_()
                for index, value in enumerate(initials):
                    start = index * width
                    head.bias[start : start + width].fill_(torch.logit(torch.tensor(value)).item())

    def forward(self, inputs: Tensor) -> ParameterGates:
        per_layer: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        for name, module in self.heads.items():
            alpha, eta, theta = torch.sigmoid(module(inputs)).chunk(3, dim=-1)
            per_layer[name] = (alpha, eta, theta)
        result: dict[str, OrderedDict[str, Tensor]] = {
            name: OrderedDict() for name in ("alpha", "eta", "theta")
        }
        for parameter_name, layer_name in self._LAYERS.items():
            is_matrix = parameter_name.endswith(".weight") and not parameter_name.startswith(
                "normalization"
            )
            for gate_index, gate_name in enumerate(("alpha", "eta", "theta")):
                value = per_layer[layer_name][gate_index]
                result[gate_name][parameter_name] = value.unsqueeze(-1) if is_matrix else value
        return ParameterGates(**result)


class NeuralMemory(nn.Module):
    """Stream-local functional MLP updated by surprise, momentum, and decay."""

    def __init__(
        self,
        d_model: int,
        *,
        expansion_factor: int = 4,
        projection_kernel_size: int = 4,
        alpha_initial: float = 0.001,
        eta_initial: float = 0.9,
        theta_initial: float = 0.001,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.projection_kernel_size = projection_kernel_size
        self.memory = PaperResidualMemory(d_model, expansion_factor)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        def convolution() -> nn.Conv1d:
            layer = nn.Conv1d(
                d_model,
                d_model,
                projection_kernel_size,
                groups=d_model,
                bias=False,
            )
            with torch.no_grad():
                layer.weight.zero_()
                layer.weight[:, 0, -1] = 1.0
            return layer

        self.query_convolution = convolution()
        self.key_convolution = convolution()
        self.value_convolution = convolution()
        self.gates = ChannelUpdateGates(
            d_model,
            expansion_factor,
            alpha_initial=alpha_initial,
            eta_initial=eta_initial,
            theta_initial=theta_initial,
        )

    def reset_update_mechanism(self) -> None:
        """Restore identity causal projections and configured gate priors."""

        with torch.no_grad():
            for convolution in (
                self.query_convolution,
                self.key_convolution,
                self.value_convolution,
            ):
                convolution.weight.zero_()
                convolution.weight[:, 0, -1] = 1.0
        self.gates.reset_parameters()

    def initial_state(self) -> NeuralMemoryState:
        fast_weights = OrderedDict(self.memory.named_parameters())
        reference = next(iter(fast_weights.values()))
        history = reference.new_zeros((self.projection_kernel_size - 1, self.d_model))
        return NeuralMemoryState(
            fast_weights=fast_weights,
            surprise=OrderedDict(
                (name, torch.zeros_like(value)) for name, value in fast_weights.items()
            ),
            context_query_history=history,
            output_query_history=history.clone(),
            write_history=history.clone(),
        )

    def _functional_memory(self, weights: Mapping[str, Tensor], inputs: Tensor) -> Tensor:
        return functional_call(self.memory, OrderedDict(weights.items()), (inputs,), strict=True)

    def _memory_inputs(self, inputs: Tensor) -> Tensor:
        return inputs.to(dtype=next(self.memory.parameters()).dtype)

    def _project(
        self,
        projection: nn.Linear,
        convolution: nn.Conv1d,
        inputs: Tensor,
        history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected = self.projection_kernel_size - 1
        if history.shape != (expected, self.d_model):
            raise ValueError("causal projection history has the wrong shape")
        combined = torch.cat((history.to(inputs), inputs), dim=0)
        projected = F.silu(projection(combined)).transpose(0, 1).unsqueeze(0)
        convolved = convolution(projected).squeeze(0).transpose(0, 1)
        return convolved, combined[-expected:]

    @staticmethod
    def _unit(values: Tensor) -> Tensor:
        epsilon = torch.finfo(values.dtype).eps
        return cast(Tensor, values / values.norm(dim=-1, keepdim=True).clamp_min(epsilon))

    def read_context(
        self,
        state: NeuralMemoryState,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Read incoming memory without mutating fast weights."""

        with torch.autocast(device_type=inputs.device.type, enabled=False):
            memory_inputs = self._memory_inputs(inputs)
            queries, history = self._project(
                self.query_projection,
                self.query_convolution,
                memory_inputs,
                state.context_query_history,
            )
            return self._functional_memory(state.fast_weights, self._unit(queries)), history

    def associative_loss(
        self,
        fast_weights: Mapping[str, Tensor],
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        reconstruction = self._functional_memory(fast_weights, key)
        return 0.5 * (reconstruction - value).square().sum()

    def _surprise_gradient(
        self,
        fast_weights: Mapping[str, Tensor],
        key: Tensor,
        value: Tensor,
        *,
        create_graph: bool,
    ) -> TensorMap:
        with torch.enable_grad():
            loss = self.associative_loss(fast_weights, key, value)
            gradients = torch.autograd.grad(
                loss,
                tuple(fast_weights.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=False,
            )
        return OrderedDict(zip(fast_weights, gradients, strict=True))

    def update_and_read(
        self,
        state: NeuralMemoryState,
        inputs: Tensor,
        *,
        valid_mask: Tensor,
        context_query_history: Tensor | None = None,
        update_memory: bool,
        create_graph: bool,
    ) -> MemoryStepOutput:
        """Causally update at each token and retrieve from that position's memory."""

        if valid_mask.shape != (inputs.shape[0],):
            raise ValueError("valid_mask must match the memory sequence length")
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            memory_inputs = self._memory_inputs(inputs)
            sanitized = memory_inputs * valid_mask.unsqueeze(-1).to(memory_inputs.dtype)
            output_queries, output_history = self._project(
                self.query_projection,
                self.query_convolution,
                sanitized,
                state.output_query_history,
            )
            keys, write_history = self._project(
                self.key_projection,
                self.key_convolution,
                sanitized,
                state.write_history,
            )
            values, value_history = self._project(
                self.value_projection,
                self.value_convolution,
                sanitized,
                state.write_history,
            )
            if not torch.equal(write_history, value_history):
                raise RuntimeError("key and value convolution histories diverged")
            output_queries = self._unit(output_queries)
            keys = self._unit(keys)
            gates = self.gates(sanitized)
            fast_weights = OrderedDict(state.fast_weights.items())
            surprise = OrderedDict(state.surprise.items())
            retrievals: list[Tensor] = []
            for position in range(memory_inputs.shape[0]):
                valid = bool(valid_mask[position].item())
                if valid and update_memory:
                    gradient = self._surprise_gradient(
                        fast_weights,
                        keys[position],
                        values[position],
                        create_graph=create_graph,
                    )
                    next_surprise: TensorMap = OrderedDict()
                    next_weights: TensorMap = OrderedDict()
                    for name in fast_weights:
                        alpha = gates.alpha[name][position]
                        eta = gates.eta[name][position]
                        theta = gates.theta[name][position]
                        next_surprise[name] = eta * surprise[name] - theta * gradient[name]
                        next_weights[name] = (1.0 - alpha) * fast_weights[name] + next_surprise[
                            name
                        ]
                    fast_weights, surprise = next_weights, next_surprise
                if valid:
                    retrievals.append(
                        self._functional_memory(fast_weights, output_queries[position])
                    )
                else:
                    retrievals.append(torch.zeros_like(memory_inputs[position]))
            next_state = NeuralMemoryState(
                fast_weights=fast_weights,
                surprise=surprise,
                context_query_history=(
                    state.context_query_history
                    if context_query_history is None
                    else context_query_history
                ),
                output_query_history=output_history,
                write_history=write_history,
            )
            return MemoryStepOutput(
                retrieval=torch.stack(retrievals),
                state=next_state,
                mean_alpha=_gate_mean(gates.alpha),
                mean_eta=_gate_mean(gates.eta),
                mean_theta=_gate_mean(gates.theta),
            )


def _gate_mean(gates: Mapping[str, Tensor]) -> Tensor:
    flattened = [value.float().mean() for value in gates.values()]
    return torch.stack(flattened).mean()
