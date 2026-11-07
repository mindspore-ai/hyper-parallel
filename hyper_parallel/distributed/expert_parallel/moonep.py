# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Collective MoonEP execution inside the native experts' FSDP boundary."""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable
from torch.profiler import record_function

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.distributed.expert_parallel.moonep_runtime import MoonEPExecution


class _Exchange(NamedTuple):
    received: torch.Tensor
    work: Any
    sent: torch.Tensor

    def wait(self) -> torch.Tensor:
        """Order the consumer on the collective while keeping the send storage live."""
        self.work.wait()
        if self.received.device.type != "cpu":
            stream = torch.get_device_module(self.received.device).current_stream()
            self.received.record_stream(stream)
            self.sent.record_stream(stream)
        return self.received


def _exchange(rows, send, recv, group):
    output = rows.new_empty((sum(recv), rows.shape[1]))
    work = dist.all_to_all_single(output, rows, recv, send, group=group, async_op=True)
    return _Exchange(output, work, rows)


def _pack(gate, down, indices):
    gate_rows = gate.index_select(0, indices).flatten(1)
    down_rows = down.index_select(0, indices).flatten(1)
    return torch.cat((gate_rows, down_rows), dim=1)


def _unpack(rows, gate, down):
    gate_rows, down_rows = rows.split((gate.shape[1] * gate.shape[2], down.shape[1] * down.shape[2]), dim=1)
    return gate_rows.reshape(-1, *gate.shape[1:]), down_rows.reshape(-1, *down.shape[1:])


def _prefetch(gate, down, plan, group):
    if not plan.weight_exchange:
        return None
    packet = _exchange(_pack(gate, down, plan.send_experts), plan.weight_send, plan.weight_recv, group)
    if gate.device.type != "cpu":
        stream = torch.get_device_module(gate.device).current_stream()
        gate.record_stream(stream)
        down.record_stream(stream)
    return packet


def _expert_weights(gate, down, plan, incoming):
    if plan.home_experts.numel() == gate.shape[0]:
        home_gate, home_down = gate, down
    else:
        home_gate = gate.index_select(0, plan.home_experts)
        home_down = down.index_select(0, plan.home_experts)
    if incoming is None:
        return home_gate, home_down
    remote_gate, remote_down = _unpack(incoming.wait(), gate, down)
    return torch.cat((home_gate, remote_gate)), torch.cat((home_down, remote_down))


def _gmm(inputs, weights, boundaries, *, weight_grad=False):
    # torch_npu is optional for applications using the rest of Expert Parallel.
    import torch_npu  # pylint: disable=import-outside-toplevel

    return torch_npu.npu_grouped_matmul(
        [inputs], [weights], bias=[], group_list=boundaries, group_list_type=0,
        group_type=2 if weight_grad else 0, split_item=2 if weight_grad else 3,
    )[0]


def _replica_gradients(dgate, ddown, plan, group):
    if not plan.weight_exchange:
        return dgate, ddown, None
    home = plan.home_experts.numel()
    remote_gate, remote_down = dgate[home:].flatten(1), ddown[home:].flatten(1)
    rows = torch.cat((remote_gate, remote_down), dim=1)
    packet = _exchange(rows, plan.weight_recv, plan.weight_send, group)
    if dgate.device.type != "cpu":
        stream = torch.get_device_module(dgate.device).current_stream()
        dgate.record_stream(stream)
        ddown.record_stream(stream)
    return dgate[:home], ddown[:home], packet


def _owner_gradient(local, original, indices, returned, send_experts):
    # Accumulate the local and replica partials before handing the original leaf to FSDP.
    if returned is None or returned.shape[0] == 0:
        if local.shape == original.shape:
            return local.contiguous()
        return torch.zeros_like(original).index_copy_(0, indices, local)
    gradient = torch.zeros_like(original, dtype=torch.float32)
    gradient.index_copy_(0, indices, local.float())
    gradient.index_add_(0, send_experts, returned.float())
    return gradient.to(original.dtype)


class _MoonEPExperts(torch.autograd.Function):
    """Save only activations and re-materialize the weights on the same execution ranks."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, gate: torch.Tensor, down: torch.Tensor,
        execution: MoonEPExecution, read_weights: Callable, activation: Callable, input_major: bool,
    ) -> torch.Tensor:
        """Evaluate grouped experts while retaining activations for their adjoint.

        Args:
            ctx: Autograd context for this expert invocation.
            x: Expert-major input rows.
            gate: Owner gate/up parameter tensor.
            down: Owner down-projection parameter tensor.
            execution: Retained route and pending replica transfer.
            read_weights: Read parameters restored by FSDP in backward.
            activation: Model-specific packed gate activation.
            input_major: Whether weights use input/output matrix order.
        """
        ctx.execution = execution
        plan = execution.plan
        ctx.read_weights, ctx.activation = read_weights, activation
        ctx.input_major = input_major
        with record_function("moonep::forward_weight_ready"):
            gate, down = _expert_weights(gate, down, plan, execution.incoming)
        if not is_recomputing():
            execution.incoming = None
        with record_function("moonep::forward_experts"):
            if x.shape[0] == 0:
                ctx.save_for_backward(x)
                return x
            if not input_major:
                gate, down = gate.transpose(-1, -2), down.transpose(-1, -2)
            preact = _gmm(x, gate, plan.boundaries)
            activated = activation(preact)
            output = _gmm(activated, down, plan.boundaries)
            ctx.save_for_backward(x, preact, activated)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dy: torch.Tensor) -> tuple:
        """Return activation and owner parameter gradients after replica accumulation.

        Args:
            ctx: Context retaining the route and expert activations.
            dy: Expert-major output gradient.
        """
        saved = ctx.saved_tensors
        execution = ctx.execution
        plan = execution.plan
        home_gate, home_down = ctx.read_weights()
        with record_function("moonep::backward_weight_ready"):
            execution.prefetch()
            gate, down = _expert_weights(home_gate, home_down, plan, execution.incoming)
            execution.incoming = None
        gate_dx = gate.transpose(-1, -2) if ctx.input_major else gate
        down_dx = down.transpose(-1, -2) if ctx.input_major else down
        dy = dy.contiguous()
        with record_function("moonep::backward_experts"):
            if dy.shape[0] != 0:
                x, preact, activated = saved
                dact = _gmm(dy, down_dx, plan.boundaries)
                ddown = _gmm(activated.transpose(0, 1), dy, plan.boundaries, weight_grad=True)
                with torch.enable_grad():
                    gate_input = preact.detach().requires_grad_()
                    dpreact, = torch.autograd.grad(ctx.activation(gate_input), gate_input, dact)
                dgate = _gmm(x.transpose(0, 1), dpreact, plan.boundaries, weight_grad=True)
                if not ctx.input_major:
                    dgate, ddown = dgate.transpose(-1, -2), ddown.transpose(-1, -2)
            else:
                dgate, ddown = torch.zeros_like(gate), torch.zeros_like(down)
        with record_function("moonep::replica_gradients"), execution.transport.stream_context(gradients=True):
            dgate, ddown, packet = _replica_gradients(dgate, ddown, plan, execution.transport.gradient_group)
        with record_function("moonep::backward_dx"):
            dx = _gmm(dpreact, gate_dx, plan.boundaries) if dy.shape[0] else dy
        with record_function("moonep::owner_gradients"):
            remote_gate, remote_down = _unpack(packet.wait(), home_gate, home_down) if packet else (None, None)
            dgate = _owner_gradient(dgate, home_gate, plan.home_experts, remote_gate, plan.send_experts)
            ddown = _owner_gradient(ddown, home_down, plan.home_experts, remote_down, plan.send_experts)
        execution.complete = True
        execution.prepare_weights = None
        return dx, dgate, ddown, None, None, None, None


class _BackwardPrefetch(torch.autograd.Function):
    """Start W after the expert FSDP pre-backward hook and before dY dispatch/replay."""

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, execution: MoonEPExecution) -> torch.Tensor:
        """Attach replica preparation to the experts' output boundary.

        Args:
            ctx: Autograd context retaining the execution record.
            output: Combined expert outputs.
            execution: Route whose weights will be needed in backward.
        """
        ctx.execution = execution
        return output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Prepare replicas before propagating gradients into token communication.

        Args:
            ctx: Context holding this invocation's execution record.
            gradient: Gradient at the combined expert output.
        """
        execution = ctx.execution
        with record_function("moonep::backward_weight_prefetch"):
            execution.prefetch()
            execution.transport.prefetch_previous(execution)
        return gradient, None


def run_moonep_experts(
    hidden: torch.Tensor,
    execution: MoonEPExecution,
    scores: torch.Tensor,
    read_weights: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    activation: Callable[[torch.Tensor], torch.Tensor],
    input_major: bool,
) -> torch.Tensor:
    """Dispatch and combine inside the experts' parameter restoration boundary.

    Dedup returns token outputs; the ordinary path returns source selection
    outputs for the native unpermute kernel. Router scores retain their gradients.

    Args:
        hidden: Source token vectors.
        execution: Compiled route and asynchronous replica preparation state.
        scores: Per-token router selection weights.
        read_weights: Read the original owner gate/up and down parameters.
        activation: Model-specific packed gate activation.
        input_major: Whether parameters use input/output matrix order.
    """
    plan, transport, dedup = execution.plan, execution.transport, execution.dedup
    gate, down = read_weights()
    execution.prepare_weights = lambda: _prefetch(*read_weights(), plan, transport.weight_group)
    with record_function("moonep::forward_weight_prefetch"):
        execution.prefetch()
    with record_function("moonep::dispatch"):
        if dedup is None:
            payload = hidden.index_select(0, plan.source_order // scores.shape[-1])
            received = ep_all_to_all(payload, plan.token_send, plan.token_recv, transport.group)
        else:
            received = dedup.dispatch(hidden, transport.group)
        expert_major = received.index_select(0, plan.receive_order)
    output = _MoonEPExperts.apply(
        expert_major, gate, down, execution, read_weights, activation, input_major,
    )
    with record_function("moonep::combine"):
        restored = torch.empty_like(output)
        restored[plan.receive_order] = output
        if dedup is None:
            result = ep_all_to_all(restored, plan.token_recv, plan.token_send, transport.group)
        else:
            selected = scores.reshape(-1, 1).index_select(0, plan.source_order)
            received_scores = ep_all_to_all(selected, plan.token_send, plan.token_recv, transport.group)
            weighted = (restored * received_scores).to(restored.dtype)
            result = dedup.combine(weighted, hidden.shape[0], transport.group)
    return _BackwardPrefetch.apply(result, execution)
