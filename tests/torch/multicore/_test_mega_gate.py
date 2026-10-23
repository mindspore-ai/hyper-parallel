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
"""Single-card precision worker for the formal MegaGate module entry."""

from __future__ import annotations

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# pylint: disable=wrong-import-position
import torch
import torch_npu  # noqa: F401  pylint: disable=unused-import
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler
from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
from hyper_parallel.core.multicore.modules.mega_gate.diagnostics import (
    MEGA_GATE_ROUTE_GRAD_STAGE_NAMES,
)
from hyper_parallel.core.multicore.torch import mega_gate_ops

# pylint: enable=wrong-import-position

_DEVICE = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
_HIDDEN = 5120
_EXPERTS = 384
_TOP_K = 6
_SCALING = 1.5


def _training_gate(hidden_size: int, experts: int, top_k: int, dtype: torch.dtype) -> MegaGate:
    """Build a small native vision router with FP32 correction biases."""
    gate = MegaGate(
        hidden_size=hidden_size, num_experts=experts, top_k=top_k,
        scoring_func="sqrtsoftplus", routed_scaling_factor=_SCALING, vision_enabled=True,
    ).to(_DEVICE)
    weight = torch.randn((experts, hidden_size), device=_DEVICE, dtype=dtype) * 0.02
    gate.weight = torch.nn.Parameter(weight)
    with torch.no_grad():
        gate.bias.uniform_(-0.1, 0.1)
        gate.bias_vl.uniform_(-0.1, 0.1)
    return gate


def _training_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Weight gradients by expert ID so unsorted TopK order cannot affect the loss."""
    logits, weights, indices = outputs
    coefficients = (indices.float() + 1) / logits.shape[-1]
    return (weights * coefficients).sum() + 0.01 * logits.square().mean()


def _gate_reference(gate: MegaGate, hidden: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    """Use the independent Torch golden with the same parameters and vision mask."""
    if mask is None or gate.bias_vl is None:
        return _golden_config(hidden, gate.weight, gate.bias, gate.top_k, gate.routed_scaling_factor)
    return _golden_vision(hidden, gate.weight, gate.bias, gate.bias_vl, mask, gate.top_k, gate.routed_scaling_factor)


def _check_training_parity(
    gate: MegaGate, hidden: torch.Tensor, mask: torch.Tensor | None,
    *, autocast_enabled: bool = False, mode: str = "combined",
) -> None:
    """Compare independent, synchronized graphs including the FP32 Route boundary."""
    snapshots: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = []
    for native in (False, True):
        leaf = hidden.detach().clone().requires_grad_()
        with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=autocast_enabled):
            outputs = gate(leaf, image_mask=mask) if native else _gate_reference(gate, leaf, mask)
            logits, weights, indices = outputs
            loss = (weights * ((indices.float() + 1) / gate.num_experts)).sum()
            if mode == "combined":
                loss = loss + 0.01 * logits.square().mean()
        gradients = torch.autograd.grad(loss, (logits, leaf, gate.weight))
        torch.npu.synchronize()
        snapshots.append((
            tuple(tensor.detach().cpu() for tensor in outputs),
            tuple(tensor.detach().cpu() for tensor in gradients),
        ))
    assert len(snapshots) == 2, f"expected golden and native snapshots, got {len(snapshots)}"
    expected = snapshots[0]
    actual = snapshots[1]
    _assert_outputs_match(actual[0], expected[0])
    expected_logits_dtype = torch.bfloat16 if autocast_enabled else torch.float32
    expected_dtypes = (expected_logits_dtype, torch.float32, torch.int64)
    assert tuple(t.dtype for t in actual[0]) == expected_dtypes, (
        f"output dtypes={tuple(t.dtype for t in actual[0])}, expected {expected_dtypes}"
    )
    for index, (gradient, reference) in enumerate(zip(actual[1], expected[1])):
        rtol, atol = (1e-2, 1e-4) if gradient.dtype == torch.bfloat16 else (1e-5, 1e-6)
        torch.testing.assert_close(gradient, reference, rtol=rtol, atol=atol, msg=f"gradient index={index}")
    assert gate.bias.grad is None and (gate.bias_vl is None or gate.bias_vl.grad is None), (
        f"ranking bias gradients must be None: text={gate.bias.grad}, vision={gate.bias_vl}"
    )


def _check_mega_gate_dynamic_vision_padding_parity() -> None:
    """Cover dynamic dimensions, mixed visual bias, a tail row, and padded experts."""
    hidden_size, experts, top_k = 96, 33, 7
    gate = _training_gate(hidden_size, experts, top_k, torch.float32)
    hidden = torch.zeros((1, 65, hidden_size), device=_DEVICE)
    mask = (torch.arange(65, device=_DEVICE) % 3 == 1).reshape(1, 65)
    with torch.no_grad():
        gate.bias.copy_(torch.linspace(2.0, 0.0, experts, device=_DEVICE))
        gate.bias_vl.copy_(torch.linspace(0.0, 2.0, experts, device=_DEVICE))
        expected = _gate_reference(gate, hidden, mask)
        torch.npu.synchronize()
        actual = gate(hidden, image_mask=mask)
        torch.npu.synchronize()
        indices = actual[2].cpu()
        assert bool(((indices >= 0) & (indices < experts)).all()), (
            f"TopK selected padding: experts={experts}, indices={indices.tolist()}"
        )
        _assert_outputs_match(actual, expected)


def _check_mega_gate_dynamic_training_graphs(dtype: torch.dtype, hidden_size: int, experts: int, top_k: int) -> None:
    """Keep multiple layers and shapes alive, then differentiate them in reverse order."""
    torch.manual_seed(2301)
    torch.npu.manual_seed(2301)
    gates = [_training_gate(hidden_size, experts, top_k, dtype) for _ in range(2)]
    pending = []
    # Includes T smaller than the AIV count, tail rows, and changing batch/sequence shapes.
    for index, leading in enumerate(((1, 17), (1, 65), (2, 33))):
        gate = gates[index % len(gates)]
        hidden = torch.randn((*leading, hidden_size), device=_DEVICE, dtype=dtype, requires_grad=True)
        mask = torch.rand(hidden.shape[:-1], device=_DEVICE) < (0.0, 1.0, 0.25)[index % 3]
        reference_hidden = hidden.detach().clone().requires_grad_()
        reference = _gate_reference(gate, reference_hidden, mask)
        reference_grad = torch.autograd.grad(_training_loss(reference), (reference_hidden, gate.weight))
        torch.npu.synchronize()
        expected = tuple(tensor.detach().cpu() for tensor in reference)
        expected_grad = tuple(tensor.detach().cpu() for tensor in reference_grad)
        del reference, reference_grad, reference_hidden
        actual = gate(hidden, image_mask=mask)
        torch.npu.synchronize()
        _assert_outputs_match(tuple(tensor.detach().cpu() for tensor in actual), expected)
        pending.append((gate, hidden, actual, expected, expected_grad))

    for gate, hidden, actual, expected, expected_grad in reversed(pending):
        gradients = torch.autograd.grad(_training_loss(actual), (hidden, gate.weight))
        torch.npu.synchronize()
        _assert_outputs_match(tuple(tensor.detach().cpu() for tensor in actual), expected)
        for gradient, reference in zip(gradients, expected_grad):
            rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-4)
            torch.testing.assert_close(gradient.cpu(), reference, rtol=rtol, atol=atol)


def _async_accumulation_step(
    gate: MegaGate, optimizer: torch.optim.Optimizer,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]], native: bool,
) -> torch.Tensor:
    """Enqueue two microbatches and one update; return device-resident finite checks."""
    optimizer.zero_grad(set_to_none=True)
    checks = []
    for hidden, mask in microbatches:
        leaf = hidden.detach().clone().requires_grad_()
        outputs = gate(leaf, image_mask=mask) if native else _gate_reference(gate, leaf, mask)
        loss = _training_loss(outputs) / len(microbatches)
        loss.backward()
        checks.extend((torch.isfinite(loss.detach()), torch.isfinite(leaf.grad).all()))
    checks.append(torch.isfinite(gate.weight.grad).all())
    optimizer.step()
    checks.append(torch.isfinite(gate.weight).all())
    optimizer.zero_grad(set_to_none=True)
    return torch.stack(checks)


def _check_async_training(
    gate: MegaGate, microbatches: list[tuple[torch.Tensor, torch.Tensor]], native: bool,
) -> None:
    """Run five optimizer steps without host reads between steps."""
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3)
    _async_accumulation_step(gate, optimizer, microbatches, native)
    torch.npu.synchronize()
    initial_weight = gate.weight.detach().clone()
    checks = []
    for _ in range(5):
        checks.append(_async_accumulation_step(gate, optimizer, microbatches, native))
    torch.npu.synchronize()
    finite = torch.stack(checks).cpu()
    assert bool(finite.all()), f"nonfinite training values: native={native}, failed={torch.nonzero(~finite).tolist()}"
    assert not torch.equal(gate.weight, initial_weight), f"weight did not change after five steps: native={native}"
    assert gate.bias.grad is None and gate.bias_vl.grad is None, (
        f"ranking bias gradients: native={native}, text={gate.bias.grad}, vision={gate.bias_vl.grad}"
    )


def _check_mega_gate_async_training_steps() -> None:
    """Compare one step, then independently run golden/native AdamW accumulation windows."""
    torch.manual_seed(2313)
    gate = _training_gate(128, 33, 7, torch.bfloat16)
    reference_gate = _training_gate(128, 33, 7, torch.bfloat16)
    reference_gate.load_state_dict(gate.state_dict(), strict=True)
    microbatches = [
        (torch.randn((1, tokens, 128), device=_DEVICE, dtype=torch.bfloat16),
         torch.arange(tokens, device=_DEVICE).reshape(1, tokens) % 4 == 0)
        for tokens in (257, 513)
    ]
    _check_training_parity(gate, *microbatches[0])
    _check_async_training(reference_gate, microbatches, False)
    torch.npu.synchronize()
    _check_async_training(gate, microbatches, True)


def _check_mega_gate_checkpoint_accumulation() -> None:
    """Compare one non-reentrant checkpoint update with golden."""
    torch.manual_seed(2302)
    torch.npu.manual_seed(2302)
    hidden_size, experts, top_k = 128, 33, 7
    token_counts = (65, 129)
    gate = _training_gate(hidden_size, experts, top_k, torch.bfloat16)
    reference_gate = _training_gate(hidden_size, experts, top_k, torch.bfloat16)
    reference_gate.load_state_dict(gate.state_dict())
    optimizer = torch.optim.SGD(gate.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.SGD(reference_gate.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    for tokens in token_counts:
        hidden = torch.randn((1, tokens, hidden_size), device=_DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.rand((1, tokens), device=_DEVICE) < 0.25
        reference_hidden = hidden.detach().clone().requires_grad_()
        expected = _gate_reference(reference_gate, reference_hidden, mask)
        (_training_loss(expected) / 2).backward()
        torch.npu.synchronize()
        with patch.object(
            mega_gate_ops,
            "_mega_gate_route",
            wraps=mega_gate_ops._mega_gate_route,  # pylint: disable=protected-access
        ) as launch:
            actual = checkpoint(gate, hidden, mask, use_reentrant=False)
            torch.npu.synchronize()
            _assert_outputs_match(actual, expected)
            (_training_loss(actual) / 2).backward()
            torch.npu.synchronize()
            assert launch.call_count == 2, f"expected native forward plus recompute: calls={launch.call_count}"
        _assert_outputs_match(actual, expected)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(gate.weight.grad, reference_gate.weight.grad, rtol=1e-2, atol=1e-4)
    assert gate.bias.grad is None and gate.bias_vl.grad is None, (
        f"ranking biases must have no gradients: text={gate.bias.grad}, vision={gate.bias_vl.grad}"
    )
    optimizer.step()
    reference_optimizer.step()
    torch.npu.synchronize()
    torch.testing.assert_close(gate.weight, reference_gate.weight, rtol=1e-2, atol=1e-4)


def _make_gate(*, scoring: str = "sqrtsoftplus", vision_enabled: bool = False) -> MegaGate:
    """Create the formal V4.1 module configuration."""
    return MegaGate(
        hidden_size=_HIDDEN,
        num_experts=_EXPERTS,
        scoring_func=scoring,
        top_k=_TOP_K,
        routed_scaling_factor=_SCALING,
        vision_enabled=vision_enabled,
    )


def _set_parameters(gate: MegaGate, weight: torch.Tensor, bias: torch.Tensor) -> None:
    """Install deterministic leaves while preserving the module parameter names."""
    gate.weight = torch.nn.Parameter(weight, requires_grad=weight.requires_grad)
    gate.bias = torch.nn.Parameter(bias, requires_grad=bias.requires_grad)


def _golden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the V4.1 Projection and text Route reference."""
    return _golden_config(hidden, weight, bias, _TOP_K, _SCALING)


def _golden_config(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the reference for an arbitrary compatible Gate shape."""
    logits = functional.linear(  # pylint: disable=not-callable
        hidden.reshape(-1, hidden.shape[-1]).float(), weight.float()
    )
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    if top_k > 1:
        selected = selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, selected * scaling, indices


def _golden_vision(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    text_bias: torch.Tensor,
    vision_bias: torch.Tensor,
    image_mask: torch.Tensor,
    top_k: int = _TOP_K,
    scaling: float = _SCALING,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the V4.1 reference with per-token text or vision bias."""
    logits = functional.linear(  # pylint: disable=not-callable
        hidden.reshape(-1, hidden.shape[-1]).float(), weight.float()
    )
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    correction_bias = torch.where(
        image_mask.reshape(-1, 1),
        vision_bias.unsqueeze(0),
        text_bias.unsqueeze(0),
    )
    indices = torch.topk(scores + correction_bias, top_k, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    if top_k > 1:
        selected = selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, selected * scaling, indices


def _assert_outputs_match(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Compare expert sets and their weights independently of TopK order."""
    actual_logits, actual_weights, actual_indices = actual
    expected_logits, expected_weights, expected_indices = expected
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1.0e-5, atol=1.0e-6)
    actual_order = actual_indices.argsort(dim=-1)
    expected_order = expected_indices.argsort(dim=-1)
    sorted_actual_indices = actual_indices.gather(1, actual_order)
    sorted_expected_indices = expected_indices.gather(1, expected_order)
    sorted_actual_weights = actual_weights.gather(1, actual_order)
    sorted_expected_weights = expected_weights.gather(1, expected_order)
    torch.testing.assert_close(sorted_actual_indices, sorted_expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(sorted_actual_weights, sorted_expected_weights, rtol=1.0e-5, atol=1.0e-6)


def _make_inputs(batch: int, sequence: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic contiguous model inputs for one supported shape."""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    hidden = torch.randn((batch, sequence, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn((_EXPERTS, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
    bias = torch.empty((_EXPERTS,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
    return hidden, weight, bias


def _check_mega_gate_module_parity() -> None:
    """Check module flattening and numerical parity at the 4K training length."""
    gate = _make_gate()
    hidden, weight, bias = _make_inputs(2, 2048, seed=20)
    _set_parameters(gate, weight, bias)
    with torch.no_grad():
        expected = _golden(hidden, weight, bias)
        actual = gate(hidden)
    _assert_outputs_match(actual, expected)


def _check_mega_gate_empty_tokens() -> None:
    """Match visual Golden outputs and projection gradients without launching native Route."""
    hidden_size, expert_count, top_k = 64, 16, 6
    gate = _training_gate(hidden_size, expert_count, top_k, torch.bfloat16)
    hidden = torch.empty((1, 0, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
    image_mask = torch.empty((1, 0), device=_DEVICE, dtype=torch.bool)

    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_weight = gate.weight.detach().clone().requires_grad_()  # pylint: disable=not-callable
    assert gate.bias_vl is not None
    expected = _golden_vision(
        reference_hidden,
        reference_weight,
        gate.bias,
        gate.bias_vl,
        image_mask,
        top_k,
        _SCALING,
    )
    expected_gradients = torch.autograd.grad(
        expected[0].sum() + expected[1].sum(), (reference_hidden, reference_weight)
    )
    torch.npu.synchronize()

    actual_hidden = hidden.detach().clone().requires_grad_()
    actual = gate(actual_hidden, image_mask=image_mask)
    actual_gradients = torch.autograd.grad(actual[0].sum() + actual[1].sum(), (actual_hidden, gate.weight))
    torch.npu.synchronize()

    _assert_outputs_match(actual, expected)
    assert tuple(tuple(tensor.shape) for tensor in actual) == ((0, expert_count), (0, top_k), (0, top_k))
    assert tuple(tensor.dtype for tensor in actual) == (torch.float32, torch.float32, torch.int64)
    for gradient, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
    assert gate.bias.grad is None and (gate.bias_vl is None or gate.bias_vl.grad is None)


def _check_mega_gate_rejects_image_mask_without_vision_bias() -> None:
    """Reject an explicitly supplied visual mask instead of silently selecting text routing."""
    gate = _make_gate(vision_enabled=False)
    hidden, _, _ = _make_inputs(1, 128, seed=28)
    image_mask = torch.ones(hidden.shape[:-1], device=_DEVICE, dtype=torch.bool)
    try:
        gate(hidden, image_mask=image_mask)
    except ValueError as error:
        if "vision_enabled=True" not in str(error):
            raise
    else:
        raise AssertionError("MegaGate must reject image_mask without a visual correction bias")


def _check_mega_gate_native_mask_layout_error() -> None:
    """Report an unsupported native mask layout at the Python boundary."""
    gate = _training_gate(128, 33, 7, torch.bfloat16)
    hidden = torch.randn((1, 257, 128), device=_DEVICE, dtype=torch.bfloat16)
    image_mask = torch.zeros((1, 514), device=_DEVICE, dtype=torch.bool)[:, ::2]
    assert not image_mask.is_contiguous()

    try:
        gate(hidden, image_mask=image_mask)
    except ValueError as error:
        assert str(error) == "image_mask must be contiguous for native MegaGate"
    else:
        raise AssertionError("native MegaGate must reject a non-contiguous image_mask")


def _check_mega_gate_rejects_second_derivative() -> None:
    """Fail deterministically when differentiating through native RouteGrad."""
    gate = _training_gate(128, 33, 7, torch.bfloat16)
    hidden = torch.randn((1, 257, 128), device=_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    routing_weights = gate(hidden)[1]
    first_gradient = torch.autograd.grad(routing_weights.sum(), hidden, create_graph=True)[0]
    try:
        torch.autograd.grad(first_gradient.float().sum(), hidden)
    except RuntimeError:
        return
    raise AssertionError("native MegaGate RouteGrad must reject second-order differentiation")


def _check_mega_gate_pipeline_profile() -> None:
    """Require the 49-token tail case to expose ten stages on 25 active AIVs."""
    tokens = 49
    gate = _make_gate()
    hidden, weight, bias = _make_inputs(1, tokens, seed=36 + tokens)
    _set_parameters(gate, weight, bias)
    with torch.no_grad():
        for _ in range(5):
            gate(hidden)
        torch.npu.synchronize()
        with multicore_profiler.mega_kernel_profile(detailed_task_names=True) as profiler:
            gate(hidden)
            profiler.step()
    with tempfile.TemporaryDirectory() as output_dir:
        trace = profiler.export_chrome_trace(Path(output_dir) / "mega_gate_trace.json")
    metadata = trace["megaKernelCycleTrace"]
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    assert metadata["invocationCount"] == 1
    worker_ids = sorted({event["args"]["block_id"] for event in events})
    assert len(worker_ids) == 25
    assert worker_ids == list(range(len(worker_ids)))
    assert metadata["recordCount"] == len(worker_ids) * 10
    assert metadata["droppedRecordCount"] == 0
    assert metadata["warnings"] == []
    assert len(events) == len(worker_ids) * 10
    assert len({event["tid"] for event in events}) == len(worker_ids)
    for worker in worker_ids:
        worker_events = [event for event in events if event["args"]["block_id"] == worker]
        assert [event["args"]["task_id"] for event in worker_events] == list(range(10))


def _check_mega_gate_grad_pipeline_profile() -> None:
    """Require RouteGrad to expose its ordered token-row stages."""
    gate = _make_gate()
    hidden, weight, bias = _gradient_inputs(*_make_inputs(1, 2048, seed=37))
    _set_parameters(gate, weight, bias)
    logits, routing_weights, _ = gate(hidden)
    logits_gradient = torch.randn_like(logits)
    routing_gradient = torch.randn_like(routing_weights)
    torch.npu.synchronize()
    with multicore_profiler.mega_kernel_profile(detailed_task_names=True) as profiler:
        torch.autograd.grad(
            (logits, routing_weights),
            (hidden, gate.weight),
            grad_outputs=(logits_gradient, routing_gradient),
        )
        profiler.step()
    with tempfile.TemporaryDirectory() as output_dir:
        trace = profiler.export_chrome_trace(Path(output_dir) / "mega_gate_grad_trace.json")
    metadata = trace["megaKernelCycleTrace"]
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    worker_ids = sorted({event["args"]["block_id"] for event in events})
    stage_count = len(MEGA_GATE_ROUTE_GRAD_STAGE_NAMES)
    assert metadata["invocationCount"] == 1
    assert 0 < len(worker_ids) <= 48
    assert metadata["recordCount"] == len(worker_ids) * stage_count
    assert metadata["droppedRecordCount"] == 0
    assert metadata["warnings"] == []
    for worker in worker_ids:
        worker_events = [event for event in events if event["args"]["block_id"] == worker]
        assert [event["args"]["task_id"] for event in worker_events] == list(range(stage_count))


def _check_mega_gate_symbol_resolution_concurrency() -> None:
    """Initialize Route and RouteGrad symbols safely from concurrent threads."""
    worker_count = 4
    barrier = threading.Barrier(worker_count)
    cases = []
    for _ in range(worker_count):
        gate = _training_gate(64, 16, 6, torch.bfloat16)
        hidden = torch.randn((1, 65, 64), device=_DEVICE, dtype=torch.bfloat16)
        cases.append((gate, hidden))
    # Keep adapter loading and plan creation outside the race under test. The
    # worker threads should contend only on the ACLNN symbol initialization.
    mega_gate_ops._load_native()  # pylint: disable=protected-access
    cases[0][0]._plan(cases[0][1])  # pylint: disable=protected-access

    def run_case(index: int) -> tuple[torch.Tensor, ...]:
        """Launch one independent forward/backward graph after all threads are ready.

        Args:
            index: Independent graph index assigned to this worker thread.
        """
        torch.npu.set_device(_DEVICE)
        gate, hidden = cases[index]
        leaf = hidden.detach().clone().requires_grad_()
        barrier.wait()
        logits, weights, indices = gate(leaf)
        gradients = torch.autograd.grad(logits.sum() + weights.sum(), (leaf, gate.weight))
        torch.npu.synchronize()
        return tuple(tensor.detach().cpu() for tensor in (logits, weights, indices, *gradients))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(run_case, range(worker_count)))
    for result in results:
        assert all(bool(torch.isfinite(tensor).all()) for tensor in (*result[:2], *result[3:])), result


def _gradient_inputs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached leaves so golden and MegaGate build independent graphs."""
    return (
        hidden.detach().clone().requires_grad_(),
        weight.detach().clone().requires_grad_(),
        bias.detach().clone().requires_grad_(),
    )


def _check_mega_gate_backward_parity() -> None:
    """Compare synchronized combined gradients for the 4096-token training shape."""
    gate = _make_gate()
    hidden, weight, bias = _make_inputs(2, 2048, seed=62)
    # Match model initialization scale so the main parity matrix compares
    # finite projection gradients. Dedicated extreme tests cover FP32
    # Softplus underflow and its NaN/Inf contract.
    weight.mul_(0.02)
    golden_inputs = _gradient_inputs(hidden, weight, bias)
    actual_inputs = _gradient_inputs(hidden, weight, bias)
    _set_parameters(gate, actual_inputs[1], actual_inputs[2])
    golden_outputs = _golden(*golden_inputs)
    torch.npu.synchronize()
    actual_outputs = gate(actual_inputs[0])
    torch.npu.synchronize()
    _assert_outputs_match(actual_outputs, golden_outputs)
    sorted_indices = actual_outputs[2].sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all().item())

    logits_gradient = torch.randn_like(golden_outputs[0])
    routing_gradient = torch.randn_like(golden_outputs[1])
    expected = torch.autograd.grad(
        golden_outputs[:2], golden_inputs[:2], grad_outputs=(logits_gradient, routing_gradient),
    )
    torch.npu.synchronize()
    actual = torch.autograd.grad(
        actual_outputs[:2], (actual_inputs[0], gate.weight), grad_outputs=(logits_gradient, routing_gradient),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=1.0e-2, atol=1.0e-4)
    torch.testing.assert_close(actual[1].float(), expected[1].float(), rtol=1.0e-2, atol=1.0e-4)
    assert gate.bias.grad is None


def _check_mega_gate_vision_weights_only_fresh_graph_regression() -> None:
    """Check the former 8K failure with independently synchronized backends."""
    torch.manual_seed(18)
    torch.npu.manual_seed(18)
    tokens = 8192
    hidden = torch.randn((1, tokens, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn((_EXPERTS, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
    weight.mul_(0.02)
    text_bias = torch.empty((_EXPERTS,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
    vision_bias = torch.empty((_EXPERTS,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
    image_mask = (torch.arange(tokens, device=_DEVICE) < round(tokens * 0.25)).reshape(1, tokens)
    gate = _make_gate(vision_enabled=True)
    routing_gradient = torch.randn((tokens, _TOP_K), device=_DEVICE, dtype=torch.float32)

    golden_hidden = hidden.detach().clone().requires_grad_()
    golden_weight = weight.detach().clone().requires_grad_()
    golden_outputs = _golden_vision(
        golden_hidden,
        golden_weight,
        text_bias,
        vision_bias,
        image_mask,
    )
    expected_gradients = torch.autograd.grad(
        (golden_outputs[1],),
        (golden_hidden, golden_weight),
        grad_outputs=(routing_gradient,),
    )
    torch.npu.synchronize()

    actual_hidden = hidden.detach().clone().requires_grad_()
    actual_weight = weight.detach().clone().requires_grad_()
    _set_parameters(gate, actual_weight, text_bias)
    gate.bias_vl = torch.nn.Parameter(vision_bias, requires_grad=False)
    actual_outputs = gate(actual_hidden, image_mask=image_mask)
    actual_gradients = torch.autograd.grad(
        (actual_outputs[1],),
        (actual_hidden, gate.weight),
        grad_outputs=(routing_gradient,),
    )
    torch.npu.synchronize()

    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=1.0e-2,
            atol=1.0e-4,
        )


def _check_mega_gate_dynamic_backward_parity() -> None:
    """Cover K=1, a tail row, dynamic dimensions, and retained backward state."""
    tokens, hidden_size, expert_count, top_k = 65, 64, 16, 1
    torch.manual_seed(72)
    torch.npu.manual_seed(72)
    hidden = torch.randn((1, tokens, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn((expert_count, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
    bias = torch.empty((expert_count,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
    gate = MegaGate(
        hidden_size=hidden_size,
        num_experts=expert_count,
        scoring_func="sqrtsoftplus",
        top_k=top_k,
        routed_scaling_factor=_SCALING,
    )
    golden_inputs = _gradient_inputs(hidden, weight, bias)
    actual_inputs = _gradient_inputs(hidden, weight, bias)
    _set_parameters(gate, actual_inputs[1], actual_inputs[2])
    actual_inputs = (actual_inputs[0], gate.weight, gate.bias)
    golden_outputs = _golden_config(*golden_inputs, top_k, _SCALING)
    actual_outputs = gate(actual_inputs[0])
    logits_gradient = torch.randn_like(golden_outputs[0])
    routing_gradient = torch.randn_like(golden_outputs[1])
    expected = torch.autograd.grad(
        golden_outputs[:2],
        golden_inputs[:2],
        grad_outputs=(logits_gradient, routing_gradient),
        retain_graph=True,
    )
    torch.npu.synchronize()
    for _ in range(2):
        actual = torch.autograd.grad(
            actual_outputs[:2],
            actual_inputs[:2],
            grad_outputs=(logits_gradient, routing_gradient),
            retain_graph=True,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=1.0e-2, atol=1.0e-4)
        torch.testing.assert_close(actual[1].float(), expected[1].float(), rtol=1.0e-2, atol=1.0e-4)

    for need_hidden, need_weight in ((True, False), (False, True)):
        branch_hidden = hidden.detach().clone().requires_grad_(need_hidden)
        branch_weight = weight.detach().clone().requires_grad_(need_weight)
        golden_hidden = hidden.detach().clone().requires_grad_(need_hidden)
        golden_weight = weight.detach().clone().requires_grad_(need_weight)
        _set_parameters(gate, branch_weight, bias)
        actual_branch_outputs = gate(branch_hidden)
        golden_branch_outputs = _golden_config(golden_hidden, golden_weight, bias, top_k, _SCALING)
        actual_inputs = tuple(
            tensor for tensor, needed in ((branch_hidden, need_hidden), (gate.weight, need_weight)) if needed
        )
        golden_branch_inputs = tuple(
            tensor for tensor, needed in ((golden_hidden, need_hidden), (golden_weight, need_weight)) if needed
        )
        actual_branch_grad = torch.autograd.grad(
            actual_branch_outputs[:2],
            actual_inputs,
            grad_outputs=(logits_gradient, routing_gradient),
        )
        torch.npu.synchronize()
        golden_branch_grad = torch.autograd.grad(
            golden_branch_outputs[:2],
            golden_branch_inputs,
            grad_outputs=(logits_gradient, routing_gradient),
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            actual_branch_grad[0].float(), golden_branch_grad[0].float(), rtol=1.0e-2, atol=1.0e-4
        )


def _check_mega_gate_route_score_operation_order(tokens: int, experts: int) -> None:
    """Check score rounding directly, including Softplus threshold and tiny-exponential branches."""
    torch.manual_seed(2302)
    logits = torch.randn((tokens, experts), device=_DEVICE, dtype=torch.float32) * 3
    logits[0, :8] = torch.tensor(
        [-80.0, -20.0, -1e-7, 0.0, 20.0 - 2 ** -19, 20.0, 20.0 + 2 ** -19, 80.0],
        device=_DEVICE, dtype=torch.float32,
    )
    expected_scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    torch.npu.synchronize()
    gate = _make_gate()
    plan = gate._plan(logits)  # pylint: disable=protected-access
    correction_bias = torch.zeros(experts, device=_DEVICE, dtype=torch.float32)
    outputs = mega_gate_ops._mega_gate_route(  # pylint: disable=protected-access
        logits,
        correction_bias,
        correction_bias,
        plan.vision_mask_placeholder,
        plan.route_runtime.runtime_config,
        plan.route_runtime.runtime_config,
        top_k=6,
        routed_scaling_factor=_SCALING,
        use_vision_bias=False,
    )
    torch.npu.synchronize()
    # The regression targets the rounding order, which the normal forward
    # tolerance cannot detect before BF16 gradient accumulation amplifies it.
    torch.testing.assert_close(outputs[2], expected_scores, rtol=0, atol=0)


def _check_mega_gate_route_grad_operation_order(top_k: int) -> None:
    """Isolate backward rounding by supplying the exact Torch forward state to native."""
    torch.manual_seed(2302)
    logits = torch.randn((257, 33), device=_DEVICE, dtype=torch.float32, requires_grad=True)
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    indices = torch.topk(scores, top_k, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    denominator = selected.sum(dim=-1, keepdim=True) + 1e-20
    weights = selected / denominator if top_k > 1 else selected
    weights = weights * _SCALING
    upstream = ((indices.float() + 1) / logits.shape[1]) / 2
    expected = torch.autograd.grad(weights, logits, grad_outputs=upstream)[0]
    torch.npu.synchronize()

    gate = _make_gate()
    plan = gate._plan(logits)  # pylint: disable=protected-access
    runtime = plan.route_grad_k1_runtime if top_k == 1 else plan.route_grad_runtime
    actual = mega_gate_ops._mega_gate_route_grad(  # pylint: disable=protected-access
        logits.detach(), scores.detach(), selected.detach(), denominator.detach(),
        indices, upstream, runtime.runtime_config, runtime.runtime_config,
        top_k=top_k, routed_scaling_factor=_SCALING,
    )
    torch.npu.synchronize()
    # Shared state removes forward approximation differences. The ordinary
    # tolerance misses the reassociation error that can cross a BF16 midpoint.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _check_mega_gate_route_grad_batch_parity(tokens: int, top_k: int) -> None:
    """Check different elementwise/reduction batches without projection MatMuls."""
    torch.manual_seed(2309 + top_k)
    logits = (torch.randn((tokens, _EXPERTS), device=_DEVICE) * 3).requires_grad_()
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    indices = torch.topk(scores, top_k, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    denominator = selected.sum(dim=-1, keepdim=True) + 1.0e-20
    weights = selected / denominator if top_k > 1 else selected
    weights = weights * _SCALING
    gradient = torch.randn_like(weights)
    expected = torch.autograd.grad(weights, logits, grad_outputs=gradient)[0]
    # Complete the reference graph before the native call, including on the
    # async task queue used in training.
    torch.npu.synchronize()

    gate = _make_gate()
    plan = gate._plan(logits)  # pylint: disable=protected-access
    runtime = plan.route_grad_k1_runtime if top_k == 1 else plan.route_grad_runtime
    with torch.no_grad():
        actual = mega_gate_ops._mega_gate_route_grad(  # pylint: disable=protected-access
            logits.detach(), scores.detach(), selected.detach(), denominator.detach(),
            indices, gradient, runtime.runtime_config, runtime.runtime_config,
            top_k=top_k, routed_scaling_factor=_SCALING,
        )
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def _check_mega_gate_route_grad_extremes() -> None:
    """Check stable SoftplusGrad and Scatter semantics at threshold and underflow inputs."""
    top_k = 3
    logits_values = (
        (-120.0, -20.0, -1.0, 0.0, 19.999, 20.0, 20.001, 80.0),
        (-100.0, -2.0, -0.0, 0.5, 19.5, 20.0, 21.0, 60.0),
    )
    reference_logits = torch.tensor(logits_values, device=_DEVICE, dtype=torch.float32, requires_grad=True)
    route_scores = functional.softplus(reference_logits).sqrt()  # pylint: disable=not-callable
    expert_indices = torch.topk(route_scores, top_k, dim=-1, sorted=False).indices
    sorted_indices = expert_indices.sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all().item())
    selected_scores = route_scores.gather(1, expert_indices)
    denominator = selected_scores.sum(dim=-1, keepdim=True) + 1.0e-20
    routing_weights = selected_scores / denominator * _SCALING
    routing_gradient = torch.tensor(
        ((0.75, -1.25, 2.0), (-0.5, 1.5, -2.25)),
        device=_DEVICE,
        dtype=torch.float32,
    )
    expected_score_grad = torch.autograd.grad(
        routing_weights,
        route_scores,
        grad_outputs=routing_gradient,
        retain_graph=True,
    )[0]
    scaled_gradient = routing_gradient * _SCALING
    expected_selected_grad = scaled_gradient / denominator
    expected_selected_grad -= (selected_scores * scaled_gradient).sum(dim=-1, keepdim=True) / denominator.square()
    scatter_reference = torch.zeros_like(route_scores).scatter_add(1, expert_indices, expected_selected_grad)
    torch.testing.assert_close(scatter_reference, expected_score_grad, rtol=1.0e-5, atol=1.0e-6)
    expected = torch.autograd.grad(routing_weights, reference_logits, grad_outputs=routing_gradient)[0]

    gate = _make_gate()
    plan = gate._plan(reference_logits)  # pylint: disable=protected-access

    def _run_route_grad() -> torch.Tensor:
        with torch.no_grad():
            result = mega_gate_ops._mega_gate_route_grad(  # pylint: disable=protected-access
                reference_logits.detach(),
                route_scores.detach(),
                selected_scores.detach(),
                denominator.detach(),
                expert_indices.detach(),
                routing_gradient,
                plan.route_grad_runtime.runtime_config,
                plan.route_grad_runtime.runtime_config,
                top_k=top_k,
                routed_scaling_factor=_SCALING,
            )
        torch.npu.synchronize()
        return result

    actual = _run_route_grad()
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1.0e-5, atol=1.0e-6)


def test_mega_gate_basic_parity() -> None:
    """Check one representative native forward and backward path."""
    torch.manual_seed(2401)
    torch.npu.manual_seed(2401)
    gate = _training_gate(128, 33, 7, torch.bfloat16)
    hidden = torch.randn((1, 257, 128), device=_DEVICE, dtype=torch.bfloat16)
    image_mask = (torch.arange(257, device=_DEVICE) % 4 == 0).reshape(1, 257)
    _check_training_parity(gate, hidden, image_mask)
    _check_training_parity(gate, hidden, None, autocast_enabled=True, mode="weights")
    _check_training_parity(gate, hidden, image_mask, autocast_enabled=True)


def test_mega_gate_forward_parity() -> None:
    """Cover public forward behavior without repeating NPU process startup."""
    _check_mega_gate_module_parity()
    _check_mega_gate_empty_tokens()
    _check_mega_gate_rejects_image_mask_without_vision_bias()
    _check_mega_gate_native_mask_layout_error()
    _check_mega_gate_dynamic_vision_padding_parity()


def test_mega_gate_backward_parity() -> None:
    """Cover projection and Route gradients across the distinct training paths."""
    _check_mega_gate_backward_parity()
    _check_mega_gate_vision_weights_only_fresh_graph_regression()
    _check_mega_gate_dynamic_backward_parity()
    _check_mega_gate_rejects_second_derivative()


def test_mega_gate_route_math_parity() -> None:
    """Preserve exact operation order, extreme values, tails, and dynamic TopK."""
    _check_mega_gate_route_score_operation_order(257, 33)
    for top_k in (1, 6):
        _check_mega_gate_route_grad_operation_order(top_k)
    for tokens, top_k in ((8193, 1), (16384, 7)):
        _check_mega_gate_route_grad_batch_parity(tokens, top_k)
    _check_mega_gate_route_grad_extremes()


def test_mega_gate_training_graphs() -> None:
    """Exercise representative dynamic graphs and asynchronous optimizer steps."""
    _check_mega_gate_dynamic_training_graphs(torch.bfloat16, 96, 33, 7)
    _check_mega_gate_async_training_steps()
    _check_mega_gate_checkpoint_accumulation()


def test_mega_gate_pipeline_profile() -> None:
    """Validate forward and backward stage traces in one NPU worker."""
    _check_mega_gate_pipeline_profile()
    _check_mega_gate_grad_pipeline_profile()


def test_mega_gate_symbol_resolution_concurrency() -> None:
    """Resolve Route and RouteGrad symbols safely from concurrent threads."""
    _check_mega_gate_symbol_resolution_concurrency()
