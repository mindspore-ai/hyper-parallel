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
"""One-card NPU acceptance cases for reentrant checkpoint exclusion."""

import contextlib
import gc
import importlib
import statistics
import time
import weakref
from unittest.mock import patch

import pytest
import torch

from hyper_parallel.components.functional.grouped_matmul import grouped_matmul
from hyper_parallel.core.activation_memory import checkpoint, checkpoint_exclude_wrapper, checkpoint_wrapper

_RECIPES = importlib.import_module("hyper_parallel.core.activation_memory.checkpoint_exclude")


def _require_npu() -> None:
    """Skip only when this worker has no usable NPU device."""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU required")


class _Excluded(torch.nn.Module):
    """Count executions of a backend operator within the excluded region."""

    def __init__(self, operator: str) -> None:
        """Configure the excluded backend operator and its call counter."""
        super().__init__()
        self.operator = operator
        self.calls = 0

    def forward(self, activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Execute an NPU operator once per forward."""
        self.calls += 1
        if self.operator == "mm":
            return activation @ weight
        if self.operator == "gmm":
            groups = torch.tensor([8, 16, 24, 32], device=activation.device, dtype=torch.int64)
            return grouped_matmul(activation, weight, group_list=groups)
        # torch-npu is optional to plain hyper_parallel imports; load only in this worker.
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_fusion_attention(
            activation, weight, weight, head_num=16, input_layout="BNSD", scale=0.125,
        )[0]


class _Block(torch.nn.Module):
    """Recompute before/after the excluded operation, with a trainable weight."""

    def __init__(self, operator: str, pointwise_layers: int = 0) -> None:
        """Build a BF16 operator block with optional recomputed pointwise work."""
        super().__init__()
        self.operator = operator
        self.pointwise_layers = pointwise_layers
        shape = (4, 64, 32) if operator == "gmm" else (64, 64) if operator == "mm" else (1, 16, 32, 64)
        self.weight = torch.nn.Parameter(torch.randn(shape, device="npu", dtype=torch.bfloat16) * 0.02)
        self.excluded = checkpoint_exclude_wrapper(_Excluded(operator))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Produce a scalar with a gradient through the excluded operator."""
        activation = tensor * 1.125 + 0.03125
        for _ in range(self.pointwise_layers):
            activation = torch.nn.functional.silu(activation + 0.015625)
        output = self.excluded(activation, self.weight)
        for _ in range(self.pointwise_layers):
            output = torch.nn.functional.silu(output + 0.015625)
        return output.float().square().sum() * 0.125


def _input(operator: str) -> torch.Tensor:
    """Create a deterministic BF16 input with the operator's expected layout."""
    shape = (32, 64) if operator in ("mm", "gmm") else (1, 16, 32, 64)
    return (torch.randn(shape, device="npu", dtype=torch.bfloat16) * 0.1).requires_grad_()


def _run_operator(operator: str, mode: str, input_data: torch.Tensor,
                  weight_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return detached loss and both gradients from an isolated model."""
    block = _Block(operator)
    with torch.no_grad():
        block.weight.copy_(weight_data)
    tensor = input_data.detach().clone().requires_grad_()
    if mode == "eager":
        loss = block(tensor)
    elif mode == "wrapper":
        loss = checkpoint_wrapper(block, use_reentrant=True)(tensor)
    else:
        loss = checkpoint(block, tensor, use_reentrant=mode == "reentrant")
    loss.backward()
    return loss.detach(), tensor.grad.detach(), block.weight.grad.detach(), block.excluded.calls


def _check_numerics(actual: torch.Tensor, reference: torch.Tensor, label: str) -> None:
    """Bound global relative error and largest-element error, even for tiny BF16 gradients."""
    expected = reference.float()
    observed = actual.float()
    magnitude = torch.linalg.vector_norm(expected).item()
    assert magnitude > 1e-8, f"{label}: reference gradient/loss is effectively zero"
    relative_error = (torch.linalg.vector_norm(observed - expected) / magnitude).item()
    # BF16 has 0.78% spacing near 1; FA/GMM backward can accumulate in a different order.
    assert relative_error < 0.08, f"{label}: relative L2 error {relative_error:.5f}"
    max_error = (observed - expected).abs().max().item()
    max_reference = expected.abs().max().item()
    assert max_error <= 0.04 * max_reference + 1e-6, f"{label}: max error {max_error:.6g}"


def test_numerics_rejects_zero_gradient() -> None:
    """A BF16-sized nonzero reference must reject a silently missing gradient."""
    reference = torch.full((32,), 2e-6)
    _check_numerics(reference.clone(), reference, "matching reference")
    with pytest.raises(AssertionError, match="relative L2 error"):
        _check_numerics(torch.zeros_like(reference), reference, "missing gradient")


def test_operator_numerics() -> None:
    """Compare loss/dx/dw and require exactly one excluded operator execution."""
    _require_npu()
    torch.manual_seed(741)
    for operator in ("mm", "gmm", "fa"):
        example = _Block(operator)
        data = _input(operator)
        weight = example.weight.detach().clone()
        reference = _run_operator(operator, "eager", data, weight)
        nonreentrant = _run_operator(operator, "nonreentrant", data, weight)
        for mode in ("nonreentrant", "reentrant", "wrapper"):
            actual = nonreentrant if mode == "nonreentrant" else _run_operator(operator, mode, data, weight)
            assert actual[3] == 1, f"{operator}/{mode} ran excluded operator {actual[3]} times"
            for name, observed, expected in zip(("loss", "dx", "dw"), actual[:3], reference[:3]):
                _check_numerics(observed, expected, f"{operator}/{mode}/{name} vs eager")
            for name, observed, expected in zip(("loss", "dx", "dw"), actual[:3], nonreentrant[:3]):
                _check_numerics(observed, expected, f"{operator}/{mode}/{name} vs nonreentrant")


def test_rng_autocast() -> None:
    """A skipped random region must advance RNG; replay must restore autocast."""
    _require_npu()

    def run(mode: str) -> tuple:
        """Run one isolated seeded autocast forward and backward."""
        torch.manual_seed(921)
        torch.npu.manual_seed(921)
        tensor = torch.randn(64, 64, device="npu", requires_grad=True)
        weight = torch.randn(64, 64, device="npu", requires_grad=True)
        calls = []

        def excluded(value: torch.Tensor) -> torch.Tensor:
            """Apply dropout inside the excluded matrix multiplication."""
            calls.append(None)
            return torch.nn.functional.dropout(value @ weight, p=0.25, training=True)

        wrapped = checkpoint_exclude_wrapper(excluded)

        def block(value: torch.Tensor) -> torch.Tensor:
            """Consume RNG before and after the excluded operation."""
            hidden = torch.nn.functional.dropout(value * 0.5, p=0.2, training=True)
            return (wrapped(hidden) * torch.rand_like(hidden)).float().square().mean()

        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            loss = block(tensor) if mode == "eager" else checkpoint(block, tensor, use_reentrant=mode == "reentrant")
        loss.backward()
        return loss.detach(), tensor.grad.detach(), weight.grad.detach(), torch.npu.get_rng_state(), len(calls)

    expected = run("eager")
    for mode in ("nonreentrant", "reentrant"):
        actual = run(mode)
        assert actual[4] == 1
        assert torch.equal(actual[3], expected[3]), f"{mode}: NPU RNG advanced differently"
        for name, observed, reference in zip(("loss", "dx", "dw"), actual[:3], expected[:3]):
            _check_numerics(observed, reference, f"autocast/{mode}/{name}")


class _MemoryBlock(torch.nn.Module):
    """Expose weak storage refs for a large recomputable input to an excluded MM."""

    def __init__(self) -> None:
        """Create the excluded matrix multiplication and storage probes."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1024, 64, device="npu", dtype=torch.bfloat16) * 0.01)
        self.excluded = checkpoint_exclude_wrapper(_Excluded("mm"))
        self.storage_refs = []

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Record a storage weakref without retaining the recomputable tensor."""
        activation = tensor.repeat(32, 1)
        if not torch.is_grad_enabled():
            self.storage_refs.append(weakref.ref(activation.untyped_storage()))
        return self.excluded(activation, self.weight).float().square().mean()


def _no_input_recipes(args: object, kwargs: object) -> tuple:
    """Keep excluded saved inputs as real tensors, preserving the same operator graph."""
    del args, kwargs
    return ()


def _measure_memory(recipe_disabled: bool) -> tuple[int, int, bool, list[int]]:
    """Measure same-body checkpoint steps with or without input reconstruction recipes."""
    torch.manual_seed(188)
    torch.npu.manual_seed(188)
    block = _MemoryBlock()
    model = checkpoint_wrapper(block, use_reentrant=True)
    source = torch.randn(256, 1024, device="npu", dtype=torch.bfloat16)
    for _ in range(2):
        model(source.detach().clone().requires_grad_()).backward()
        block.weight.grad = None
        block.storage_refs.clear()
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    observations = []
    # This isolates saved-input recipe savings, not the detached-leaf AccumulateGrad design.
    context = (patch.object(_RECIPES, "_capture_recompute_inputs", _no_input_recipes)
               if recipe_disabled else contextlib.nullcontext())
    with context:
        for _ in range(4):
            baseline = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            tensor = source.detach().clone().requires_grad_()
            loss = model(tensor)
            torch.npu.synchronize()
            gc.collect()
            storage_ref = block.storage_refs[-1]
            retained = storage_ref() is not None
            assert retained == recipe_disabled, "excluded input storage lifetime changed unexpectedly"
            forward = torch.npu.memory_allocated() - baseline
            loss.backward()
            torch.npu.synchronize()
            peak = torch.npu.max_memory_allocated() - baseline
            assert block.excluded.calls == 3 + len(observations), "excluded MM ran during replay"
            del loss, tensor
            block.weight.grad = None
            gc.collect()
            torch.npu.synchronize()
            assert storage_ref() is None, "completed step kept excluded input storage alive"
            observations.append((forward, peak, retained, torch.npu.memory_allocated()))
            block.storage_refs.clear()
    return (int(statistics.median(row[0] for row in observations)),
            int(statistics.median(row[1] for row in observations)),
            any(row[2] for row in observations), [row[3] for row in observations])


def test_memory_lifecycle() -> None:
    """Compare recipe-enabled exclusion against same-body recipe-disabled exclusion."""
    _require_npu()
    optimized = [_measure_memory(False) for _ in range(3)]
    no_recipe = [_measure_memory(True) for _ in range(3)]
    assert all(not result[2] for result in optimized), "optimized exclude retained recomputable input storage"
    assert all(result[2] for result in no_recipe), "recipe-disabled exclude did not retain saved input storage"
    # The 8192x1024 BF16 activation is 16 MiB. Replay allocates its own activation,
    # so the full-step peak need not fall by the same amount as forward residency.
    activation_bytes = 8192 * 1024 * 2
    min_gap = activation_bytes // 2
    for field, label in ((0, "forward-end allocated"), (1, "full-step peak")):
        enabled = statistics.median(result[field] for result in optimized)
        disabled = statistics.median(result[field] for result in no_recipe)
        gap = disabled - enabled
        print(f"{label}: recipe_enabled={enabled:.0f} recipe_disabled={disabled:.0f} gap={gap:.0f} bytes")
        if field == 0:
            assert abs(gap - activation_bytes) < min_gap, f"{label} gap {gap} differs from input bytes"
        else:
            assert gap > 0, f"{label} did not decrease (gap {gap} bytes)"
    for result in optimized + no_recipe:
        assert max(result[3]) - min(result[3]) < min_gap, "allocated bytes grew across completed steps"


def test_timing_observation() -> None:
    """Report forward/step host submissions and synchronized steps, without speed assertions."""
    _require_npu()
    torch.manual_seed(189)
    for operator in ("mm", "gmm", "fa"):
        reference = _Block(operator, pointwise_layers=24)
        weight = reference.weight.detach().clone()
        input_data = _input(operator).detach()
        for mode in ("nonreentrant", "reentrant"):
            block = _Block(operator, pointwise_layers=24)
            with torch.no_grad():
                block.weight.copy_(weight)
            samples = []
            for step in range(7):
                value = input_data.clone().requires_grad_()
                block.weight.grad = None
                torch.npu.synchronize()
                start = time.perf_counter()
                loss = checkpoint(block, value, use_reentrant=mode == "reentrant")
                forward_host = time.perf_counter()
                loss.backward()
                submitted = time.perf_counter()
                torch.npu.synchronize()
                completed = time.perf_counter()
                if step >= 2:
                    samples.append(((forward_host - start) * 1000, (submitted - start) * 1000,
                                    (completed - start) * 1000))
            medians = [statistics.median(sample[index] for sample in samples) for index in range(3)]
            print(f"{operator}/{mode}: forward_host_ms={medians[0]:.3f} "
                  f"full_host_ms={medians[1]:.3f} sync_step_ms={medians[2]:.3f}")
