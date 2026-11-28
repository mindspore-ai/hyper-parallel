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
"""Shared native-oracle and real-backend forward/VJP precision checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import Any

import torch

from hyper_parallel.components.ops import ops


@dataclass(frozen=True)
class PrecisionCase:
    """One public operator contract and an independently expressed oracle.

    Inputs are quantized once on CPU; each run receives fresh device leaves.
    ``precision_dtype`` describes computation precision, including FP32 outputs
    and coefficient gradients produced by a low-precision computation.
    """

    family: str
    variant: str
    name: str
    operation: Callable[..., Any]
    reference: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    precision_dtype: torch.dtype = torch.float32
    expert_parameters: tuple[int, ...] = ()
    # Explicit per-input exceptions for fused VJPs; native/oracle checks keep default limits.
    backend_gradient_tolerances: dict[str, Tolerance] = field(default_factory=dict)


@dataclass(frozen=True)
class Tolerance:
    """Pointwise and whole-tensor error limits, both of which must pass."""

    atol: float
    rtol: float
    relative_l2: float


_LIMITS = {
    torch.float64: (Tolerance(1e-10, 1e-8, 1e-9), Tolerance(1e-9, 1e-7, 1e-8)),
    torch.float32: (Tolerance(1e-5, 1e-4, 1e-5), Tolerance(1e-4, 1e-3, 1e-4)),
    torch.float16: (Tolerance(1e-3, 1e-3, 1e-3), Tolerance(3e-3, 3e-3, 3e-3)),
    torch.bfloat16: (Tolerance(1e-2, 1e-2, 5e-3), Tolerance(2e-2, 2e-2, 1e-2)),
}


def assert_precision(
    actual: torch.Tensor, expected: torch.Tensor, *, dtype: torch.dtype,
    backward: bool = False, label: str = "tensor", tolerance: Tolerance | None = None,
) -> None:
    """Check metadata, finite values, exact discrete values, allclose and relative L2.

    Zero reference tensors require exact zeros. Diagnostics include the worst
    coordinate, absolute/relative errors, relative L2 and cosine similarity.
    """
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError(
            f"{label}: actual={actual.shape}/{actual.dtype}/{actual.device}, "
            f"expected={expected.shape}/{expected.dtype}/{expected.device}"
        )
    if not actual.is_floating_point():
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"{label}: discrete values differ")
        return
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(
            f"{label}: actual_nonfinite={(~torch.isfinite(actual)).sum().item()}, "
            f"expected_nonfinite={(~torch.isfinite(expected)).sum().item()}; non-finite values"
        )
    if not actual.numel():
        return
    # Ascend has no FP64 arithmetic; diagnostics must retain FP64 on the host.
    value, target = actual.detach().cpu().double(), expected.detach().cpu().double()
    difference = value - target
    error = difference.abs()
    target_norm, value_norm = target.norm(), value.norm()
    if target_norm == 0:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"{label}: structural zero differs")
        return
    relative_l2 = (difference.norm() / target_norm).item()
    cosine = ((value * target).sum() / (value_norm * target_norm)).item() if value_norm else 0.0
    worst = int(error.flatten().argmax())
    diagnostic = (
        f"{label}: max_abs={error.max().item():.6g}, "
        f"max_rel={(error / target.abs().clamp_min(1e-12)).max().item():.6g}, "
        f"relative_l2={relative_l2:.6g}, cosine={cosine:.9g}, "
        f"worst_flat_index={worst}, actual={value.flatten()[worst].item():.6g}, "
        f"expected={target.flatten()[worst].item():.6g}"
    )
    logging.getLogger(__name__).info(diagnostic)
    tolerance = tolerance or _LIMITS[dtype][int(backward)]
    torch.testing.assert_close(actual, expected, atol=tolerance.atol, rtol=tolerance.rtol, msg=diagnostic)
    if relative_l2 > tolerance.relative_l2:
        raise AssertionError(f"{diagnostic}; limit={tolerance.relative_l2}")


def _clone(value: Any, device: str, leaves: dict[str, torch.Tensor], path: str) -> Any:
    """Create independent device leaves and record each differentiable input path."""
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device).clone(memory_format=torch.preserve_format)
        result.requires_grad_(value.requires_grad)
        leaves[path] = result
        return result
    if isinstance(value, tuple):
        return tuple(_clone(part, device, leaves, f"{path}.{index}") for index, part in enumerate(value))
    if isinstance(value, dict):
        return {key: _clone(part, device, leaves, f"{path}.{key}") for key, part in value.items()}
    return value


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return [part for item in value for part in _flatten(item)]
    return [value]


def _evaluate(case: PrecisionCase, operation: Callable[..., Any], device: str) -> tuple[list[Any], dict]:
    """Evaluate outputs and reproducible VJPs while checking input immutability."""
    leaves: dict[str, torch.Tensor] = {}
    args = _clone(case.args, device, leaves, "args")
    kwargs = _clone(case.kwargs, device, leaves, "kwargs")
    saved = {name: tensor.detach().clone() for name, tensor in leaves.items()}
    outputs = _flatten(operation(*args, **kwargs))
    generator = torch.Generator().manual_seed(9182)
    terms = [
        (output.float() * torch.randn(output.shape, generator=generator).to(device)).sum()
        for output in outputs if isinstance(output, torch.Tensor) and output.requires_grad
    ]
    differentiable = {name: tensor for name, tensor in leaves.items() if tensor.requires_grad}
    gradients = dict.fromkeys(differentiable)
    if terms and differentiable:
        values = torch.autograd.grad(sum(terms), tuple(differentiable.values()), allow_unused=True)
        gradients.update(zip(differentiable, values))
    for name, tensor in leaves.items():
        torch.testing.assert_close(tensor, saved[name], atol=0, rtol=0, msg=f"{case.name}: input {name} mutated")
    return outputs, gradients


def _compare_outputs(
    case: PrecisionCase, backend: str, actual: list[Any], expected: list[Any],
    compare: Callable[[torch.Tensor, torch.Tensor, str], None], failures: list[str],
) -> None:
    """Compare every returned value while retaining all output metadata failures."""
    if len(actual) != len(expected):
        raise AssertionError(f"{case.name}: output count={len(actual)}, expected={len(expected)}")
    for index, (value, target) in enumerate(zip(actual, expected)):
        label = f"{case.family}.{case.variant}/{case.name}/{backend}/output[{index}]"
        if isinstance(target, torch.Tensor):
            if value.requires_grad != target.requires_grad:
                failures.append(
                    f"{label}: requires_grad={value.requires_grad}, expected={target.requires_grad}"
                )
            compare(value, target, label)
        elif value != target:
            failures.append(f"{label}: actual={value}, expected={target}")


def check_case(case: PrecisionCase, *, backend: str = "native", device: str = "cpu") -> None:
    """Compare native with the oracle, or a selected backend with same-device native.

    Complete the first VJP before replacing process-wide configuration. Restore
    the caller's configuration even on failure and assert the actual selection.
    """
    failures = []

    def compare(
        value: torch.Tensor, target: torch.Tensor, label: str, backward: bool = False,
        tolerance: Tolerance | None = None,
    ) -> None:
        """Collect a tensor comparison failure without hiding later VJP diagnostics."""
        try:
            assert_precision(value, target, dtype=case.precision_dtype, backward=backward, label=label,
                             tolerance=tolerance)
        except AssertionError as error:
            # Report every VJP even when an earlier output fails its budget.
            failures.append(str(error))

    previous = ops.get_backends()
    try:
        ops.set_backends({case.family: "native"})
        reference = case.reference if backend == "native" else case.operation
        expected, expected_gradients = _evaluate(case, reference, device)
        ops.set_backends({case.family: backend})
        actual, gradients = _evaluate(case, case.operation, device)
        used = ops.used_backends()
        if (case.family, case.variant, backend) not in used:
            raise AssertionError(f"{case.name}: selected={used}, expected={(case.family, case.variant, backend)}")
        _compare_outputs(case, backend, actual, expected, compare, failures)
        for name, target in expected_gradients.items():
            value = gradients[name]
            label = f"{case.family}.{case.variant}/{case.name}/{backend}/gradient[{name}]"
            if (value is None) != (target is None):
                failures.append(f"{label}: actual={value}, expected={target}")
                continue
            if target is not None:
                tolerance = case.backend_gradient_tolerances.get(name) if backend != "native" else None
                compare(value, target, label, backward=True, tolerance=tolerance)
                if name in {f"args.{index}" for index in case.expert_parameters}:
                    for expert, (expert_value, expert_target) in enumerate(zip(value, target)):
                        compare(expert_value, expert_target, f"{label}/expert[{expert}]", backward=True)
        if failures:
            raise AssertionError("\n".join(failures))
    finally:
        ops.set_backends(previous)
