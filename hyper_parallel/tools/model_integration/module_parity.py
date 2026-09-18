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
"""Generic final-module forward/intermediate/backward parity runner."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import fnmatch
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.utils._pytree import tree_flatten, tree_map

from hyper_parallel.tools.model_integration.evidence_store import EvidenceStore
from hyper_parallel.tools.model_integration.schemas import (
    ComparisonMetric,
    tolerance_value,
)
from hyper_parallel.models.validation_spec import ModelValidationSpec, ModuleParityCase


@dataclass(frozen=True)
class ParityContext:
    """Resolved execution context supplied to model-owned parity builders."""

    manifest: Any
    dtype: str
    device: str
    evidence_store: EvidenceStore
    options: dict[str, Any]


@dataclass
class ParityExecution:
    """Values exposed to observation getters after one forward/backward."""

    module: Any
    output: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _select_module(value: Any, selector: str) -> Any:
    if selector in ("", "<root>"):
        return value
    module_by_fqn = dict(value.named_modules())
    selected = module_by_fqn.get(selector)
    if selected is None:
        raise ValueError(f"candidate selector matched no final module: {selector!r}")
    return selected


def _clone_value(value: Any) -> Any:
    if not torch.is_tensor(value):
        return value
    clone = value.detach().clone()
    if value.requires_grad and (clone.is_floating_point() or clone.is_complex()):
        clone.requires_grad_(True)
    return clone


def _clone_call_inputs(inputs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(inputs, Mapping) and ("args" in inputs or "kwargs" in inputs):
        args = tuple(inputs.get("args", ()))
        kwargs = dict(inputs.get("kwargs", {}))
    elif isinstance(inputs, Mapping):
        args = ()
        kwargs = dict(inputs)
    elif isinstance(inputs, tuple):
        args = inputs
        kwargs = {}
    else:
        args = (inputs,)
        kwargs = {}
    return tree_map(_clone_value, args), tree_map(_clone_value, kwargs)


def _floating_tensors(value: Any) -> list[torch.Tensor]:
    return [
        item for item in tree_flatten(value)[0]
        if torch.is_tensor(item) and (item.is_floating_point() or item.is_complex())
    ]


def _default_objective(output: Any) -> torch.Tensor:
    tensors = _floating_tensors(output)
    if not tensors:
        raise ValueError("parity output contains no differentiable tensor")
    return sum(tensor.float().sum() for tensor in tensors)


def _metric(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    exact: bool,
    atol: float,
    rtol: float,
    relative_l2_limit: float | None = None,
) -> ComparisonMetric:
    reference_cpu = reference.detach().float().cpu()
    candidate_cpu = candidate.detach().float().cpu()
    same_shape = tuple(reference_cpu.shape) == tuple(candidate_cpu.shape)
    finite = bool(torch.isfinite(reference_cpu).all() and torch.isfinite(candidate_cpu).all())
    if not same_shape:
        return ComparisonMetric(
            name=name,
            status="FAIL",
            max_abs=None,
            relative_l2=None,
            finite=finite,
            reference_shape=tuple(reference_cpu.shape),
            candidate_shape=tuple(candidate_cpu.shape),
            reason="shape mismatch",
        )
    equal = bool(torch.equal(reference_cpu, candidate_cpu))
    delta = candidate_cpu.double() - reference_cpu.double()
    max_abs = float(delta.abs().max()) if delta.numel() else 0.0
    reference_norm = torch.linalg.vector_norm(  # pylint: disable=not-callable
        reference_cpu.double().reshape(-1)
    )
    delta_norm = torch.linalg.vector_norm(delta.reshape(-1))  # pylint: disable=not-callable
    relative_l2 = float(delta_norm / reference_norm.clamp_min(torch.finfo(torch.float64).eps))
    passed = equal if exact else bool(torch.allclose(reference_cpu, candidate_cpu, atol=atol, rtol=rtol))
    if not exact and relative_l2_limit is not None:
        passed = passed and relative_l2 <= relative_l2_limit
    passed = passed and finite
    return ComparisonMetric(
        name=name,
        status="PASS" if passed else "FAIL",
        max_abs=max_abs,
        relative_l2=relative_l2,
        finite=finite,
        reference_shape=tuple(reference_cpu.shape),
        candidate_shape=tuple(candidate_cpu.shape),
    )


def _compare_tree(
    prefix: str,
    reference: Any,
    candidate: Any,
    *,
    exact: bool,
    atol: float,
    rtol: float,
    relative_l2_limit: float | None = None,
) -> list[ComparisonMetric]:
    reference_values, reference_spec = tree_flatten(reference)
    candidate_values, candidate_spec = tree_flatten(candidate)
    if reference_spec != candidate_spec:
        return [
            ComparisonMetric(
                name=prefix,
                status="FAIL",
                max_abs=None,
                relative_l2=None,
                finite=True,
                reference_shape=(),
                candidate_shape=(),
                reason=(
                    "output structure mismatch: "
                    f"reference={reference_spec!r}, candidate={candidate_spec!r}"
                ),
            )
        ]
    metrics = []
    for index, (reference_value, candidate_value) in enumerate(zip(reference_values, candidate_values)):
        name = f"{prefix}.{index}"
        if torch.is_tensor(reference_value) and torch.is_tensor(candidate_value):
            tensor_exact = exact or not (
                reference_value.is_floating_point() or reference_value.is_complex()
            )
            metrics.append(
                _metric(
                    name,
                    reference_value,
                    candidate_value,
                    exact=tensor_exact,
                    atol=atol,
                    rtol=rtol,
                    relative_l2_limit=relative_l2_limit,
                )
            )
        elif reference_value != candidate_value:
            metrics.append(
                ComparisonMetric(
                    name=name,
                    status="FAIL",
                    max_abs=None,
                    relative_l2=None,
                    finite=True,
                    reference_shape=(),
                    candidate_shape=(),
                    reason=f"non-tensor values differ: {reference_value!r} != {candidate_value!r}",
                )
            )
    return metrics


def _parameter_mapping(
    case: ModuleParityCase,
    reference: Any,
    candidate: Any,
    context: ParityContext,
) -> dict[str, str]:
    if case.weight_adapter is None:
        raise TypeError("in-process parity case requires weight_adapter")
    mapping_result = case.weight_adapter(reference, candidate, context)
    if mapping_result is None:
        mapping = {
            name: name for name, parameter in candidate.named_parameters()
            if parameter.requires_grad
        }
    elif isinstance(mapping_result, Mapping):
        mapping = {str(key): str(value) for key, value in mapping_result.items()}
    else:
        raise TypeError("parity weight_adapter must return a candidate-to-reference mapping or None")
    candidate_trainable = {
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    }
    reference_names = dict(reference.named_parameters())
    missing = sorted(candidate_trainable - set(mapping))
    invalid = sorted(set(mapping.values()) - set(reference_names))
    if missing or invalid:
        raise ValueError(
            "parity weight coverage is incomplete: "
            f"missing_candidate={missing}, unknown_reference={invalid}"
        )
    for pattern in case.required_parameter_patterns:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in candidate_trainable):
            raise ValueError(f"required parity parameter pattern matched nothing: {pattern!r}")
    return mapping


def _input_gradients(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[torch.Tensor | None]:
    return [
        value.grad if torch.is_tensor(value) and value.requires_grad else None
        for value in tree_flatten((args, kwargs))[0]
    ]


def _run_in_process_case(case: ModuleParityCase, context: ParityContext) -> dict[str, Any]:
    candidate_builder = case.candidate_builder
    reference_builder = case.reference_builder
    input_builder = case.input_builder
    if not all(callable(value) for value in (candidate_builder, reference_builder, input_builder)):
        raise TypeError("in-process parity case has an incomplete callable contract")
    candidate_tree = candidate_builder(context)
    candidate = _select_module(candidate_tree, case.candidate_selector)
    reference = reference_builder(context)
    if candidate is reference or type(candidate) is type(reference):
        raise ValueError(
            "reference and candidate must be independent implementations; "
            f"got {type(candidate).__module__}.{type(candidate).__qualname__}"
        )
    mapping = _parameter_mapping(case, reference, candidate, context)
    inputs = input_builder(context)
    reference_args, reference_kwargs = _clone_call_inputs(inputs)
    candidate_args, candidate_kwargs = _clone_call_inputs(inputs)
    reference_output = reference(*reference_args, **reference_kwargs)
    candidate_output = candidate(*candidate_args, **candidate_kwargs)
    tolerance = context.options.get("tolerance", {})
    atol = tolerance_value(tolerance, "atol", "max_abs")
    rtol = tolerance_value(tolerance, "rtol")
    relative_l2_limit = tolerance.get("relative_l2")
    if relative_l2_limit is not None:
        relative_l2_limit = float(relative_l2_limit)
    metrics = _compare_tree(
        "output",
        reference_output,
        candidate_output,
        exact=False,
        atol=atol,
        rtol=rtol,
        relative_l2_limit=relative_l2_limit,
    )
    reference_execution = ParityExecution(reference, reference_output, reference_args, reference_kwargs)
    candidate_execution = ParityExecution(candidate, candidate_output, candidate_args, candidate_kwargs)
    for observation in case.observations:
        reference_value = (
            observation.reference_getter(reference_execution)
            if observation.reference_getter is not None
            else reference_output
        )
        candidate_value = (
            observation.candidate_getter(candidate_execution)
            if observation.candidate_getter is not None
            else candidate_output
        )
        if reference_value is None or candidate_value is None:
            if observation.required:
                raise ValueError(
                    f"required parity observation {observation.name!r} was not produced: "
                    f"reference={reference_value is not None}, "
                    f"candidate={candidate_value is not None}"
                )
            if reference_value is None and candidate_value is None:
                continue
        metrics.extend(
            _compare_tree(
                observation.name,
                reference_value,
                candidate_value,
                exact=observation.comparison == "exact",
                atol=observation.atol if observation.atol is not None else atol,
                rtol=observation.rtol if observation.rtol is not None else rtol,
                relative_l2_limit=relative_l2_limit,
            )
        )
    objective = case.objective or _default_objective
    objective(reference_output).backward()
    objective(candidate_output).backward()
    reference_input_grads = _input_gradients(reference_args, reference_kwargs)
    candidate_input_grads = _input_gradients(candidate_args, candidate_kwargs)
    for index, (reference_grad, candidate_grad) in enumerate(
        zip(reference_input_grads, candidate_input_grads)
    ):
        if reference_grad is None and candidate_grad is None:
            continue
        if reference_grad is None or candidate_grad is None:
            raise ValueError(f"input gradient presence differs at flattened input {index}")
        metrics.append(
            _metric(
                f"input_grad.{index}",
                reference_grad,
                candidate_grad,
                exact=False,
                atol=atol,
                rtol=rtol,
                relative_l2_limit=relative_l2_limit,
            )
        )
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    for candidate_name, reference_name in sorted(mapping.items()):
        reference_grad = reference_parameters[reference_name].grad
        candidate_grad = candidate_parameters[candidate_name].grad
        if reference_grad is None and candidate_grad is None:
            continue
        if reference_grad is None or candidate_grad is None:
            raise ValueError(
                f"parameter gradient presence differs: {candidate_name} -> {reference_name}"
            )
        metrics.append(
            _metric(
                f"parameter_grad.{candidate_name}",
                reference_grad,
                candidate_grad,
                exact=False,
                atol=atol,
                rtol=rtol,
                relative_l2_limit=relative_l2_limit,
            )
        )
    status = "PASS" if all(metric.status == "PASS" for metric in metrics) else "FAIL"
    return {
        "case": case.name,
        "status": status,
        "candidate_type": f"{type(candidate).__module__}.{type(candidate).__qualname__}",
        "reference_type": f"{type(reference).__module__}.{type(reference).__qualname__}",
        "weight_mapping": mapping,
        "metrics": [metric.to_dict() for metric in metrics],
    }


def run_module_parity(
    validation_spec: ModelValidationSpec,
    context: ParityContext,
) -> dict[str, Any]:
    """Execute every declared parity case and persist per-case evidence."""
    reports = []
    for case in validation_spec.module_cases:
        if context.dtype not in case.required_dtypes:
            continue
        try:
            if case.execution == "isolated_process":
                report = case.isolated_runner(context)  # type: ignore[misc]
                if not isinstance(report, dict) or report.get("status") not in (
                    "PASS", "FAIL", "BLOCKED"
                ):
                    raise ValueError("isolated parity runner returned an invalid report")
            else:
                report = _run_in_process_case(case, context)
        except NotImplementedError as error:
            report = {
                "case": case.name,
                "status": "BLOCKED",
                "reason": "OPERATOR_UNSUPPORTED",
                "error": str(error),
            }
        except Exception as error:  # pylint: disable=broad-except
            report = {
                "case": case.name,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        reports.append(report)
        context.evidence_store.write_json(
            f"module_parity/{case.name}/comparison.json",
            report,
        )
    statuses = [report["status"] for report in reports]
    status = (
        "FAIL" if "FAIL" in statuses
        else "BLOCKED" if "BLOCKED" in statuses or not statuses
        else "PASS"
    )
    aggregate = {"status": status, "cases": reports}
    if not reports:
        aggregate["reason"] = (
            "no module parity case supports the requested dtype, or the adapter "
            "declared no authoritative case"
        )
    context.evidence_store.write_json("module_parity/comparison.json", aggregate)
    return aggregate


__all__ = ["ParityContext", "ParityExecution", "run_module_parity"]
