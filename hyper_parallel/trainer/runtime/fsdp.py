# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-side FSDP lifecycle policies and runtime diagnostics.

Split out of the former ``auto_models/trainer/base.py`` in stage 7
(05 §15.11 step 3). ``BaseTrainer`` keeps the same-named methods as thin
delegating subclass hooks; the policy itself lives here. The FSDP config is
duck-typed (``reshard_after_backward`` / ``dp_shard_size`` /
``requires_grad_sync`` attributes) so this module does not import Trainer DTOs.
"""
# pylint: disable=forbidden-backend-import

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from torch.utils._pytree import tree_flatten


logger = logging.getLogger(__name__)


@dataclass
class _FSDPMicroStepTrace:
    """Observed FSDP unit order for one forward/backward micro-step."""

    global_step: int
    micro_step: int
    forward: list[str] = field(default_factory=list)
    backward: list[str] = field(default_factory=list)


class FSDPRuntimeDiagnostics:
    """Validate Torch FSDP prefetch order at the optimizer-step boundary.

    This observer does not launch, wait, release, or otherwise mutate FSDP
    communication. It records the actual module execution order and inspects
    outstanding all-gather state immediately before the optimizer may update
    parameter shards.
    """

    def __init__(self, hsdp_model_parts: List[Any]) -> None:
        """Attach diagnostic hooks to all HSDP units in the model parts."""
        self._module_by_name: dict[str, Any] = {}
        self._name_by_module_id: dict[int, str] = {}
        self._scheduler_by_name: dict[str, Any] = {}
        self._hook_handles: list[Any] = []
        self._traces: list[_FSDPMicroStepTrace] = []
        self._active_trace: Optional[_FSDPMicroStepTrace] = None
        self._discover_units(hsdp_model_parts)
        self._register_hooks()

    @property
    def unit_count(self) -> int:
        """Return the number of observed HSDP modules."""
        return len(self._module_by_name)

    def _discover_units(self, hsdp_model_parts: List[Any]) -> None:
        """Build stable names for HSDP modules without importing backend internals."""
        multiple_parts = len(hsdp_model_parts) > 1
        for part_index, model_part in enumerate(hsdp_model_parts):
            part_prefix = f"part[{part_index}]" if multiple_parts else ""
            for module_fqn, module in model_part.named_modules():
                scheduler = getattr(module, "hsdp_scheduler", None)
                if scheduler is None or id(module) in self._name_by_module_id:
                    continue
                if part_prefix and module_fqn:
                    module_name = f"{part_prefix}.{module_fqn}"
                elif part_prefix:
                    module_name = part_prefix
                else:
                    module_name = module_fqn or "<root>"
                self._module_by_name[module_name] = module
                self._name_by_module_id[id(module)] = module_name
                self._scheduler_by_name[module_name] = scheduler

    def _register_hooks(self) -> None:
        """Register hooks on every unit for order and child-bypass diagnostics."""
        for module_name, module in self._module_by_name.items():
            self._hook_handles.append(
                module.register_forward_pre_hook(
                    self._make_forward_pre_hook(module_name)
                )
            )
            self._hook_handles.append(
                module.register_forward_hook(
                    self._make_backward_registration_hook(module_name)
                )
            )

    def _make_forward_pre_hook(self, module_name: str):
        """Create a hook recording non-recompute forward execution."""

        def record_forward(module: Any, inputs: Any) -> None:
            """Record one observed module forward entry."""
            del inputs
            if self._active_trace is None:
                return
            scheduler = getattr(module, "hsdp_scheduler", None)
            scheduler_ctx = getattr(scheduler, "scheduler_ctx", None)
            if getattr(scheduler_ctx, "root_bp_state", False):
                return
            self._active_trace.forward.append(module_name)

        return record_forward

    def _make_backward_registration_hook(self, module_name: str):
        """Create a forward hook attaching a one-shot output-gradient observer."""

        def register_backward_observer(module: Any, inputs: Any, outputs: Any) -> None:
            """Attach a one-shot gradient observer to differentiable outputs."""
            del inputs
            scheduler = getattr(module, "hsdp_scheduler", None)
            scheduler_ctx = getattr(scheduler, "scheduler_ctx", None)
            if getattr(scheduler_ctx, "root_bp_state", False):
                return
            output_tensors = [
                output
                for output in tree_flatten(outputs)[0]
                if torch.is_tensor(output) and output.requires_grad
            ]
            if not output_tensors:
                return

            fired = [False]
            hook_handles = []

            def record_backward(gradient: Any) -> Any:
                """Record the module on its first output-gradient callback."""
                if not fired[0]:
                    fired[0] = True
                    if self._active_trace is not None:
                        self._active_trace.backward.append(module_name)
                    for hook_handle in hook_handles:
                        hook_handle.remove()
                return gradient

            for output_tensor in output_tensors:
                hook_handles.append(output_tensor.register_hook(record_backward))

        return register_backward_observer

    def begin_micro_step(self, global_step: int, micro_step: int) -> None:
        """Start tracing one micro-step in the current optimizer step."""
        if micro_step == 0:
            self._traces.clear()
        trace = _FSDPMicroStepTrace(global_step, micro_step)
        self._traces.append(trace)
        self._active_trace = trace

    @staticmethod
    def _first_positions(execution_order: list[str]) -> dict[str, int]:
        """Return first-observed positions while preserving repeated calls in logs."""
        positions = {}
        for position, module_name in enumerate(execution_order):
            positions.setdefault(module_name, position)
        return positions

    def _prefetch_target_names(self, module_name: str, phase: str) -> list[str]:
        """Resolve configured prefetch targets to diagnostic module names."""
        scheduler = self._scheduler_by_name[module_name]
        attribute = (
            "forward_prefetch_cells"
            if phase == "forward"
            else "backward_prefetch_cells"
        )
        target_names = []
        for target in getattr(scheduler, attribute, ()):
            target_names.append(
                self._name_by_module_id.get(
                    id(target),
                    f"<untracked:{type(target).__name__}>",
                )
            )
        return target_names

    def _find_order_violations(self) -> list[str]:
        """Compare configured prefetch edges with each observed execution trace."""
        violations = []
        for trace in self._traces:
            for phase in ("forward", "backward"):
                execution_order = getattr(trace, phase)
                positions = self._first_positions(execution_order)
                for source_name, source_position in positions.items():
                    for target_name in self._prefetch_target_names(source_name, phase):
                        target_position = positions.get(target_name)
                        if target_position is None:
                            violations.append(
                                f"step={trace.global_step} micro_step={trace.micro_step} "
                                f"phase={phase}: {source_name} prefetched {target_name}, "
                                "but the target did not execute"
                            )
                        elif target_position <= source_position:
                            violations.append(
                                f"step={trace.global_step} micro_step={trace.micro_step} "
                                f"phase={phase}: {source_name} prefetched {target_name} "
                                f"at positions {source_position}->{target_position}"
                            )
        return violations

    @staticmethod
    def _parameter_name(hsdp_param: Any, param_index: int) -> str:
        """Return a useful parameter name for diagnostics."""
        return getattr(hsdp_param, "_param_fqn", None) or f"param[{param_index}]"

    def _find_pending_all_gathers(self) -> list[str]:
        """Find unconsumed fused and non-fused all-gathers without mutating them."""
        pending = []
        seen_schedulers = set()
        for module_name, scheduler in self._scheduler_by_name.items():
            if id(scheduler) in seen_schedulers:
                continue
            seen_schedulers.add(id(scheduler))
            state = getattr(scheduler, "hsdp_state", None)
            if state is None:
                continue

            param_group = getattr(state, "param_group", None)
            for bucket_index, bucket in enumerate(
                getattr(param_group, "all_gather_buckets", ())
            ):
                result = getattr(bucket, "all_gather_result", None)
                if result is not None:
                    handle = getattr(result, "handle", None)
                    pending.append(
                        f"{module_name}: fused_bucket[{bucket_index}] "
                        f"handle={'set' if handle is not None else 'completed-not-consumed'}"
                    )

            for param_index, hsdp_param in enumerate(getattr(state, "hsdp_params", ())):
                comm_ctx = getattr(hsdp_param, "allgather_comm_ctx", None)
                handle = getattr(comm_ctx, "allgather_handle", None)
                output = getattr(comm_ctx, "allgather_output", None)
                if handle is not None or output is not None:
                    state_details = []
                    if handle is not None:
                        state_details.append("handle=set")
                    if output is not None:
                        state_details.append("output=not-consumed")
                    pending.append(
                        f"{module_name}: "
                        f"{self._parameter_name(hsdp_param, param_index)} "
                        + ", ".join(state_details)
                    )
        return pending

    @staticmethod
    def _hsdp_parameter_has_gradient(hsdp_param: Any) -> bool:
        """Return whether a managed parameter participated in backward."""
        sharded_param = getattr(hsdp_param, "sharded_param", None)
        candidates = (
            getattr(sharded_param, "grad", None),
            getattr(sharded_param, "main_grad", None),
            getattr(hsdp_param, "_grad", None),
            getattr(hsdp_param, "unsharded_accumulated_grad", None),
            getattr(hsdp_param, "unsharded_accumulated_grad_data", None),
        )
        return any(value is not None for value in candidates)

    def _find_bypassed_child_units(self) -> list[str]:
        """Find child FSDP parameters used without executing their owner module."""
        if not self._traces:
            return []
        forward_counts = defaultdict(int)
        for trace in self._traces:
            for module_name in trace.forward:
                forward_counts[module_name] += 1
        bypassed = []
        root_names = {"<root>"}
        root_names.update(
            name for name in self._module_by_name
            if name.startswith("part[") and "." not in name
        )
        for module_name, scheduler in self._scheduler_by_name.items():
            if module_name in root_names or forward_counts[module_name] > 0:
                continue
            state = getattr(scheduler, "hsdp_state", None)
            gradient_parameters = [
                self._parameter_name(hsdp_param, param_index)
                for param_index, hsdp_param in enumerate(getattr(state, "hsdp_params", ()))
                if self._hsdp_parameter_has_gradient(hsdp_param)
            ]
            if gradient_parameters:
                bypassed.append(
                    f"{module_name}: forward_count=0, gradient_parameters={gradient_parameters}"
                )
        return bypassed

    def validate_before_optimizer_step(self) -> None:
        """Fail before parameter updates if the traced FSDP lifecycle is unsafe."""
        self._active_trace = None
        order_violations = self._find_order_violations()
        pending_all_gathers = self._find_pending_all_gathers()
        bypassed_child_units = self._find_bypassed_child_units()
        error_sections = []
        if bypassed_child_units:
            error_sections.append(
                "[HP-FSDP-007] child FSDP module was bypassed while its parameter "
                "participated in backward:\n  - " + "\n  - ".join(bypassed_child_units)
            )
        if order_violations:
            error_sections.append(
                "[HP-FSDP-008] configured prefetch targets do not follow "
                "the observed runtime order:\n  - " + "\n  - ".join(order_violations)
            )
        if pending_all_gathers:
            error_sections.append(
                "[HP-FSDP-009] unconsumed parameter all-gather exists at "
                "the optimizer boundary:\n  - " + "\n  - ".join(pending_all_gathers)
            )
        if error_sections:
            raise RuntimeError(
                "\n".join(error_sections)
                + "\nThe optimizer was not executed. Configure FSDP prefetch from the "
                "model's actual forward/backward execution order; do not carry an "
                "all-gather across optimizer.step()."
            )

    def trace_evidence(self) -> dict[str, Any]:
        """Return stable execution-order evidence for structured reports."""
        return {
            "units": sorted(self._module_by_name),
            "micro_steps": [
                {
                    "global_step": trace.global_step,
                    "micro_step": trace.micro_step,
                    "forward": list(trace.forward),
                    "backward": list(trace.backward),
                }
                for trace in self._traces
            ],
        }

    def close(self) -> None:
        """Remove diagnostic hooks when a caller no longer needs the observer."""
        for hook_handle in self._hook_handles:
            hook_handle.remove()
        self._hook_handles.clear()


def build_fsdp_runtime_diagnostics(
    hsdp_model_parts: List[Any],
) -> Optional[FSDPRuntimeDiagnostics]:
    """Create diagnostics for Torch Trainer model parts, or return ``None``."""
    if not hsdp_model_parts:
        return None
    diagnostics = FSDPRuntimeDiagnostics(hsdp_model_parts)
    if diagnostics.unit_count == 0:
        diagnostics.close()
        return None
    logger.info(
        "Enabled Trainer FSDP runtime diagnostics for %d modules",
        diagnostics.unit_count,
    )
    return diagnostics


def model_reshard(
    hsdp_model_parts: List[Any],
    fsdp_config: Any,
    micro_step: int,
    num_micro_steps: int,
) -> None:
    """Reshard model after backward pass."""
    if (
            fsdp_config.reshard_after_backward is False
            and num_micro_steps > 1
    ):
        if micro_step == 0:
            for model_part in hsdp_model_parts:
                model_part.set_reshard_after_backward(False)
        elif micro_step == num_micro_steps - 1:
            for model_part in hsdp_model_parts:
                model_part.set_reshard_after_backward(True)


def configure_fsdp_gradient_sync(
    hsdp_model_parts: List[Any],
    fsdp_config: Any,
    dp_replicate_size: int,
    micro_step: int,
    num_micro_steps: int,
) -> None:
    """Configure FSDP gradient synchronization for one micro step."""
    if (
            fsdp_config.dp_shard_size > 1
            and num_micro_steps > 1
    ):
        is_last_micro_batch = micro_step == num_micro_steps - 1
        requires_gradient_sync = (
            fsdp_config.requires_grad_sync
            or is_last_micro_batch
        )
        is_hsdp = dp_replicate_size > 1
        for model_part in hsdp_model_parts:
            model_part.set_requires_gradient_sync(requires_gradient_sync)
            model_part.set_is_last_backward(is_last_micro_batch)
            if is_hsdp:
                model_part.set_requires_all_reduce(is_last_micro_batch)


__all__ = [
    "FSDPRuntimeDiagnostics",
    "build_fsdp_runtime_diagnostics",
    "configure_fsdp_gradient_sync",
    "model_reshard",
]
