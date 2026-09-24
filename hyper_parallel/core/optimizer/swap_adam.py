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
"""Adam/AdamW optimizer-state swap adapters."""

# pylint: disable=protected-access
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Sequence, Tuple

import torch

from hyper_parallel.core.optimizer.swap_optimizer_base import (
    StateSwapAdapter,
    SwapOptimizer as CoreSwapOptimizer,
    SwapSlot,
    _slot_tensor,
)

#: Logical state keys an Adam/AdamW optimizer owns.
ADAM_STATE_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


class GroupArgs(NamedTuple):
    """Per-parameter argument lists one functional Adam/AdamW step consumes."""

    params: List[Any]
    grads: List[Any]
    exp_avgs: List[Any]
    exp_avg_sqs: List[Any]
    max_exp_avg_sqs: List[Any]
    state_steps: List[Any]


@dataclass
class AdamUpdateUnit:
    """Per-parameter Adam/AdamW update unit consumed by the pipeline runtime.

    ``adapter_index`` identifies the Torch parameter group owning ``param``.
    """

    adapter_index: int
    param: Any
    grad: Any
    slots: List[SwapSlot]


class AdamSwapAdapter(StateSwapAdapter):
    """Adam/AdamW optimizer-state swap adapter."""

    functional_name = "adam"
    decoupled_weight_decay = False
    is_new_adamw = False
    supports_fused = False

    def validate(self) -> None:
        """Validate unsupported optimizer flags."""
        for group in self.optimizer.param_groups:
            if group.get("foreach", False) is True:
                raise ValueError("Swap optimizer does not support foreach=True.")
            if group.get("fused", False) is True and not self.supports_fused:
                raise ValueError("Swap optimizer does not support fused=True.")
            if group.get("differentiable", False):
                raise ValueError("Swap optimizer does not support differentiable=True.")
            if group.get("capturable", False):
                raise ValueError("Swap optimizer does not support capturable=True.")

    def prepare_step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Initialize lazy state and collect this step's update units."""
        if args or kwargs:
            raise ValueError("Torch swap optimizer step does not support closure or extra arguments.")
        if self.runtime.packed_enabled:
            return self._prepare_packed_step()

        units = []
        for group_index, group in enumerate(self.optimizer.param_groups):
            if self.is_new_adamw:
                group["step"] = (group.get("step") or 0) + 1
            for param in group["params"]:
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue
                if getattr(grad, "is_sparse", False):
                    raise ValueError("Swap optimizer only supports dense Adam/AdamW gradients.")
                state = self.optimizer.state[param]
                self._init_param_state(param, grad, group)
                slots = self._build_slots(param, state)
                units.append(AdamUpdateUnit(
                    adapter_index=group_index,
                    param=param,
                    grad=grad,
                    slots=slots,
                ))
        return {"units": units}

    def _prepare_packed_step(self) -> Dict[str, Any]:
        """Build a stable packed layout while retaining inactive materialized states."""
        records = []
        for group_index, group in enumerate(self.optimizer.param_groups):
            if self.is_new_adamw:
                group["step"] = (group.get("step") or 0) + 1
            for param in group["params"]:
                grad = getattr(param, "grad", None)
                state = self.optimizer.state.get(param)
                if grad is not None:
                    if getattr(grad, "is_sparse", False):
                        raise ValueError("Swap optimizer only supports dense Adam/AdamW gradients.")
                    state = self.optimizer.state[param]
                    self._init_param_state(param, grad, group)
                if state:
                    self._register_present_slots(param, state)
                has_slots = any((id(param), key) in self._slots for key in self._configured_state_keys())
                if grad is None and not has_slots:
                    continue
                records.append((group_index, param, grad))

        self.runtime.prepare_packed_host(self._ordered_slots())
        self.publish_packed_state()
        units = []
        for group_index, param, grad in records:
            state = self.optimizer.state[param]
            slots = self._build_slots(param, state)
            if grad is None and not any(slot.swappable and slot.packed for slot in slots):
                continue
            units.append(AdamUpdateUnit(
                adapter_index=group_index,
                param=param,
                grad=grad,
                slots=slots,
            ))
        return {"units": units}

    def _collect_group_args(
            self,
            units: Sequence[AdamUpdateUnit],
            group: Dict[str, Any],
    ) -> GroupArgs:
        """Gather the per-parameter argument lists one optimizer step needs.

        Units without a gradient are skipped.  ``max_exp_avg_sqs`` stays empty
        unless the group is amsgrad, and ``state_steps`` holds ``None`` for new
        AdamW, which advances the step counter itself.
        """
        args = GroupArgs([], [], [], [], [], [])
        for unit in units:
            if unit.grad is None:
                continue
            state = self.optimizer.state[unit.param]
            args.params.append(unit.param)
            args.grads.append(unit.grad)
            args.exp_avgs.append(_slot_tensor(unit, "exp_avg", state["exp_avg"]))
            args.exp_avg_sqs.append(_slot_tensor(unit, "exp_avg_sq", state["exp_avg_sq"]))
            if group.get("amsgrad", False):
                args.max_exp_avg_sqs.append(
                    _slot_tensor(unit, "max_exp_avg_sq", state["max_exp_avg_sq"])
                )
            if self.is_new_adamw:
                args.state_steps.append(None)
            else:
                args.state_steps.append(state["step"])
        return args

    def step_batch(self, batch: List[AdamUpdateUnit], step_context: Dict[str, Any]) -> None:
        """Run Torch functional Adam/AdamW for one batch."""
        del step_context
        by_group: Dict[int, List[AdamUpdateUnit]] = defaultdict(list)
        for unit in batch:
            by_group[unit.adapter_index].append(unit)
        for group_index, units in by_group.items():
            self._step_group(self.optimizer.param_groups[group_index], units)

    def _step_group(self, group: Dict[str, Any], units: List[AdamUpdateUnit]) -> None:
        """Run one group's parameters through the matching functional Adam/AdamW."""
        args = self._collect_group_args(units, group)
        params = args.params

        if not params:
            return

        if self.is_new_adamw:
            self._step_new_adamw(
                group,
                args.params,
                args.grads,
                args.exp_avgs,
                args.exp_avg_sqs,
                args.max_exp_avg_sqs,
            )
            return

        func = getattr(torch.optim._functional, self.functional_name)
        kwargs = {
            "amsgrad": group["amsgrad"],
            "beta1": group["betas"][0],
            "beta2": group["betas"][1],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "eps": group["eps"],
            "maximize": group["maximize"],
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": bool(group.get("fused", False)),
            "grad_scale": getattr(self.optimizer, "grad_scale", None),
            "found_inf": getattr(self.optimizer, "found_inf", None),
            "has_complex": False,
        }
        if self.functional_name == "adam":
            if "decoupled_weight_decay" in inspect.signature(func).parameters:
                kwargs["decoupled_weight_decay"] = self._decoupled_weight_decay(group)
        func(
            args.params,
            args.grads,
            args.exp_avgs,
            args.exp_avg_sqs,
            args.max_exp_avg_sqs,
            args.state_steps,
            **kwargs,
        )

    def _decoupled_weight_decay(self, group: Dict[str, Any]) -> bool:
        """Resolve the decoupled weight decay flag for one parameter group."""
        return self.decoupled_weight_decay or group.get("decoupled_weight_decay", False)

    def _step_new_adamw(
            self,
            group: Dict[str, Any],
            params: Sequence[Any],
            grads: Sequence[Any],
            exp_avgs: Sequence[Any],
            exp_avg_sqs: Sequence[Any],
            max_exp_avg_sqs: Sequence[Any],
    ) -> None:
        """Run the new AdamW functional for one group."""
        if params and params[0].device.type == "cpu":
            # torch.optim._functional.adamw increments tensor state_steps
            # internally. New AdamW already advanced group["step"] in
            # prepare_step(), so feed step - 1 to preserve outer-step
            # semantics for CPU-only tests.
            step_tensor = torch.tensor(float(group["step"] - 1), dtype=torch.float32)
            torch.optim._functional.adamw(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                [step_tensor] * len(params),
                amsgrad=group["amsgrad"],
                beta1=group["betas"][0],
                beta2=group["betas"][1],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=False,
                capturable=False,
                differentiable=False,
                fused=False,
                grad_scale=None,
                found_inf=None,
                has_complex=False,
            )
            return
        _new_adamw_func()(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            group["step"],
            amsgrad=group["amsgrad"],
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
        )

    def _init_param_state(self, param: Any, grad: Any, group: Dict[str, Any]) -> None:
        """Initialize missing Adam state and swap slots for one parameter."""
        del grad
        state = self.optimizer.state[param]
        if not self.is_new_adamw and len(state) == 0:
            step_device = (
                param.device
                if group.get("fused", False)
                else ("cpu" if self.runtime.packed_enabled else param.device)
            )
            state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
        state_keys = ["exp_avg", "exp_avg_sq"]
        if group.get("amsgrad", False):
            state_keys.append("max_exp_avg_sq")
        configured_keys = set(self._configured_state_keys())
        for key in state_keys:
            if key in state or (id(param), key) in self._slots:
                continue
            if (
                    key in configured_keys
                    and not self.runtime.packed_enabled
                    and self.runtime.is_swappable_tensor(param, self.config.min_numel)
            ):
                cpu_tensor = self.runtime.make_zero_cpu_tensor_like(param)
                device_tensor = torch.empty_like(param, memory_format=torch.preserve_format)
                state[key] = device_tensor
                slot = self._make_slot(key, device_tensor)
                slot.cpu_tensor = cpu_tensor
                slot.state = "host"
                self._slots[(id(param), key)] = slot
                self.runtime.release_device_storage(slot)
                continue
            if key in configured_keys and self.runtime.is_packable_template(param, self.config.min_numel):
                self._slots[(id(param), key)] = self._make_slot(key, None, template=param)
                continue
            state[key] = torch.zeros_like(param, memory_format=torch.preserve_format)

    @staticmethod
    def _is_active_device_slot(slot: SwapSlot, key: str) -> bool:
        """Return whether ``slot`` holds ``key`` live on the local device."""
        if slot.name != key or not slot.swappable or slot.state != "device":
            return False
        return slot.tensor is not None

    @staticmethod
    def default_state_keys() -> Tuple[str, ...]:
        """Return the swap state keys an Adam/AdamW optimizer owns."""
        return ADAM_STATE_KEYS


class TorchNativeAdamAdapter(AdamSwapAdapter):
    """Adapter for ``torch.optim.Adam``."""

    functional_name = "adam"

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        """Match ``torch.optim.Adam``, excluding its AdamW subclass."""
        # AdamW inherits Adam in PyTorch.  Keep Adam subclasses supported, but
        # let AdamW select its dedicated adapter (which preserves fused=True).
        return (
            isinstance(optimizer, torch.optim.Adam)
            and not isinstance(optimizer, torch.optim.AdamW)
        )


class TorchNativeAdamWAdapter(AdamSwapAdapter):
    """Adapter for ``torch.optim.AdamW``."""

    functional_name = "adamw"
    supports_fused = True

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        """Match ``torch.optim.AdamW``."""
        return isinstance(optimizer, torch.optim.AdamW)


class TorchNewAdamWAdapter(AdamSwapAdapter):
    """Adapter for hyper-parallel's fused AdamW."""

    functional_name = "adamw"
    is_new_adamw = True

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        """Match hyper-parallel's lazily imported AdamW."""
        return isinstance(optimizer, _new_adamw_cls())

def _new_adamw_cls() -> type:
    """Return hyper-parallel's AdamW class without importing it at module load."""
    from hyper_parallel.core.optimizer.adamw import (  # pylint: disable=import-outside-toplevel
        AdamW as new_adamw_cls,
    )
    return new_adamw_cls


def _new_adamw_func() -> Callable[..., Any]:
    """Return hyper-parallel's functional AdamW without importing it at module load."""
    from hyper_parallel.core.optimizer.adamw import (  # pylint: disable=import-outside-toplevel
        adamw as new_adamw_func,
    )
    return new_adamw_func


ADAM_ADAPTERS = (
    TorchNewAdamWAdapter,
    TorchNativeAdamAdapter,
    TorchNativeAdamWAdapter,
)


def build_adam_swap_adapter(optimizer: Any, config: Any, runtime: Any) -> AdamSwapAdapter:
    """Build the Adam/AdamW swap adapter matching ``optimizer``.

    Args:
        optimizer: Wrapped Torch or HyperParallel Adam/AdamW optimizer.
        config: Resolved ``SwapOptimizerConfig``.
        runtime: Shared ``PipelineSwapRuntime``.

    Returns:
        The adapter instance for ``optimizer``.

    Raises:
        ValueError: If ``optimizer`` is not a supported Adam/AdamW optimizer.
    """
    for adapter_cls in ADAM_ADAPTERS:
        if adapter_cls.matches(optimizer):
            return adapter_cls(optimizer, config, runtime)
    raise ValueError(
        "Swap optimizer only supports torch.optim.Adam, torch.optim.AdamW, "
        "and hyper_parallel.core.optimizer.adamw.AdamW on the Torch backend. "
        f"Got {type(optimizer)!r}."
    )


def swap_adam(optimizer: Any, config: Any) -> Any:
    """Wrap an Adam/AdamW optimizer with optimizer-state swap.

    Args:
        optimizer: Supported Adam/AdamW optimizer instance.
        config: Resolved ``SwapOptimizerConfig``.

    Returns:
        The shared core swap-optimizer wrapper.
    """
    return CoreSwapOptimizer(
        optimizer,
        config,
        adapter_factory=build_adam_swap_adapter,
    )
