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
"""Torch Adam/AdamW swap optimizer adapters."""
# pylint: disable=protected-access

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from hyper_parallel.core.optimizer.adamw import AdamW as HyperAdamW
from hyper_parallel.core.optimizer.adamw import adamw as hyper_adamw
from hyper_parallel.core.optimizer.swap_optimizer_base import (
    OptimizerSwapAdapter,
    SwapSlot,
    UpdateUnit,
)


class TorchAdamBaseAdapter(OptimizerSwapAdapter):
    """Common Torch Adam/AdamW adapter logic."""

    functional_name = "adam"
    supported_cls = ()
    decoupled_weight_decay = False
    is_hyper_adamw = False
    supports_fused = False

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        """Return whether this adapter supports ``optimizer``."""
        return isinstance(optimizer, cls.supported_cls)

    def __init__(self, optimizer: Any, config: Any, runtime: Any) -> None:
        super().__init__(optimizer, config, runtime)
        self._slots: Dict[Tuple[int, str], SwapSlot] = {}

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
            if self.is_hyper_adamw:
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
                units.append(UpdateUnit(
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
            if self.is_hyper_adamw:
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
            units.append(UpdateUnit(
                adapter_index=group_index,
                param=param,
                grad=grad,
                slots=slots,
            ))
        return {"units": units}

    def iter_update_units(self, step_context: Dict[str, Any]) -> List[UpdateUnit]:
        """Return units collected in ``prepare_step``."""
        return step_context["units"]

    def initial_slots(self) -> Iterable[SwapSlot]:
        """Discover optimizer states materialized before the swap wrapper was created."""
        slots = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if state:
                    slots.extend(self._build_slots(param, state))
        return tuple(slots)

    def step_batch(self, batch: List[UpdateUnit], step_context: Dict[str, Any]) -> None:
        """Run Torch functional Adam/AdamW for one batch."""
        del step_context
        by_group: Dict[int, List[UpdateUnit]] = defaultdict(list)
        for unit in batch:
            by_group[unit.adapter_index].append(unit)
        for group_index, units in by_group.items():
            group = self.optimizer.param_groups[group_index]
            state_steps = []
            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            for unit in units:
                if unit.grad is None:
                    continue
                state = self.optimizer.state[unit.param]
                params.append(unit.param)
                grads.append(unit.grad)
                exp_avgs.append(self._slot_tensor(unit, "exp_avg", state["exp_avg"]))
                exp_avg_sqs.append(self._slot_tensor(unit, "exp_avg_sq", state["exp_avg_sq"]))
                if group.get("amsgrad", False):
                    max_exp_avg_sqs.append(
                        self._slot_tensor(unit, "max_exp_avg_sq", state["max_exp_avg_sq"])
                    )
                if self.is_hyper_adamw:
                    state_steps.append(None)
                else:
                    state_steps.append(state["step"])

            if not params:
                continue

            if self.is_hyper_adamw:
                if params and params[0].device.type == "cpu":
                    # torch.optim._functional.adamw increments tensor state_steps
                    # internally. Hyper AdamW already advanced group["step"] in
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
                else:
                    hyper_adamw(
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
                continue

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
                kwargs["decoupled_weight_decay"] = self.decoupled_weight_decay or group.get(
                    "decoupled_weight_decay", False
                )
            func(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, **kwargs)

    def all_slots(self):
        """Iterate known swap slots."""
        return tuple(self._ordered_slots())

    def checkpoint_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return optimizer checkpoint state using CPU mirrors for swapped slots."""
        del args, kwargs
        self.runtime.synchronize_cpu_mirrors(self.all_slots())
        return self.export_swappable_state(self.optimizer.state_dict())

    def load_checkpoint_state_dict(
            self,
            state_dict: Dict[str, Any],
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """Load optimizer checkpoint state while restoring swap-managed slots."""
        del args, kwargs
        stripped, removed = self.strip_swappable_state(state_dict)
        self.optimizer.load_state_dict(stripped)
        self.load_swappable_state(state_dict, removed)

    def publish_packed_state(self) -> None:
        """Publish persistent packed CPU mirrors to the wrapped optimizer state."""
        if not self.runtime.packed_enabled:
            return
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                for key in self._configured_state_keys():
                    slot = self._slots.get((id(param), key))
                    if slot is None or not slot.packed or slot.cpu_tensor is None:
                        continue
                    if state is None:
                        state = self.optimizer.state[param]
                    state[key] = slot.cpu_tensor

    def export_swappable_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build a checkpoint-safe Torch optimizer state dict.

        Torch optimizer state dicts are keyed by saved parameter ids, while the
        adapter tracks live swap slots by the current parameter objects.  This
        method walks both orders together and exports each parameter's optimizer
        state with the data source that currently owns the valid tensor values.

        If an Adam state tensor such as ``exp_avg`` or ``exp_avg_sq`` has been
        offloaded, the live device tensor may only be a placeholder with its
        storage released.  In that case, write a cloned CPU mirror into the
        exported state dict so checkpoints contain the real optimizer values.
        Non-swappable state, metadata, and tensors that are still resident on
        device are deep-copied from the original Torch state dict unchanged.
        """
        exported = {
            key: copy.deepcopy(value)
            for key, value in state_dict.items()
            if key not in ("state", "param_groups")
        }
        exported["param_groups"] = copy.deepcopy(state_dict.get("param_groups", []))
        exported["state"] = {}
        saved_groups = exported.get("param_groups", [])
        params_in_order = []
        for group in self.optimizer.param_groups:
            params_in_order.extend(group["params"])
        ids_in_order = []
        for group in saved_groups:
            ids_in_order.extend(group["params"])
        for param, param_id in zip(params_in_order, ids_in_order):
            saved_state = state_dict.get("state", {}).get(param_id)
            if not saved_state:
                continue
            exported_state = {}
            for key in self._state_keys_for_param(param):
                slot = self._slots.get((id(param), key))
                if slot is not None and slot.state == "host" and slot.cpu_tensor is not None and key in saved_state:
                    exported_state[key] = slot.cpu_tensor.detach().clone()
            for key, value in saved_state.items():
                if key not in exported_state:
                    exported_state[key] = copy.deepcopy(value)
            exported["state"][param_id] = exported_state
        return exported

    def strip_swappable_state(self, state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
        """Split Adam state tensors out before delegating to Torch loading.

        PyTorch's ``load_state_dict`` eagerly restores tensors into the
        optimizer state.  For swap-managed Adam buffers, that would bypass the
        adapter/runtime bookkeeping and can place large tensors directly on the
        device.  This method therefore deep-copies the checkpoint state dict,
        removes the Adam buffers that may be swap-managed, and returns them in a
        side table keyed by the checkpoint parameter id.

        The stripped state dict is safe to pass to the wrapped optimizer's
        ``load_state_dict`` for ordinary fields such as parameter groups and
        step counters.  The removed tensors must be handed to
        ``load_swappable_state`` afterwards so they can be restored with the
        correct CPU mirror/device placeholder layout.
        """
        stripped = copy.deepcopy(state_dict)
        removed: Dict[int, Dict[str, Any]] = {}
        swappable_keys = self._configured_state_keys()
        for param_id, saved_state in list(stripped.get("state", {}).items()):
            if not isinstance(saved_state, dict):
                continue
            for key in swappable_keys:
                if key in saved_state:
                    removed.setdefault(param_id, {})[key] = saved_state.pop(key)
        return stripped, removed

    def load_swappable_state(self, original_state_dict: Dict[str, Any], removed: Dict[int, Dict[str, Any]]) -> None:
        """Restore removed Adam buffers under swap runtime control.

        This is the second half of checkpoint loading.  After Torch has loaded
        the stripped state dict, this method maps checkpoint parameter ids back
        to the current parameter objects by walking saved and current parameter
        groups in order.  Each removed Adam buffer is then recreated as an
        optimizer state entry and registered as a ``SwapSlot``.

        Packed runtimes place checkpoint values directly in persistent pinned
        host views. Legacy runtimes retain an empty device placeholder whose
        storage is restored only during prefetch. Buffers that do not meet the
        runtime's swappability criteria are materialized directly on the
        parameter's device and tracked as normal device-resident slots.
        """
        saved_groups = original_state_dict.get("param_groups", [])
        current_groups = self.optimizer.param_groups
        self._slots = {}
        saved_ids = []
        current_params = []
        for saved_group, current_group in zip(saved_groups, current_groups):
            saved_ids.extend(saved_group["params"])
            current_params.extend(current_group["params"])
        for saved_id, param in zip(saved_ids, current_params):
            key_to_tensor = removed.get(saved_id, {})
            if not key_to_tensor:
                continue
            state = self.optimizer.state[param]
            for key, saved_tensor in key_to_tensor.items():
                cpu_tensor = self._cast_swappable_tensor_to_cpu(param, saved_tensor)
                if self.runtime.packed_enabled and self.runtime.is_packable_template(param, self.config.min_numel):
                    if self.runtime.is_distributed_tensor(param):
                        logical_tensor = torch.zeros_like(
                            param,
                            memory_format=torch.preserve_format,
                        )
                        slot = self._make_slot(key, logical_tensor)
                        state[key] = logical_tensor
                        self.runtime.release_device_storage(slot)
                    else:
                        slot = self._make_slot(key, None, template=param)
                        state[key] = cpu_tensor
                        slot.tensor = cpu_tensor
                    slot.cpu_tensor = cpu_tensor
                    slot.state = "host"
                    self._slots[(id(param), key)] = slot
                    continue
                device_tensor = self.runtime.make_empty_device_tensor_like(param, cpu_tensor)
                slot = self._make_slot(key, device_tensor)
                if slot.swappable:
                    state[key] = device_tensor
                    slot.cpu_tensor = self.runtime.make_cpu_tensor(cpu_tensor)
                    slot.state = "host"
                    self._slots[(id(param), key)] = slot
                    self.runtime.release_device_storage(slot)
                else:
                    device_tensor = self._cast_state_tensor_like_torch(
                        param,
                        saved_tensor,
                        saved_id,
                        saved_groups,
                        key,
                    )
                    state[key] = device_tensor
                    self._slots[(id(param), key)] = self._make_slot(key, device_tensor)
        if self.runtime.packed_enabled:
            self.runtime.prepare_packed_host(self._ordered_slots())
            self.publish_packed_state()

    def _init_param_state(self, param: Any, grad: Any, group: Dict[str, Any]) -> None:
        """Initialize missing Adam state and swap slots for one parameter."""
        del grad
        state = self.optimizer.state[param]
        if not self.is_hyper_adamw and len(state) == 0:
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

    def _register_present_slots(self, param: Any, state: Dict[str, Any]) -> None:
        """Register configured state tensors that already exist in an optimizer state mapping."""
        for key in self._configured_state_keys():
            tensor = state.get(key)
            if tensor is None or (id(param), key) in self._slots:
                continue
            self._slots[(id(param), key)] = self._make_slot(key, tensor)

    def _build_slots(self, param: Any, state: Dict[str, Any]) -> List[SwapSlot]:
        """Return swap slots associated with the current parameter state."""
        slots = []
        for key in self._state_keys_for_param(param):
            tensor = state.get(key)
            if tensor is None:
                continue
            slot = self._slots.get((id(param), key))
            if slot is None:
                slot = self._make_slot(key, tensor)
                self._slots[(id(param), key)] = slot
            elif slot.tensor is not tensor and slot.cpu_tensor is not tensor:
                slot = self._make_slot(key, tensor)
                self._slots[(id(param), key)] = slot
            slots.append(slot)
        return slots

    def _make_slot(self, key: str, tensor: Any, template: Optional[Any] = None) -> SwapSlot:
        """Create a swap slot for a state tensor or a packed-state template."""
        metadata_tensor = tensor if tensor is not None else template
        if metadata_tensor is None:
            raise ValueError(f"Cannot build swap slot {key!r} without a tensor or template.")
        if tensor is None:
            swappable = self.runtime.is_packable_template(metadata_tensor, self.config.min_numel)
        else:
            swappable = self.runtime.is_swappable_tensor(tensor, self.config.min_numel)
        packed = bool(self.runtime.packed_enabled and swappable)
        slot = SwapSlot(
            name=key,
            tensor=tensor,
            cpu_tensor=None,
            swappable=swappable,
            state="device" if tensor is not None else "pending",
            packed=packed,
            logical_tensor=tensor if packed and self.runtime.is_distributed_tensor(tensor) else None,
        )
        self.runtime.populate_slot_metadata(slot, metadata_tensor)
        return slot

    def _ordered_slots(self) -> List[SwapSlot]:
        """Return slots in optimizer parameter and configured state-key order."""
        slots = []
        seen_slots = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                for key in self._configured_state_keys():
                    slot = self._slots.get((id(param), key))
                    if slot is not None and id(slot) not in seen_slots:
                        slots.append(slot)
                        seen_slots.add(id(slot))
        return slots

    def _state_keys_for_param(self, param: Any) -> Tuple[str, ...]:
        state = self.optimizer.state[param]
        keys = self._configured_state_keys()
        result = []
        for key in keys:
            if key in state:
                result.append(key)
            elif self.config.state_keys is not None:
                raise ValueError(f"Requested state key '{key}' is not present for parameter.")
        return tuple(result)

    @staticmethod
    def _slot_tensor(unit: UpdateUnit, key: str, fallback: Any) -> Any:
        """Return an active swap slot tensor, or the optimizer state fallback."""
        for slot in unit.slots:
            if slot.name == key and slot.swappable and slot.state == "device" and slot.tensor is not None:
                return slot.tensor
        return fallback

    def _configured_state_keys(self) -> Tuple[str, ...]:
        """Return Adam state keys selected for swap by the current config."""
        keys = self.config.state_keys or self._default_state_keys()
        result = []
        for key in keys:
            if key == "master_param":
                if self.config.state_keys is not None:
                    raise ValueError(f"Requested state key '{key}' is not available for {type(self.optimizer)!r}.")
                continue
            result.append(key)
        return tuple(result)

    def _cast_state_tensor_like_torch(
            self,
            param: Any,
            saved_tensor: Any,
            saved_id: int,
            saved_groups: List[Dict[str, Any]],
            key: str,
    ) -> Any:
        """Cast a loaded state tensor using PyTorch optimizer load semantics."""
        if not isinstance(saved_tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in optimizer state, got {type(saved_tensor)!r}.")
        process = getattr(torch.optim.Optimizer, "_process_value_according_to_param_policy", None)
        if process is not None:
            return process(param, saved_tensor, saved_id, saved_groups, key).detach().clone()
        if key == "step":
            return saved_tensor.detach().clone()
        if param.is_floating_point():
            return saved_tensor.detach().to(dtype=param.dtype, device=param.device).clone()
        return saved_tensor.detach().to(device=param.device).clone()

    def _cast_swappable_tensor_to_cpu(self, param: Any, saved_tensor: Any) -> Any:
        """Cast swappable state dtype like PyTorch while keeping values on CPU."""
        if not isinstance(saved_tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in optimizer state, got {type(saved_tensor)!r}.")
        if param.is_floating_point():
            return saved_tensor.detach().to(dtype=param.dtype, device="cpu")
        return saved_tensor.detach().to(device="cpu")

    @staticmethod
    def _default_state_keys() -> Tuple[str, ...]:
        return ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


class TorchNativeAdamAdapter(TorchAdamBaseAdapter):
    """Adapter for ``torch.optim.Adam``."""

    functional_name = "adam"

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        # AdamW inherits Adam in PyTorch.  Keep Adam subclasses supported, but
        # let AdamW select its dedicated adapter (which preserves fused=True).
        return (
            isinstance(optimizer, torch.optim.Adam)
            and not isinstance(optimizer, torch.optim.AdamW)
        )


class TorchNativeAdamWAdapter(TorchAdamBaseAdapter):
    """Adapter for ``torch.optim.AdamW``."""

    functional_name = "adamw"
    supports_fused = True

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        return isinstance(optimizer, torch.optim.AdamW)


class TorchHyperAdamWAdapter(TorchAdamBaseAdapter):
    """Adapter for hyper-parallel's fused AdamW."""

    functional_name = "adamw"
    supported_cls = (HyperAdamW,)
    is_hyper_adamw = True

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        return isinstance(optimizer, HyperAdamW)
