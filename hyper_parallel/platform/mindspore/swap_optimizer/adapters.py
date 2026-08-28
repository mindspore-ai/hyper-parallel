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
"""MindSpore Adam/AdamW swap optimizer adapters."""
# pylint: disable=protected-access

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Tuple

import mindspore as ms
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F

from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
from hyper_parallel.core.optimizer.swap_optimizer_base import (
    OptimizerSwapAdapter,
    SUPPORTED_STATE_KEYS,
    SwapSlot,
    UpdateUnit,
)


def _to_tuple(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return tuple(value)


class MindSporeAdamBaseAdapter(OptimizerSwapAdapter):
    """Common MindSpore optimizer adapter logic."""

    def __init__(self, optimizer: Any, config: Any, runtime: Any) -> None:
        super().__init__(optimizer, config, runtime)
        self._slots: Dict[Tuple[int, str], SwapSlot] = {}

    def validate(self) -> None:
        """Base validation."""
        if getattr(self.optimizer, "use_parallel", False):
            raise ValueError("MindSpore swap optimizer does not support parallel optimizer yet.")

    def iter_update_units(self, step_context: Dict[str, Any]) -> List[UpdateUnit]:
        """Return units collected in prepare_step."""
        return step_context["units"]

    def all_slots(self) -> Iterable[SwapSlot]:
        """Iterate known slots."""
        return tuple(self._slots.values())

    def initial_slots(self) -> Iterable[SwapSlot]:
        """Build optimizer state slots that can be offloaded before the first update."""
        return self._checkpoint_slots()

    def packed_layout_units(self) -> List[UpdateUnit]:
        """Return stable optimizer units used to build the packed host layout."""
        return []

    def checkpoint_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return checkpoint-safe optimizer state dict."""
        del args, kwargs
        state = self._state_dict()
        slot_by_name = self._checkpoint_slot_map()
        for name, slot in slot_by_name.items():
            if name not in state or not slot.swappable:
                continue
            if slot.cpu_tensor is None:
                if slot.state == "host":
                    raise RuntimeError(f"Swap slot {slot.name!r} is host-resident but has no CPU mirror.")
                slot.cpu_tensor = self.runtime.make_cpu_tensor(slot.tensor)
            state[name] = ms.Parameter(self.runtime.make_cpu_tensor(slot.cpu_tensor), name=name)
        return state

    def load_checkpoint_state_dict(
            self,
            state_dict: Dict[str, Any],
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """Load checkpoint-safe parameter dict."""
        del args, kwargs
        slot_by_name = self._checkpoint_slot_map(promote_checkpoint_swappable=True)
        remaining = dict(state_dict)
        for name, slot in slot_by_name.items():
            if name not in remaining or not slot.swappable:
                continue
            value = remaining.pop(name)
            tensor = getattr(value, "data", value)
            cpu_tensor = self.runtime.make_cpu_tensor(tensor)
            if slot.packed and slot.cpu_tensor is not None:
                self.runtime.copy_cpu_tensor(slot.cpu_tensor, cpu_tensor)
            else:
                slot.cpu_tensor = cpu_tensor
                self.runtime.release_device_storage(slot)
            if slot.packed:
                slot.tensor = slot.cpu_tensor
            slot.state = "host"
        if remaining:
            self._load_state_dict(remaining)
        self.publish_packed_state()

    def _state_dict(self) -> Dict[str, Any]:
        if not hasattr(self.optimizer, "state_dict"):
            raise RuntimeError(
                "The installed MindSpore version does not support optimizer.state_dict()."
            )
        return self.optimizer.state_dict()

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not hasattr(self.optimizer, "load_state_dict"):
            raise RuntimeError(
                "The installed MindSpore version does not support optimizer.load_state_dict()."
            )
        self.optimizer.load_state_dict(state_dict, strict=False)

    def _checkpoint_slots(self) -> Iterable[SwapSlot]:
        """Return slots for the optimizer's current checkpoint-visible state."""
        return tuple(self._slots.values())

    def _checkpoint_slot_map(self, *, promote_checkpoint_swappable: bool = False) -> Dict[str, SwapSlot]:
        """Build a name-to-slot map from current optimizer state Parameters."""
        checkpoint_slot_ids = {id(slot) for slot in self._checkpoint_slots()}
        slot_by_name: Dict[str, SwapSlot] = {}
        for (index, key), slot in self._slots.items():
            if id(slot) not in checkpoint_slot_ids:
                continue
            if (
                    promote_checkpoint_swappable
                    and not slot.swappable
                    and self._is_checkpoint_swappable_slot(slot)
            ):
                slot.swappable = True
                slot.storage_nbytes = self.runtime.storage_nbytes(slot.tensor)
            name = getattr(self._state_parameter(index, key), "name", None)
            if not name:
                continue
            previous = slot_by_name.get(name)
            if previous is not None and previous is not slot:
                raise ValueError(f"Duplicate optimizer state parameter name in swap slots: {name!r}.")
            slot_by_name[name] = slot
        return slot_by_name

    def _is_checkpoint_swappable_slot(self, slot: SwapSlot) -> bool:
        """Return whether a checkpoint slot meets the per-tensor swap requirements."""
        if slot.name not in SUPPORTED_STATE_KEYS:
            return False

        tensor = slot.tensor
        if isinstance(tensor, ms.Parameter):
            tensor = tensor.data
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()

        dtype_text = str(getattr(tensor, "dtype", "")).lower()
        if "float" not in dtype_text and "bfloat" not in dtype_text:
            return False
        if int(tensor.numel()) < int(self.config.min_numel):
            return False
        if not tensor.is_contiguous():
            return False
        try:
            storage = tensor.untyped_storage()
            if storage.size() != int(tensor.numel()) * int(tensor.itemsize):
                return False
        except (AttributeError, RuntimeError):
            return False
        return True

    def _make_slot(self, index: int, key: str, tensor: Any) -> SwapSlot:
        """Return the stable swap slot for an optimizer state tensor."""
        slot = self._slots.get((index, key))
        if slot is not None:
            return slot
        swappable = self.runtime.is_swappable_tensor(tensor, self.config.min_numel)
        is_packable = getattr(self.runtime, "is_packable_tensor", None)
        packed = bool(getattr(self.runtime, "packed_enabled", False)
                      and is_packable is not None
                      and is_packable(tensor, self.config.min_numel))
        swappable = swappable or packed
        slot = SwapSlot(
            name=key,
            tensor=tensor,
            cpu_tensor=None,
            storage_nbytes=self.runtime.storage_nbytes(tensor),
            swappable=swappable,
            state="device",
            packed=packed,
        )
        populate_metadata = getattr(self.runtime, "populate_slot_metadata", None)
        if populate_metadata is not None:
            populate_metadata(slot, tensor)
            slot.device = self._parameter_device(index)
        self._slots[(index, key)] = slot
        return slot

    def publish_packed_state(self) -> None:
        """Publish persistent packed CPU mirrors to optimizer state Parameters."""
        if not getattr(self.runtime, "packed_enabled", False):
            return
        for (index, key), slot in self._slots.items():
            if not slot.packed or slot.cpu_tensor is None:
                continue
            parameter = self._state_parameter(index, key)
            set_data = getattr(parameter, "set_data", None)
            if callable(set_data):
                set_data(slot.cpu_tensor)
                continue
            if hasattr(parameter, "data"):
                parameter.data = slot.cpu_tensor
                continue
            raise RuntimeError(
                f"MindSpore optimizer state Parameter for slot {key!r} cannot publish a packed CPU mirror."
            )

    def _state_parameter(self, index: int, key: str) -> Any:
        """Return the optimizer-owned Parameter for one logical state key."""
        raise NotImplementedError

    def _parameter_device(self, index: int) -> Any:
        """Return the target update device for an optimizer state slot."""
        params = getattr(self.optimizer, "_parameters", None)
        if params is None:
            params = getattr(self.optimizer, "fp32_params")
        param = _to_tuple(params)[index]
        if hasattr(param, "to_local"):
            param = param.to_local()
        return param.device

    @staticmethod
    def _slot_tensor(unit: UpdateUnit, key: str, fallback: Any) -> Any:
        """Return the active staging view for a logical state key."""
        for slot in unit.slots:
            if slot.name == key:
                return slot.tensor
        return fallback

    def _selected_keys(self, available: Tuple[str, ...]) -> Tuple[str, ...]:
        """Return configured state keys that are available for this optimizer."""
        keys = self.config.state_keys or available
        result = []
        for key in keys:
            if key == "master_param":
                if self.config.state_keys is not None:
                    raise ValueError(f"Requested state key '{key}' is not available for {type(self.optimizer)!r}.")
                continue
            if key in available:
                result.append(key)
            elif self.config.state_keys is not None:
                raise ValueError(f"Requested state key '{key}' is not available for {type(self.optimizer)!r}.")
        return tuple(result)

    def _validate_gradient_count(
            self,
            gradients: Any,
            params: Any,
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Normalize parameters and gradients, and require one gradient per parameter."""
        grad_tuple = _to_tuple(gradients)
        param_tuple = _to_tuple(params)
        if len(grad_tuple) != len(param_tuple):
            raise ValueError(
                f"MindSpore swap optimizer expected {len(param_tuple)} gradients, but got {len(grad_tuple)}."
            )
        return grad_tuple, param_tuple


class MindSporeNativeAdamAdapter(MindSporeAdamBaseAdapter):
    """Adapter for ``mindspore.nn.Adam``."""

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        return isinstance(optimizer, nn.Adam)

    def validate(self) -> None:
        super().validate()
        if getattr(self.config, "packed_swap", True):
            raise ValueError(
                "MindSpore nn.Adam does not support packed_swap=True. "
                "Set packed_swap=False to use per-tensor swap, or use "
                "mindformers AdamW for packed swap."
            )
        if getattr(self.optimizer, "use_lazy", False):
            raise ValueError("MindSpore Adam swap optimizer does not support use_lazy=True.")
        if getattr(self.optimizer, "use_offload", False):
            raise ValueError("MindSpore Adam swap optimizer does not support use_offload=True.")

    def prepare_step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Prepare native Adam step."""
        if len(args) != 1 or kwargs:
            raise ValueError("MindSpore swap optimizer only accepts gradients.")
        gradients = args[0]
        opt = self.optimizer
        grad_tuple, params = self._validate_gradient_count(gradients, opt._parameters)
        gradients = opt.decay_weight(grad_tuple)
        gradients = opt.gradients_centralization(gradients)
        gradients = opt.scale_grad(gradients)
        gradients = opt._grad_sparse_indices_deduplicate(gradients)
        lr = opt.get_lr()
        opt.assignadd(opt.global_step, opt.global_step_increase_tensor)
        beta1_power = opt.beta1_power * opt.beta1
        opt.beta1_power = beta1_power
        beta2_power = opt.beta2_power * opt.beta2
        opt.beta2_power = beta2_power

        grad_tuple = _to_tuple(gradients)
        units = []
        for index, (param, grad) in enumerate(zip(params, grad_tuple)):
            if grad is None:
                continue
            slots = self._build_slots(index)
            units.append(UpdateUnit(
                adapter_index=index,
                param=param,
                grad=grad,
                slots=slots,
            ))
        return {
            "units": units,
            "gradients": grad_tuple,
            "lr": lr,
            "beta1_power": beta1_power,
            "beta2_power": beta2_power,
        }

    def step_batch(self, batch: List[UpdateUnit], step_context: Dict[str, Any]) -> Tuple[Any, ...]:
        """Run native Adam for one batch."""
        opt = self.optimizer
        results = []
        for unit in batch:
            lr = self._index_lr(step_context["lr"], unit.adapter_index)
            if opt.use_amsgrad:
                result = opt.opt(
                    unit.param,
                    opt.moment1[unit.adapter_index],
                    opt.moment2[unit.adapter_index],
                    opt.vhat[unit.adapter_index],
                    step_context["beta1_power"],
                    step_context["beta2_power"],
                    lr,
                    opt.beta1,
                    opt.beta2,
                    opt.eps,
                    unit.grad,
                )
            else:
                result = opt._apply_adam(
                    (unit.param,),
                    step_context["beta1_power"],
                    step_context["beta2_power"],
                    (opt.moment1[unit.adapter_index],),
                    (opt.moment2[unit.adapter_index],),
                    (lr,) if opt.is_group_lr else lr,
                    (unit.grad,),
                )
            results.append(result)
        return tuple(results)

    def _build_slots(self, index: int) -> List[SwapSlot]:
        available = ["exp_avg", "exp_avg_sq"]
        if getattr(self.optimizer, "use_amsgrad", False) and hasattr(self.optimizer, "vhat"):
            available.append("max_exp_avg_sq")
        slots = []
        for key in self._selected_keys(tuple(available)):
            slots.append(self._make_slot(index, key, self._state_parameter(index, key)))
        return slots

    def _state_parameter(self, index: int, key: str) -> Any:
        if key == "exp_avg":
            return self.optimizer.moment1[index]
        if key == "exp_avg_sq":
            return self.optimizer.moment2[index]
        if key == "max_exp_avg_sq":
            return self.optimizer.vhat[index]
        raise ValueError(f"Unknown native Adam state key: {key!r}.")

    def _checkpoint_slots(self) -> Iterable[SwapSlot]:
        """Rebuild slots from native Adam state containers for checkpoint load."""
        slots = []
        for index in range(len(_to_tuple(self.optimizer._parameters))):
            slots.extend(self._build_slots(index))
        return tuple(slots)

    @staticmethod
    def _index_lr(lr: Any, index: int) -> Any:
        try:
            return lr[index]
        except (TypeError, IndexError):
            return lr


class MindSporeNativeAdamWAdapter(MindSporeAdamBaseAdapter):
    """Adapter for ``mindspore.nn.AdamWeightDecay``."""

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        adamw_cls = getattr(nn, "AdamW", None)
        return isinstance(optimizer, nn.AdamWeightDecay) or (
            adamw_cls is not None and isinstance(optimizer, adamw_cls)
        )

    def validate(self) -> None:
        super().validate()
        if getattr(self.config, "packed_swap", True):
            raise ValueError(
                "MindSpore nn.AdamWeightDecay does not support packed_swap=True. "
                "Set packed_swap=False to use per-tensor swap, or use "
                "mindformers AdamW for packed swap."
            )
        if not getattr(self.optimizer, "use_fused_opt", False):
            raise ValueError("MindSpore AdamWeightDecay swap optimizer only supports use_fused_opt=True.")

    def prepare_step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Prepare native AdamWeightDecay step."""
        if len(args) != 1 or kwargs:
            raise ValueError("MindSpore swap optimizer only accepts gradients.")
        gradients = args[0]
        opt = self.optimizer
        grad_tuple, params = self._validate_gradient_count(gradients, opt._parameters)
        weight_decay = opt.get_weight_decay()
        lr = opt.get_lr()
        opt.assignadd(opt.global_step, opt.global_step_increase_tensor)
        units = []
        for index, (param, grad) in enumerate(zip(params, grad_tuple)):
            if grad is None:
                continue
            slots = self._build_slots(index)
            units.append(UpdateUnit(
                adapter_index=index,
                param=param,
                grad=grad,
                slots=slots,
            ))
        return {"units": units, "gradients": grad_tuple, "lr": lr, "weight_decay": weight_decay}

    def step_batch(self, batch: List[UpdateUnit], step_context: Dict[str, Any]) -> Tuple[Any, ...]:
        """Run AdamWeightDecay fused primitive for one batch."""
        opt = self.optimizer
        results = []
        for unit in batch:
            if not opt.optim_filter[unit.adapter_index]:
                results.append(True)
                continue
            lr = self._indexed(step_context["lr"], unit.adapter_index, opt.is_group_lr)
            weight_decay = self._indexed(step_context["weight_decay"], unit.adapter_index, opt.is_group)
            decay = weight_decay if opt.decay_flags[unit.adapter_index] else 0.0
            grad = F.cast(unit.grad, F.dtype(unit.param))
            results.append(opt.fused_opt(
                unit.param,
                opt.moments1[unit.adapter_index],
                opt.moments2[unit.adapter_index],
                lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                decay,
                grad,
            ))
        return tuple(results)

    def _build_slots(self, index: int) -> List[SwapSlot]:
        slots = []
        for key in self._selected_keys(("exp_avg", "exp_avg_sq")):
            slots.append(self._make_slot(index, key, self._state_parameter(index, key)))
        return slots

    def _state_parameter(self, index: int, key: str) -> Any:
        if key == "exp_avg":
            return self.optimizer.moments1[index]
        if key == "exp_avg_sq":
            return self.optimizer.moments2[index]
        raise ValueError(f"Unknown native AdamWeightDecay state key: {key!r}.")

    def _checkpoint_slots(self) -> Iterable[SwapSlot]:
        """Rebuild slots from native AdamWeightDecay state containers for checkpoint load."""
        slots = []
        for index in range(len(_to_tuple(self.optimizer._parameters))):
            slots.extend(self._build_slots(index))
        return tuple(slots)

    @staticmethod
    def _indexed(value: Any, index: int, is_indexed: bool) -> Any:
        return value[index] if is_indexed else value


class MindFormersAdamWAdapter(MindSporeAdamBaseAdapter):
    """Adapter for ``mindformers.pynative.optimizer.adamw.AdamW``."""

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        optimizer_type = type(optimizer)
        return (
            optimizer_type.__name__ == "AdamW"
            and optimizer_type.__module__ == "mindformers.pynative.optimizer.adamw"
        )

    def validate(self) -> None:
        super().validate()
        if getattr(self.optimizer, "enable_cpu_offload", False):
            raise ValueError("mindformers AdamW enable_cpu_offload is not supported with swap optimizer.")

    def packed_layout_units(self) -> List[UpdateUnit]:
        """Return all MindFormers AdamW units in stable optimizer order."""
        return [
            UpdateUnit(
                adapter_index=index,
                param=param,
                grad=None,
                slots=self._build_slots(index),
            )
            for index, param in enumerate(_to_tuple(self.optimizer.fp32_params))
        ]

    def prepare_step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Prepare mindformers PyNative AdamW step."""
        if len(args) != 1 or kwargs:
            raise ValueError("MindSpore swap optimizer only accepts gradients.")
        gradients = args[0]
        opt = self.optimizer
        grad_tuple, params = self._validate_gradient_count(gradients, opt.fp32_params)
        weight_decay = opt.get_weight_decay()
        lr = opt.get_lr()
        opt._increase_global_step()

        lr = [float(x) for x in lr] if (opt.is_group and opt.is_group_lr) else float(lr)
        weight_decay = [float(x) for x in weight_decay] if opt.is_group else float(weight_decay)
        units = []
        for index, (param, grad) in enumerate(zip(params, grad_tuple)):
            if grad is None and not self.runtime.packed_enabled:
                continue
            slots = self._build_slots(index)
            units.append(UpdateUnit(
                adapter_index=index,
                param=param,
                grad=grad,
                slots=slots,
            ))
        return {"units": units, "gradients": grad_tuple, "lr": lr, "weight_decay": weight_decay}

    def step_batch(self, batch: List[UpdateUnit], step_context: Dict[str, Any]) -> Tuple[Any, ...]:
        """Run mindformers AdamW helpers for one batch."""
        with SkipDTensorDispatch():
            opt = self.optimizer
            module = importlib.import_module(type(opt).__module__)
            results = []
            is_lr_list = isinstance(step_context["lr"], list)
            is_wd_list = isinstance(step_context["weight_decay"], list)
            if getattr(opt, "enable_fused_opt", False):
                step = module.op_cast(opt.global_step, mstype.int64)
                for unit in batch:
                    if unit.grad is None:
                        continue
                    if not opt.optim_filter[unit.adapter_index]:
                        results.append(True)
                        continue
                    update_param = self._slot_tensor(unit, "master_param", unit.param)
                    learning_rate = (
                        step_context["lr"][unit.adapter_index]
                        if is_lr_list else step_context["lr"]
                    )
                    weight_decay = (
                        step_context["weight_decay"][unit.adapter_index]
                        if is_wd_list else step_context["weight_decay"]
                    )
                    results.append(module._run_fused_adamw_opt(
                        opt.fused_adamw_opt,
                        opt.amsgrad,
                        opt.maximize,
                        opt.beta1_value,
                        opt.beta2_value,
                        opt.eps_value,
                        step,
                        learning_rate,
                        weight_decay,
                        update_param,
                        unit.grad,
                        self._slot_tensor(unit, "exp_avg", opt.exp_avg[unit.adapter_index]),
                        self._slot_tensor(unit, "exp_avg_sq", opt.exp_avg_sq[unit.adapter_index]),
                        self._slot_tensor(unit, "max_exp_avg_sq", opt.max_exp_avg_sq[unit.adapter_index]),
                    ))
                self._sync_batch_master_params(batch)
                return tuple(results)

            bias_correction1 = 1.0 - opt.beta1 ** opt.global_step
            bias_correction2 = 1.0 - opt.beta2 ** opt.global_step
            for unit in batch:
                if unit.grad is None:
                    continue
                update_param = self._slot_tensor(unit, "master_param", unit.param)
                results.append(module._run_adamw_opt(
                    opt.beta1,
                    opt.beta2,
                    opt.eps,
                    step_context["lr"][unit.adapter_index] if is_lr_list else step_context["lr"],
                    step_context["weight_decay"][unit.adapter_index] if is_wd_list else step_context["weight_decay"],
                    update_param,
                    unit.grad,
                    self._slot_tensor(unit, "exp_avg", opt.exp_avg[unit.adapter_index]),
                    self._slot_tensor(unit, "exp_avg_sq", opt.exp_avg_sq[unit.adapter_index]),
                    opt.optim_filter[unit.adapter_index],
                    bias_correction1,
                    bias_correction2,
                    opt.one_minus_beta2,
                ))
            self._sync_batch_master_params(batch)
            return tuple(results)

    def finish_step(self, step_context: Dict[str, Any]) -> None:
        del step_context
        if not self.config.include_master_params:
            with SkipDTensorDispatch():
                self.optimizer._copy_main_params_to_model_params()

    def _build_slots(self, index: int) -> List[SwapSlot]:
        """Build swap slots for one MindFormers optimizer parameter."""
        opt = self.optimizer
        available = ["exp_avg", "exp_avg_sq"]
        max_slot = getattr(opt, "max_exp_avg_sq", None)
        if max_slot is not None and max_slot is not opt.exp_avg_sq:
            available.append("max_exp_avg_sq")
        slots = []
        selected_keys = []
        for key in (self.config.state_keys or tuple(available)):
            if key == "master_param":
                continue
            if key not in available:
                raise ValueError(f"Requested state key '{key}' is not available for {type(self.optimizer)!r}.")
            selected_keys.append(key)
        for key in tuple(selected_keys):
            slots.append(self._make_slot(index, key, self._state_parameter(index, key)))
        if self.config.include_master_params and hasattr(opt, "fp32_params"):
            fp32_param = opt.fp32_params[index]
            model_param = opt._parameters[index]
            if fp32_param is not model_param:
                slots.append(self._make_slot(index, "master_param", fp32_param))
        return slots

    def _state_parameter(self, index: int, key: str) -> Any:
        opt = self.optimizer
        if key == "exp_avg":
            return opt.exp_avg[index]
        if key == "exp_avg_sq":
            return opt.exp_avg_sq[index]
        if key == "max_exp_avg_sq":
            return opt.max_exp_avg_sq[index]
        if key == "master_param":
            return opt.fp32_params[index]
        raise ValueError(f"Unknown MindFormers AdamW state key: {key!r}.")

    def _checkpoint_slots(self) -> Iterable[SwapSlot]:
        """Rebuild slots from MindFormers AdamW state containers for checkpoint load."""
        slots = []
        for index in range(len(_to_tuple(self.optimizer.fp32_params))):
            slots.extend(self._build_slots(index))
        return tuple(slots)

    def _sync_batch_master_params(self, batch: List[UpdateUnit]) -> None:
        opt = self.optimizer
        if not self.config.include_master_params:
            return
        module = importlib.import_module(type(opt).__module__)
        for unit in batch:
            if unit.grad is None:
                continue
            if opt._is_low_precision_param[unit.adapter_index]:
                module.inplace_copy(
                    opt._parameters[unit.adapter_index],
                    module.op_cast(
                        self._slot_tensor(unit, "master_param", opt.fp32_params[unit.adapter_index]),
                        opt._parameters[unit.adapter_index].dtype,
                    ),
                )
