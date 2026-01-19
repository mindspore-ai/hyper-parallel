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
from typing import *
import threading
import torch
import warnings
import weakref
import hyper_parallel.utils.logger as logger
import functools

from collections import defaultdict
from torch.utils._pytree import tree_map


class Storage:
    def __init__(self):
        self.save_storage: Dict[Any, List[Any]] = defaultdict(list)
        self.swap_storage: Dict[Any, List[Any]] = defaultdict(list)

    def launch_load(self):
        def _async_load(x):
            if isinstance(x, SwapTensor):
                x.async_load()
            return x

        for stroage_list in self.swap_storage.values():
            for item in stroage_list:
                tree_map(_async_load, item)

    def wait_load(self):
        def _wait_load(x):
            if isinstance(x, SwapTensor):
                x.wait_load()
            return x

        for stroage_list in self.swap_storage.values():
            for item in stroage_list:
                tree_map(_wait_load, item)

    def wait_offload(self):
        def _wait_offload(x):
            if isinstance(x, SwapTensor):
                x.wait_offload()
            return x

        for stroage_list in self.swap_storage.values():
            for item in stroage_list:
                tree_map(_wait_offload, item)

    def launch_offload(self):
        def _async_offload(x):
            if isinstance(x, SwapTensor):
                x.async_offload()
            return x

        for stroage_list in self.swap_storage.values():
            for item in stroage_list:
                tree_map(_async_offload, item)


class SwapManager:
    _instance: Optional["SwapManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._groups: Dict[str, SwapGroup] = dict()
                    cls._instance._current_group_name: str = ""
                    cls._instance._counter_lock = threading.Lock()
                    cls._instance._layer_count = 0
                    cls._copy_stream = None
        return cls._instance

    def add_storage(self, group_name: str, storage: Storage) -> None:
        logger.info("[SwapManager.add_storage] add storage in group '%s'", group_name)
        if group_name not in self._groups:
            self._groups[group_name] = SwapGroup(group_name)
        self._groups[group_name].add(storage)

    def launch_offload(self, group_name: str, copy_stream=None):
        logger.info("[SwapManager.launch_offload] '%s'", group_name)
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        if copy_stream is None:
            copy_stream = self._get_copy_stream()
        group.launch_offload(copy_stream)

    def wait_offload(self, group_name: str):
        logger.info("[SwapManager.wait_offload] '%s'", group_name)
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        group.wait_offload()

    def launch_load(self, group_name: str, copy_stream=None):
        logger.info("[SwapManager.launch_load] '%s'", group_name)
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        if copy_stream is None:
            copy_stream = self._get_copy_stream()
        group.launch_load(copy_stream)

    def wait_load(self, group_name: str):
        logger.info("[SwapManager.wait_load] '%s'", group_name)
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        group.wait_load()

    def get_current_group_name(self):
        return self._current_group_name

    def set_current_group_name(self, group_name):
        self._current_group_name = group_name

    def set_forward_prefetch_layer(self, first_layer, second_layer):
        """
        Configure prefetching and offloading order between two consecutive layers.

        Usage:
            for i in range(len(model.layers) - 1):
                set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])

        Ensures idempotency: safe to call multiple times on the same layer pair.
        """

        def _ensure_group_name(module):
            """Assign a unique swap group name to the module if not already assigned."""
            if not hasattr(module, "_swap_group_name"):
                name = f"swap_group_{self._layer_count}"
                self._layer_count += 1
                module._swap_group_name = name
                module._swap_group_order = {"prev": None, "next": None}
            return module._swap_group_name
        first_name = _ensure_group_name(first_layer)
        second_name = _ensure_group_name(second_layer)

        if first_layer._swap_group_order["next"] is None:
            first_layer._swap_group_order["next"] = second_name
        if second_layer._swap_group_order["prev"] is None:
            second_layer._swap_group_order["prev"] = first_name

        def _forward_pre_hook(group_name, module, _):
            logger.info(f"enter {group_name} swap forward pre hook")
            if getattr(module, "_swap_state", None) == "pre_backward":
                logger.info(f"in pre backward, skip {group_name} forward pre hook")
                return
            SwapManager().set_current_group_name(group_name)

        def _forward_hook(group_name, module, args, output):
            """
            compute stream:       layer 0 | layer 1 | - | layer 2
            copy stream offlod:             layer 0 | - | layer 1
                                                    ├── launch offload layer 1
                                                    └── wait offload layer 0
            after layer finish, if it has next layer offload current mem
            after layer finish, if it has prev layer, waitoffload prev layer
            """
            if getattr(module, "_swap_state", None) == "pre_backward":
                logger.info(f"in pre backward, skip {group_name} swap forward hook")
                return
            next_name = module._swap_group_order.get('next', None)
            if next_name:
                SwapManager().launch_offload(group_name)
            prev_name = module._swap_group_order.get('prev', None)
            if prev_name:
                SwapManager().wait_offload(prev_name)
            logger.info(f"finished {group_name} forward hook")

        def _backward_pre_hook(group_name, module, grad_input):
            module._swap_state = "pre_backward"
            logger.info(f"enter {group_name} swap pre backward hook")
            prev_name = module._swap_group_order.get('prev', None)
            if prev_name:
                SwapManager().launch_load(prev_name)

            next_name = module._swap_group_order.get('next', None)
            if next_name:
                SwapManager().wait_load(group_name)

        def _backward_hook(group_name, module, grad_input, grad_output):
            module._swap_state = "backward"
            logger.info(f"finished {group_name} swap backward hook")

        def _register_hooks_once(module, group_name):
            hooks = [
                ("_swap_forward_pre_hook_handle",
                lambda h: module.register_forward_pre_hook(h, prepend=True),
                functools.partial(_forward_pre_hook, group_name)),

                ("_swap_forward_hook_handle",
                module.register_forward_hook,
                functools.partial(_forward_hook, group_name)),

                ("_swap_backward_pre_hook_handle",
                lambda h: module.register_full_backward_pre_hook(h, prepend=True),
                functools.partial(_backward_pre_hook, group_name)),

                ("_swap_backward_hook_handle",
                module.register_full_backward_hook,
                functools.partial(_backward_hook, group_name)),
            ]

            for attr_name, register_func, hook in hooks:
                if not hasattr(module, attr_name):
                    handle = register_func(hook)
                    setattr(module, attr_name, handle)
        # Register for both layers
        _register_hooks_once(first_layer, first_name)
        _register_hooks_once(second_layer, second_name)

    def _get_copy_stream(self):
        """Return a singleton NPU copy stream, created on first access."""
        if self._copy_stream is None:
            self._copy_stream = torch.npu.Stream()
        return self._copy_stream

class SwapGroup:
    def __init__(self, group_name: str):
        self.group_name = group_name
        self._storages = weakref.WeakSet()
        self._load_event = None
        self._offload_event = None

    def add(self, storage):
        self._storages.add(storage)

    def launch_offload(self, copy_stream):
        logger.info(f"[SwapGroup.launch_offload] Group '{self.group_name}'")

        compute_event = torch.npu.Event()
        compute_event.record(torch.npu.current_stream())
        self._offload_event = torch.npu.Event()
        with torch.no_grad(), torch.npu.stream(copy_stream):
            compute_event.wait(copy_stream)
            for storage in self._storages:
                storage.launch_offload()
            self._offload_event.record(copy_stream)

    def wait_offload(self):
        compute_stream = torch.npu.current_stream()
        logger.info(f"[SwapGroup.wait_offload] Group '{self.group_name}'")

        with torch.no_grad(), torch.npu.stream(compute_stream):
            self._offload_event.wait(compute_stream)
            self._offload_event = None
            for storage in self._storages:
                storage.wait_offload()

    def launch_load(self, copy_stream):
        logger.info(f"[SwapGroup.launch_load] Group '{self.group_name}'")

        compute_event = torch.npu.Event()
        compute_event.record(torch.npu.current_stream())
        self._load_event = torch.npu.Event()
        with torch.no_grad(), torch.npu.stream(copy_stream):
            compute_event.wait(copy_stream)
            for storage in self._storages:
                storage.launch_load()
            self._load_event.record(copy_stream)

    def wait_load(self):
        compute_stream = torch.npu.current_stream()
        logger.info(f"[SwapGroup.wait_load] Group '{self.group_name}'")

        with torch.no_grad(), torch.npu.stream(compute_stream):
            self._load_event.wait(compute_stream)
            self._load_event = None
            for storage in self._storages:
                storage.wait_load()


class SwapTensor:
    STATE_DEVICE = "device"
    STATE_HOST = "host"
    STATE_D2H = "d2h"
    STATE_H2D = "h2d"
    STATE_NON_TENSOR = "non_tensor"

    def __init__(self, val: Any) -> None:
        self.val = val

        if isinstance(val, torch.Tensor) and not val.is_cpu:
            self._state = self.STATE_DEVICE
            self.is_slice_tensor = val.untyped_storage().size() != val.numel()
            self.val_cpu: Optional[torch.Tensor] = torch.empty_like(
                val, device="cpu", pin_memory=True
            )
            self.storage_size = val.untyped_storage().size()
        else:
            self._state = self.STATE_NON_TENSOR
            self.val_cpu = None

    def get_val(self) -> Any:
        if self._state == self.STATE_NON_TENSOR:
            return self.val
        if self._state != self.STATE_DEVICE:
            raise RuntimeError(
                f"Cannot call get_val(): tensor is in '{self._state}' state. "
                f"Must be in 'device' state."
            )
        return self.val

    def async_load(self):
        if self._state == self.STATE_NON_TENSOR:
            return

        if self._state != self.STATE_HOST:
            warnings.warn(
                f"[SwapTensor.async_load] Invalid state: current={self._state}, "
                f"expected 'host'. Operation skipped."
            )
            return

        assert self.val_cpu is not None
        self.val.untyped_storage().resize_(self.storage_size)
        logger.debug("loading, copy stream is %s", torch.npu.current_stream())
        if self.is_slice_tensor:
            self.val.copy_(self.val_cpu, non_blocking=True)
        else:
            self.val.untyped_storage().copy_(self.val_cpu.untyped_storage(), non_blocking=True)
        self._state = self.STATE_H2D

    def wait_load(self):
        if self._state == self.STATE_NON_TENSOR:
            return

        if self._state == self.STATE_DEVICE:
            return  # already loaded
        if self._state != self.STATE_H2D:
            warnings.warn(
                f"[SwapTensor.wait_load] Called in invalid state: {self._state}. "
                f"Expected 'h2d'. Skipped."
            )
            return
        self._state = self.STATE_DEVICE

    def async_offload(self):
        if self._state == self.STATE_NON_TENSOR:
            return

        if self._state != self.STATE_DEVICE:
            warnings.warn(
                f"[SwapTensor.async_offload] Invalid state: current={self._state}, "
                f"expected 'device'. Operation skipped."
            )
            return

        logger.debug("offloading, copy stream is %s", torch.npu.current_stream())
        if self.is_slice_tensor:
            self.val_cpu.copy_(self.val, non_blocking=True)
        else:
            self.val_cpu.untyped_storage().copy_(self.val.untyped_storage(), non_blocking=True)
        self._state = self.STATE_D2H

    def wait_offload(self):
        if self._state == self.STATE_NON_TENSOR:
            return

        if self._state == self.STATE_HOST:
            return
        if self._state != self.STATE_D2H:
            warnings.warn(
                f"[SwapTensor.wait_offload] Called in invalid state: {self._state}. "
                f"Expected 'd2h'. Skipped."
            )
            return
        self.val.untyped_storage().resize_(0)
        self._state = self.STATE_HOST

    @property
    def state(self) -> str:
        return self._state

    def __repr__(self):
        if self._state == self.STATE_NON_TENSOR:
            return f"<SwapTensor state=non_tensor, val_type={type(self.val).__name__}>"
        else:
            return f"<SwapTensor state={self._state}, device_val={'exists' if self.val is not None else 'None'}>"
