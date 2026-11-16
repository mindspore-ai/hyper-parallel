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
"""Observe reduce-scatter lifetime without retaining tensors or storages."""
import weakref
from typing import NamedTuple, Optional

from torch import Tensor, UntypedStorage, nn
from torch.multiprocessing.reductions import StorageWeakRef

from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState


class _PendingGradient(NamedTuple):
    """Weak references and byte counts for one in-flight gradient."""

    hsdp_param: weakref.ReferenceType
    param: weakref.ReferenceType
    accumulated_grad: Optional[weakref.ReferenceType]
    output: weakref.ReferenceType
    output_storage: StorageWeakRef
    output_nbytes: int
    input_storage: Optional[StorageWeakRef]
    input_nbytes: int


class MemTrackerWrapper(nn.Module):
    """Check the previous layer's communication after this layer's backward hook."""

    def __init__(self, module: nn.Module, enabled: bool = True) -> None:
        super().__init__()
        self.module = module
        self.enabled = enabled
        self._next_layer = None
        self._pending = []
        self.release_history = []
        self.checked_count = 0
        self.output_hook_count = 0

    def set_next_layer(self, next_layer: "MemTrackerWrapper") -> None:
        """Observe the layer whose backward precedes this layer's backward.

        Args:
            next_layer: Next layer in forward order.
        """
        self._next_layer = weakref.ref(next_layer)

    def forward(self, inputs: Tensor) -> Tensor:
        """Register observation hooks around the wrapped layer.

        Args:
            inputs: Input to the wrapped layer.

        Returns:
            The wrapped layer's output.
        """
        if self.enabled and inputs.requires_grad:
            inputs.register_hook(self._after_input_backward)
        outputs = self.module(inputs)
        if self.enabled and outputs.requires_grad:
            outputs.register_hook(self._after_output_backward)
        return outputs

    def _after_output_backward(self, grad: Tensor) -> Tensor:
        assert self.module.hsdp_scheduler.scheduler_state == FSDPSchedulerState.PRE_BACKWARD
        self.output_hook_count += 1
        return grad

    def _after_input_backward(self, grad: Tensor) -> Tensor:
        assert self.module.hsdp_scheduler.scheduler_state == FSDPSchedulerState.BACKWARD
        if self._next_layer is not None:
            next_layer = self._next_layer()
            assert next_layer is not None
            next_layer.check_pending()
        self._capture_pending()
        return grad

    def _capture_pending(self) -> None:
        """Record weak references and actual storage sizes before the next wait."""
        assert not self._pending
        for hsdp_param in self.module.hsdp_scheduler.hsdp_state.hsdp_params:
            param = hsdp_param.sharded_param
            output = hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output
            if output is None:
                continue
            source_grad = hsdp_param._grad  # pylint: disable=protected-access
            input_storage = source_grad.untyped_storage() if source_grad is not None else None
            output_storage = output.untyped_storage()
            self._pending.append(_PendingGradient(
                hsdp_param=weakref.ref(hsdp_param),
                param=weakref.ref(param),
                accumulated_grad=weakref.ref(param.grad) if param.grad is not None else None,
                output=weakref.ref(output),
                output_storage=StorageWeakRef(output_storage),
                output_nbytes=output_storage.nbytes(),
                input_storage=StorageWeakRef(input_storage) if input_storage is not None else None,
                input_nbytes=input_storage.nbytes() if input_storage is not None else 0,
            ))

    @staticmethod
    def _remaining_storage_bytes(reference: StorageWeakRef) -> int:
        if reference.expired():
            return 0
        # An input storage may survive through an autograd view after resize_(0).
        # Recreate only its Python wrapper to inspect size; no device buffer is allocated.
        storage = UntypedStorage._new_with_weak_ptr(reference.cdata)  # pylint: disable=protected-access
        return storage.nbytes() if storage is not None else 0

    def check_pending(self) -> None:
        """Record storage bytes released or adopted by each backward wait."""
        if not self._pending:
            return
        input_released, output_released, output_retained = 0, 0, 0
        seen_inputs, seen_outputs = set(), set()
        for pending in self._pending:
            hsdp_param, param = pending.hsdp_param(), pending.param()
            assert hsdp_param is not None and param is not None
            assert hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle is None
            assert hsdp_param._grad is None  # pylint: disable=protected-access
            if pending.input_storage is not None and pending.input_storage not in seen_inputs:
                input_released += pending.input_nbytes - self._remaining_storage_bytes(pending.input_storage)
                seen_inputs.add(pending.input_storage)
            assert pending.output_storage not in seen_outputs
            seen_outputs.add(pending.output_storage)
            if pending.accumulated_grad is None:
                assert not pending.output_storage.expired()
                if param.grad is None:
                    assert hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output is pending.output()
                else:
                    assert StorageWeakRef(param.grad._local_tensor.untyped_storage()) == pending.output_storage
                output_retained += pending.output_nbytes
            else:
                assert pending.accumulated_grad() is not None and param.grad is pending.accumulated_grad()
                assert hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output is None
                assert pending.output() is None and pending.output_storage.expired()
                output_released += pending.output_nbytes
                self.checked_count += 1
        self.release_history.append((input_released, output_released, output_retained))
        self._pending.clear()
