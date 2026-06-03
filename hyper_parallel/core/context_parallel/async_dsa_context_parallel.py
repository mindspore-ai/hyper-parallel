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
"""Async context parallel styles for DeepSeek Sparse Attention (DSA)."""
from typing import Any, Optional

from hyper_parallel.core.context_parallel.async_context_parallel import (
    _allgather_reconstruct,
    _launch_async_allgather_seq,
)
from hyper_parallel.core.context_parallel.context_parallel import _ensure_1d, _localize_foreign_dtensor
from hyper_parallel.core.context_parallel.dsa_context_parallel import (
    DSAIndexerContextParallel,
    DSAIndexerLossContextParallel,
    DSASparseAttentionContextParallel,
    _apply_sparse_attention_boundary,
    _is_tensor_or_dtensor,
    _to_sequence_replicate,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module


class _AsyncSequenceReplicateSlot:
    """Per-boundary async sequence-replicate state.

    Producer hooks append pre-launched all-gather handles here. Consumer
    pre-hooks pop and wait on them. If no producer hook ran, the consumer falls
    back to the synchronous sequence-gather path so async DSA styles remain
    compatible with the synchronous DSA module shape.
    """

    def __init__(self, device_mesh: DeviceMesh, seq_dim: int) -> None:
        self.device_mesh = device_mesh
        self.seq_dim = seq_dim
        self.world_size = device_mesh.mesh.numel()
        self.group = device_mesh.get_group() if self.world_size > 1 else None
        self._slots = {}
        self._bwd_slots = {}

    @staticmethod
    def _extract_tensor_output(output: Any) -> Any:
        if _is_tensor_or_dtensor(output):
            return output
        if isinstance(output, (tuple, list)) and len(output) == 1 and _is_tensor_or_dtensor(output[0]):
            return output[0]
        return None

    def _local_tensor(self, value: Any) -> Any:
        value = _localize_foreign_dtensor(value, self.device_mesh, self.seq_dim)
        return value.to_local() if isinstance(value, DTensor) else value

    def register_launch_hook(self, module: Optional[Module], slot_name: str) -> None:
        """Register a producer-side launch hook when a handoff module exists."""
        if module is None:
            return
        bwd_slot = self._bwd_slots.setdefault(slot_name, [])

        def _post_hook(hook_module, hook_inputs, output):
            del hook_module, hook_inputs
            tensor = self._extract_tensor_output(output)
            if tensor is not None:
                self.launch(slot_name, tensor)
            return output

        def _backward_pre_hook(hook_module, grad_output):
            del hook_module
            return self._producer_bwd_pre_hook(grad_output, bwd_slot)

        module.register_forward_hook(_post_hook)
        platform.register_full_backward_pre_hook(module, _backward_pre_hook)

    def _producer_bwd_pre_hook(self, grad_output: Any, bwd_slot: list) -> Any:
        """Wait deferred reduce-scatter before gradients cross the producer boundary."""
        if not bwd_slot:
            return grad_output
        work, out_perm, gather_dim = bwd_slot.pop()
        work.wait()
        d_local = _allgather_reconstruct(out_perm, gather_dim)
        if isinstance(grad_output, tuple):
            return (d_local,) + grad_output[1:]
        return (d_local,)

    def launch(self, slot_name: str, value: Any) -> None:
        """Launch all-gather for ``value`` and enqueue its handle."""
        if not _is_tensor_or_dtensor(value):
            return
        local = self._local_tensor(value)
        if not platform.is_tensor(local):
            return
        if self.world_size <= 1:
            self._slots.setdefault(slot_name, []).append((local, None, None))
            return
        work, out_perm = _launch_async_allgather_seq(local, self.group, self.world_size, self.seq_dim)
        self._slots.setdefault(slot_name, []).append((local, work, out_perm))

    def wait(self, slot_name: str, value: Any) -> Any:
        """Wait on a pre-launched gather, or fall back to consumer-local launch."""
        if not _is_tensor_or_dtensor(value):
            return value
        slot = self._slots.get(slot_name)
        if slot:
            local, work, out_perm = slot.pop(0)
            if work is None:
                return DTensor.from_local(local, self.device_mesh, (Replicate(),))
            gathered = platform.differentiable_async_allgather_wait(
                local,
                work,
                out_perm,
                self.group,
                self.world_size,
                self.seq_dim,
                self._bwd_slots.setdefault(slot_name, []),
            )
            return DTensor.from_local(gathered, self.device_mesh, (Replicate(),))
        return _to_sequence_replicate(value, self.device_mesh, self.seq_dim)


class AsyncDSAIndexerContextParallel(DSAIndexerContextParallel):
    """Async-first DSA indexer CP.

    When ``key_handoff`` is provided, its forward hook launches key-side
    all-gather and the indexer boundary pre-hook waits on it. Without a
    handoff, this style falls back to the synchronous key-side gather.
    """

    def apply(self, module: Module, device_mesh: DeviceMesh, *, key_handoff: Optional[Module] = None) -> Module:
        """Register async DSA indexer CP hooks on ``module`` and optional producer."""
        cp_mesh = _ensure_1d(device_mesh)
        async_state = _AsyncSequenceReplicateSlot(cp_mesh, self.seq_dim)
        async_state.register_launch_hook(key_handoff, "key")
        specs = self._build_specs(
            cp_mesh,
            key_fn=lambda value: async_state.wait("key", value),
        )
        return self._apply_with_specs(module, specs, cp_mesh)


class AsyncDSASparseAttentionContextParallel(DSASparseAttentionContextParallel):
    """Async-first DSA sparse-attention CP."""

    def apply(  # pylint: disable=arguments-differ
            self,
            module: Module,
            device_mesh: DeviceMesh,
            *,
            key_handoff: Optional[Module] = None,
            value_handoff: Optional[Module] = None,
            key_rope_handoff: Optional[Module] = None,
    ) -> Module:
        """Register async DSA sparse-attention CP hooks on ``module`` and producers."""
        cp_mesh = _ensure_1d(device_mesh)
        async_state = _AsyncSequenceReplicateSlot(cp_mesh, self.seq_dim)
        async_state.register_launch_hook(key_handoff, "key")
        async_state.register_launch_hook(value_handoff, "value")
        async_state.register_launch_hook(key_rope_handoff, "key_rope")
        return _apply_sparse_attention_boundary(self, module, device_mesh, async_state=async_state)


class AsyncDSAIndexerLossContextParallel(DSAIndexerLossContextParallel):
    """Async-first DSA indexer-loss CP."""

    def apply(  # pylint: disable=arguments-differ,too-many-arguments
            self,
            module: Module,
            device_mesh: DeviceMesh,
            *,
            key_handoff: Optional[Module] = None,
            key_indexer_handoff: Optional[Module] = None,
            key_rope_handoff: Optional[Module] = None,
    ) -> Module:
        """Register async DSA indexer-loss CP hooks on ``module`` and producers."""
        cp_mesh = _ensure_1d(device_mesh)
        async_state = _AsyncSequenceReplicateSlot(cp_mesh, self.seq_dim)
        async_state.register_launch_hook(key_handoff, "key")
        async_state.register_launch_hook(key_indexer_handoff, "key_indexer")
        async_state.register_launch_hook(key_rope_handoff, "key_rope")
        specs = self._build_loss_specs(
            cp_mesh,
            replicate_fn_map={
                "key": lambda value: async_state.wait("key", value),
                "key_indexer": lambda value: async_state.wait("key_indexer", value),
                "key_rope": lambda value: async_state.wait("key_rope", value),
            },
        )
        return self._apply_with_loss_specs(module, specs, self._get_local_idx(cp_mesh), cp_mesh)
