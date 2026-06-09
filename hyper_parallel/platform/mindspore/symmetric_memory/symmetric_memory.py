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
"""Symmetric Memory"""
import os
import mindspore as ms
from mindspore.runtime import Stream, StreamCtx, Event
from hyper_parallel.platform import get_platform
from . import aclshmem_ms

_is_shmem_available = False
current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(current_dir, 'aclshmem_ms/aclshmem_ms.so')
if os.path.exists(_lib_path):
    _is_shmem_available = True


class MSSymmetricMemoryHandler:
    """SymmetricMemory is used for one-sided communication."""

    _is_init = False
    _mem_pool = None
    streamlist = []

    @classmethod
    def _init_shmem(cls):
        """init platform"""
        if cls._is_init:
            return
        allocator = ms.runtime.PluggableAllocator(_lib_path, "Alloc", "Free")
        cls._mem_pool = ms.runtime.MemPool(allocator)
        cls.streamlist = [Stream() for _ in range(get_platform().get_world_size())]
        cls._is_init = True

    @staticmethod
    def is_shmem_available():
        """Check whether the symmetric memory shared library is available on this host."""
        return _is_shmem_available

    @staticmethod
    def empty(size, dtype):
        """Allocate an uninitialized symmetric memory tensor of the given size and dtype."""
        if not MSSymmetricMemoryHandler._is_init:
            MSSymmetricMemoryHandler._init_shmem()
        with ms.runtime.use_mem_pool(MSSymmetricMemoryHandler._mem_pool):
            return ms.mint.empty(size, dtype=dtype)

    @staticmethod
    def rendezvous(tensor, group):
        """Not implemented: symmetric memory is allocated at init time in CANN SHMEM v1.0.0."""
        raise NotImplementedError("In CANN SHMEM v1.0.0, rendezvous is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "so this function is not implemented. ")

    @staticmethod
    def set_signal_pad_size(size: int) -> None:
        """Not implemented: signal padding is not needed in CANN SHMEM v1.0.0."""
        raise NotImplementedError("In CANN SHMEM v1.0.0, set_signal_pad_size is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "you can create symmetric signal memory by empty() "
                                  "so this function is not implemented. ")

    @staticmethod
    def get_signal_pad_size() -> int:
        """Not implemented: signal padding is not needed in CANN SHMEM v1.0.0."""
        raise NotImplementedError("In CANN SHMEM v1.0.0, get_signal_pad_size is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "you can create symmetric signal memory by empty() "
                                  "so this function is not implemented. ")

    @staticmethod
    def shmem_put(target, target_offset, src, src_offset, size, target_rank):
        """shmem_put operator: shmem_put(target, target_offset, src, src_offset, size, target_rank)"""
        world_size = get_platform().get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        aclshmem_ms.put_mem(target, target_offset, src, src_offset, size, target_rank)

    @staticmethod
    def shmem_get(target, target_offset, src, src_offset, size, target_rank):
        """shmem_get operator: shmem_get(target, target_offset, src, src_offset, size, target_rank)"""
        world_size = get_platform().get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        aclshmem_ms.get_mem(target, target_offset, src, src_offset, size, target_rank)

    @staticmethod
    def shmem_signal_op(signal, signal_offset, signal_value, signal_op, target_rank):
        """shmem_signal_op operator: shmem_signal_op(signal, signal_offset, signal_value, signal_op, target_rank)"""
        world_size = get_platform().get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        aclshmem_ms.signal_op(signal, signal_offset, signal_value, signal_op, target_rank)

    @staticmethod
    def shmem_wait_for_signal(depend_tensor, signal, signal_offset, compare_value, compare_op):
        """Block until the symmetric memory signal matches the expected value and comparison operation."""
        aclshmem_ms.signal_wait_until(depend_tensor, signal, signal_offset, compare_value, compare_op)

    @staticmethod
    def shmem_put_with_signal(target, target_offset, src, src_offset,
                              size, signal, signal_offset, signal_value, signal_op, target_rank):
        """shmem_put_with_signal operator"""
        world_size = get_platform().get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        aclshmem_ms.put_mem_signal(target, target_offset, src, src_offset,
                                   size, signal, signal_offset, signal_value, signal_op, target_rank)

    @staticmethod
    def barrier():
        """Synchronize all ranks with a distributed barrier."""
        ms.mint.distributed.barrier()

    @classmethod
    def shmem_allgather(cls, output_tensor, input_tensor):
        """
        allgather operator: shmem_allgather(output_tensor, input_tensor)

        Performs an allgather collective operation using symmetric memory (SHMEM).
        Each process/rank contributes its local 'input_tensor', and upon completion,
        all processes receive the concatenated data from all ranks in 'output_tensor'.

        Parameters:
            output_tensor: Output tensor that will contain the gathered data from all ranks.
                        Its size should be (world_size * local_input_size).
            input_tensor:  Local input tensor contributed by this rank.
        """
        rank_id= get_platform().get_rank()
        world_size = get_platform().get_world_size()
        size= input_tensor.numel()
        if size * world_size != output_tensor.numel():
            raise ValueError(f"All tensor must have same size, but in rank {world_size}, the size "
                             f"of input_tensor is {size}, the size of output_tensor is {output_tensor.numel()}")
        signal = cls.empty(1, dtype=ms.int32)
        signal.fill_(0)
        end_event_list=[]
        event_signal = Event()
        event_signal.record()
        for i in range(world_size):
            with StreamCtx(cls.streamlist[i]):
                event_signal.wait()
                cls.shmem_put_with_signal(output_tensor, ms.Tensor(rank_id * size, dtype=ms.int64), input_tensor,
                                      ms.Tensor(0, dtype=ms.int64),ms.Tensor(size,dtype=ms.int64),
                                      signal, ms.Tensor(0, dtype=ms.int64), ms.Tensor(1, dtype=ms.int32), 1, i)
                end_event = Event()
                end_event.record()
                end_event_list.append(end_event)
        for event in end_event_list:
            event.wait()
        cls.shmem_wait_for_signal(output_tensor, signal, ms.Tensor(0, dtype=ms.int64),
                                  ms.Tensor(world_size, dtype=ms.int32), 0)

    @classmethod
    def shmem_alltoall(cls, send_tensor_list, receive_tensor, receive_list):
        """Not yet implemented: symmetric memory all-to-all collective."""
        raise NotImplementedError("Mindspore SymmetricMemory will implement shmem_alltoall later")

    @classmethod
    def fused_all_gather_matmul(cls, a, b, c, gather_out, signal, block_size=None):
        """Not yet implemented: fused all-gather followed by matrix multiplication."""
        raise NotImplementedError("Mindspore SymmetricMemory will implement fused_all_gather_matmul later")

    @classmethod
    def fused_matmul_reduce_scatter(cls, x1, x2, symm_tensor, signal, reduce_op):
        """Not yet implemented: fused matrix multiplication followed by reduce-scatter."""
        raise NotImplementedError("Mindspore SymmetricMemory will implement fused_matmul_reduce_scatter later")
