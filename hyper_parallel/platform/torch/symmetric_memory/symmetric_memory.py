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
from logging import getLogger
import torch
import torch.distributed as dist
from hyper_parallel.platform import get_platform

logger = getLogger(__name__)

_is_shmem_available = False

_manager = None
_ops = None
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'libaclshmem_torch.so')
if os.path.exists(file_path):
    torch.ops.load_library(file_path)
    _manager = torch.classes.SymmetricMemory.Manager()
    _ops = torch.classes.SymmetricMemory.Ops()
    _is_shmem_available = True


class TorchSymmetricMemoryHandler:
    """SymmetricMemory is used for one-sided communication."""
    _is_init = False
    comm_streams = []
    compute_streams = []

    @classmethod
    def _init_shmem(cls):
        """init platform"""
        if cls._is_init:
            return
        logger.info("start init torch symmetric memory")
        platform = get_platform()
        rank_id = platform.get_rank()
        world_size = platform.get_world_size()
        local_mem_size = os.getenv("SYMMETRIC_MEMORY_HEAP_SIZE")
        if local_mem_size is not None:
            local_mem_size = int(local_mem_size)
        else:
            local_mem_size = 1024 * 1024 * 1024
        ipports = "tcp://127.0.0.1:8662"
        logger.info("start init ach shmem: rank_id:%d, rank size:%d, heap size:%d, ipports:%s",
                    rank_id, world_size, local_mem_size, ipports)
        _manager.attr_init(rank_id, world_size, local_mem_size, ipports)
        cls.comm_streams = [torch.npu.Stream() for _ in range(min(world_size, 16))]
        cls.compute_streams = [torch.npu.Stream() for _ in range(world_size)]
        cls._is_init = True
        logger.info("init symmetric memory success!")

    @staticmethod
    def is_shmem_available():
        return _is_shmem_available

    @staticmethod
    def empty(shape, dtype):
        """create symmetric memory tensor"""
        if isinstance(shape, int):
            shape = [shape]
        elif isinstance(shape, tuple):
            shape = list(shape)
        if not TorchSymmetricMemoryHandler._is_init:
            TorchSymmetricMemoryHandler._init_shmem()
        return _manager.malloc(shape, dtype)

    @staticmethod
    def barrier():
        dist.barrier()

    @staticmethod
    def rendezvous(tensor, group):
        raise NotImplementedError("In CANN SHMEM v1.0.0, rendezvous is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "so this function is not implemented. ")

    @staticmethod
    def set_signal_pad_size(size: int) -> None:
        raise NotImplementedError("In CANN SHMEM v1.0.0, set_signal_pad_size is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "you can create symmetric signal memory by empty() "
                                  "so this function is not implemented. ")

    @staticmethod
    def get_signal_pad_size() -> int:
        raise NotImplementedError("In CANN SHMEM v1.0.0, get_signal_pad_size is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "you can create symmetric signal memory by empty() "
                                  "so this function is not implemented. ")

    @staticmethod
    def shmem_put(target, target_offset, src, src_offset, size, target_rank):
        """shmem_put operator: shmem_put(target, target_offset, src, src_offset, size, target_rank)"""
        world_size = dist.get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        _ops.put_mem(target, target_offset, src, src_offset, size, target_rank)

    @staticmethod
    def shmem_get(target, target_offset, src, src_offset, size, target_rank):
        """shmem_get operator: shmem_get(target, target_offset, src, src_offset, size, target_rank)"""
        world_size = dist.get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        _ops.get_mem(target, target_offset, src, src_offset, size, target_rank)

    @staticmethod
    def shmem_signal_op(signal, signal_offset, signal_value, signal_op, target_rank):
        """shmem_signal_op operator: shmem_signal_op(signal, signal_offset, signal_value, signal_op, target_rank)"""
        world_size = dist.get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        _ops.signal_op(signal, signal_offset, signal_value, signal_op, target_rank)

    @staticmethod
    def shmem_wait_for_signal(depend_tensor, signal, signal_offset, compare_value, compare_op):
        """
        shmem_wait_for_signal operator:
        shmem_wait_for_signal(depend_tensor, signal, signal_offset, compare_value, compare_op)
        """
        _ops.signal_wait_until(depend_tensor, signal, signal_offset, compare_value, compare_op)

    @staticmethod
    def shmem_put_with_signal(target, target_offset, src, src_offset,
                              size, signal, signal_offset, signal_value, signal_op, target_rank):
        """
        shmem_put_with_signal operator:
        shmem_put_with_signal(target, target_offset, src, src_offset,
                            size, signal, signal_offset, signal_value, signal_op, target_rank)
        """
        world_size = get_platform().get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        _ops.put_mem_signal(target, target_offset, src, src_offset,
                            size, signal, signal_offset, signal_value, signal_op, target_rank)

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
        def to_tensor(x, dtype=torch.int64):
            return torch.tensor([x], dtype=dtype, device='npu')
        rank_id = dist.get_rank()
        world_size = dist.get_world_size()
        size = input_tensor.numel()
        if size * world_size != output_tensor.numel():
            raise ValueError(f"All tensor must have same size, but in rank {world_size}, the size "
                             f"of input_tensor is {size}, the size of output_tensor is {output_tensor.numel()}")
        signal = _manager.malloc(1, torch.int32)
        torch.zero_(signal)
        dist.barrier()
        remain = rank_id
        now_pe = 0
        while remain:
            for i in range(min(16, remain)):
                target_pe = now_pe + i
                with torch.npu.stream(cls.comm_streams[i]):
                    _ops.put_mem_signal(output_tensor, to_tensor(size * world_size), input_tensor, to_tensor(0),
                                        to_tensor(size), signal, to_tensor(0), to_tensor(1, torch.int32), 1, target_pe)
            now_pe += 16
            remain -= min(16, remain)
        _ops.signal_wait_until(output_tensor, signal, to_tensor(0), to_tensor(rank_id, torch.int32), 0)
        _manager.free(signal)

    @classmethod
    def shmem_alltoall(cls, send_tensor_list, receive_tensor, receive_list):
        """
        alltoall operator: shmem_alltoall(send_tensor_list, receive_tensor, receive_list)

        Performs an all-to-all collective operation using symmetric memory (SHMEM).
        Each process sends distinct data blocks to all other processes and receives 
        corresponding blocks from all other processes.

        Parameters:
            send_tensor_list: List of tensors to send to each rank.
                            send_tensor_list[i] is the tensor sent to rank i.
            receive_tensor:   Output tensor that will contain all received data.
                            Its total size should be sum(receive_list).
            receive_list:     List specifying the size of data to receive from each rank.
                            receive_list[i] is the size (e.g., number of elements) 
                            of data to receive from rank i.
        """
        def to_tensor(x, dtype=torch.int64):
            return torch.tensor([x], dtype=dtype, device='npu')
        world_size = dist.get_world_size()
        receive_offsets = torch.zeros_like(receive_list)
        send_offsets = torch.zeros_like(receive_list)
        for i in range(1, world_size):
            receive_offsets[i] = receive_offsets[i - 1] + receive_list[i - 1]
        dist.all_to_all_single(send_offsets, receive_offsets)
        signal = _manager.malloc(1, torch.int32)
        torch.zero_(signal)
        dist.barrier()
        remain = world_size
        now_pe = 0
        while remain:
            for i in range(min(16, remain)):
                target_pe = now_pe + i
                with torch.npu.stream(cls.comm_streams[i]):
                    _ops.put_mem_signal(receive_tensor, send_offsets[target_pe], send_tensor_list[target_pe],
                                        to_tensor(0), to_tensor(send_tensor_list[target_pe].numel()),
                                        signal, to_tensor(0), to_tensor(1, torch.int32), 1, target_pe)
            now_pe += 16
            remain -= min(16, remain)
        _ops.signal_wait_until(receive_tensor, signal, to_tensor(0), to_tensor(world_size, torch.int32), 0)
        _manager.free(signal)

    @classmethod
    def fused_all_gather_matmul(cls, a, b, c, gather_out, signal, block_size=None):
        """
        fused_all_gather_matmul(a, b, c, gather_out, signal, block_size=None)
        Fused operator combining allgather and matmul operations.

        Computational flow:
            1. gather_out = allgather(a)    # Gather local tensor 'a' from all ranks
            2. c = ReduceScatter(gather_out @ b)  # Matrix multiplication followed by reduce-scatter

        Parameters:
            a: Local input tensor with shape (m_local, k).
            b: Weight matrix with shape (k, n).
            c: Output tensor with shape (m, n).
            gather_out: Output tensor containing gathered 'a' from all ranks, 
                        shape (m, k) where m = m_local * world_size.
            signal: Symmetric memory tensor with shape (world_size) and dtype int32.
            block_size: Optional block size for tiled computation.

        Note: 
            - This fusion reduces communication overhead by combining gather and matmul operations.
        """
        def to_tensor(value, dtype=torch.int64):
            return torch.tensor(value, dtype=dtype, device='npu')

        world_size = dist.get_world_size()
        rank_id = dist.get_rank()

        m, k = a.shape

        if block_size is None:
            block_size = max(1, m // min(world_size, 4))
        block_size = min(block_size, m)
        num_blocks = (m + block_size - 1) // block_size

        if m * world_size != gather_out.shape[0]:
            raise ValueError(f"gather_out shape mismatch: expected [{a.shape[0] * world_size}, {k}], "
                             f"got {gather_out.shape}")

        if gather_out.shape[0] != c.shape[0] or b.shape[1] != c.shape[1]:
            raise ValueError(f"Matmul output shape mismatch: expected [{gather_out.shape[0]}, {b.shape[1]}], "
                             f"got {c.shape}")

        int32_1 = torch.ones(1, dtype=torch.int32, device='npu')
        signal_offsets = torch.arange(0, world_size * num_blocks, dtype=torch.int64, device='npu')
        block_sizes = []
        for i in range(num_blocks):
            if i < num_blocks - 1:
                block_sizes.append(block_size)
            else:
                block_sizes.append(m - i * block_size)
        for block_idx in range(num_blocks):

            start_row = block_idx * block_size
            start_idx_tensor = to_tensor(start_row * k)
            block_local_size = block_sizes[block_idx] * k
            block_local_size_tensor = to_tensor(block_local_size)

            remain = world_size
            now_pe = 0
            dst_offset_tensor = to_tensor((rank_id * m + start_row) * k)
            while remain:
                for i in range(min(16, remain)):
                    target_pe = now_pe + i
                    with torch.npu.stream(cls.comm_streams[i]):
                        _ops.put_mem_signal(
                            gather_out, dst_offset_tensor,
                            a.view(-1), start_idx_tensor,
                            block_local_size_tensor, signal,
                            signal_offsets[rank_id * num_blocks + block_idx], int32_1,
                            0, target_pe
                        )
                now_pe += 16
                remain -= min(16, remain)
        for rank in range(world_size):
            with torch.npu.stream(cls.compute_streams[rank]):
                for block_idx in range(num_blocks):
                    _ops.signal_wait_until(gather_out, signal,
                                           signal_offsets[rank * num_blocks + block_idx],
                                           int32_1, 0
                                           )
                    start_row = rank * m + block_size * block_idx
                    end_row = start_row + block_sizes[block_idx]
                    c[start_row:end_row, :] = torch.matmul(gather_out[start_row:end_row, :], b)
        for stream in cls.comm_streams:
            stream.synchronize()
        for stream in cls.compute_streams:
            stream.synchronize()

        return gather_out, c

    @classmethod
    def fused_matmul_reduce_scatter(cls, x1, x2, symm_tensor, signal, reduce_op='sum'):
        """
        Fusion operator: Fuses Matmul and ReduceScatter operations.
        Computation formula: output = ReduceScatter(x1 @ x2)

        Parameters:
        x1: Left matrix with shape (m, k). 'm' must be an integer multiple of the number of devices (world_size).
        x2: Right matrix with shape (k, n).
        symm_tensor: Symmetric memory tensor with shape (m , n).
        signal: Symmetric memory tensor with shape (world_size) and dtype int32.
        reduce_op: Operator of scatter, only support 'sum' and 'avg'. Default value is 'sum'.

        output: Output matrix with shape (m / world_size, n).
        """
        def to_tensor(x, dtype=torch.int64):
            return torch.tensor([x], dtype=dtype, device='npu')

        world_size = dist.get_world_size()
        rank_id = dist.get_rank()

        m, k = x1.shape
        k2, n = x2.shape

        if k != k2:
            raise ValueError(f"Dimension k of x1 and x2 does not match: x1.k={k}, x2.k={k2}.")

        if x1.dtype != x2.dtype:
            raise ValueError(f"Matrix multiplication requires both tensors to have the same data type:"
                             f"x1.dtype={x1.dtype}, x2.dtype={x2.dtype}.")

        if m % world_size != 0:
            raise ValueError(f"The number of rows m={m} in x1 must be divisible by the world size (number of devices) "
                             f"world_size={world_size}.")

        if reduce_op not in ['sum', 'avg']:
            raise ValueError(f"The operator of scatter only supports sum and avg, but get {reduce_op}.")

        block_size = m // world_size

        size_tensor = to_tensor(block_size * n)
        int32_1 = torch.ones(1, dtype=torch.int32, device='npu')
        offsets = torch.arange(0, world_size, dtype=torch.int64, device='npu')
        dst_offset_tensor = to_tensor(block_size * n * rank_id)
        output = torch.matmul(x1[rank_id * block_size:rank_id * block_size + block_size, :], x2)
        for rank in range(1, world_size):
            with torch.npu.stream(cls.compute_streams[rank]):

                block_idx = (rank_id + rank) % world_size
                start_row = block_idx * block_size
                end_row = start_row + block_size
                x1_block = x1[start_row:end_row, :]

                block_result = torch.matmul(x1_block, x2)

                _ops.put_mem_signal(
                    symm_tensor, dst_offset_tensor,
                    block_result, offsets[0],
                    size_tensor, signal,
                    offsets[rank_id], int32_1,
                    0, block_idx
                )
        for rank in range(1, world_size):
            block_idx = (rank_id - rank) % world_size
            with torch.npu.stream(cls.comm_streams[block_idx]):
                _ops.signal_wait_until(symm_tensor, signal,
                                       offsets[block_idx], int32_1, 0
                                       )
                output.add_(symm_tensor[block_idx * block_size:block_idx * block_size + block_size, :])

        for stream in cls.compute_streams:
            stream.synchronize()
        for stream in cls.comm_streams:
            stream.synchronize()
        if reduce_op == 'sum':
            return output
        if reduce_op == 'avg':
            return output / world_size
