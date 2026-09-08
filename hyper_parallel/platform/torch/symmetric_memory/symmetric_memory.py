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
"""Torch symmetric-memory operations backed by the shared lifecycle."""

from __future__ import annotations

import threading
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

logger = getLogger(__name__)

_is_shmem_available = False

_manager = None
_ops = None
_NATIVE_LOAD_LOCK = threading.Lock()


def _require_library() -> Path:
    """Locate the installed or source-build Torch symmetric-memory adapter."""
    module_path = Path(__file__).resolve()
    package_root = module_path.parents[3]
    relative_path = Path("core/symmetric_memory/lib/framework/torch/libaclshmem_torch.so")
    candidates = [package_root / relative_path]
    repository_root = module_path.parents[4]
    if (repository_root / "setup.py").is_file():
        candidates.insert(0, repository_root / "build/native/payload/hyper_parallel" / relative_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise ImportError(
        "[HP-NATIVE-PAYLOAD-MISSING] component=symmetric_memory framework=torch "
        f"searched={searched}. The current wheel does not include this optional component, or the local build failed; "
        "inspect the build log and run ./build.sh --shmem torch for source/PYTHONPATH development."
    )


def _load_native() -> None:
    """Load the optional native adapter exactly once on first SHMEM use."""
    import torch  # pylint: disable=C0415

    # pylint: disable=global-statement
    global _is_shmem_available, _manager, _ops
    if _manager is not None:
        return
    with _NATIVE_LOAD_LOCK:
        if _manager is not None:
            return
        file_path = str(_require_library())
        try:
            torch.ops.load_library(file_path)
            manager = torch.classes.SymmetricMemory.Manager()
            ops = torch.classes.SymmetricMemory.Ops()
        except (OSError, RuntimeError) as error:
            raise ImportError(
                "[HP-NATIVE-LOAD-FAILED] component=symmetric_memory framework=torch "
                f"library={file_path} error={error}. Check the Python/Torch/torch_npu/CANN "
                "version combination and build log."
            ) from error
        _manager = manager
        _ops = ops
        _is_shmem_available = True


def _get_manager() -> Any:
    """Return the native manager loaded by this binding module."""
    _load_native()
    return _manager


class TorchSymmetricMemoryHandler:
    """SymmetricMemory is used for one-sided communication."""

    _owner: ClassVar[Any | None] = None
    _owner_lock = threading.Lock()
    _stream_lock = threading.Lock()
    comm_streams: ClassVar[list] = []
    compute_streams: ClassVar[list] = []

    @classmethod
    def _init_shmem(cls) -> None:
        """Acquire the legacy API's owner from the shared lifecycle."""
        cls._get_owner()

    @classmethod
    def _get_owner(cls) -> Any:
        """Return the process-lifetime owner used by the legacy API."""
        if cls._owner is not None and not cls._owner.closed:
            return cls._owner
        with cls._owner_lock:
            if cls._owner is not None and not cls._owner.closed:
                return cls._owner
            # pylint: disable=C0415
            from .lifecycle import acquire_symmetric_memory

            logger.info("start init torch symmetric memory")
            cls._owner = acquire_symmetric_memory()
            logger.info("init symmetric memory success!")
            return cls._owner

    @classmethod
    def _init_streams(cls) -> None:
        """Create streams only for legacy collective helpers that use them."""
        import torch  # pylint: disable=C0415
        import torch.distributed as dist  # pylint: disable=C0415

        cls._get_owner()
        if cls.comm_streams:
            return
        with cls._stream_lock:
            if cls.comm_streams:
                return
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            cls.comm_streams = [torch.npu.Stream() for _ in range(min(world_size, 16))]
            cls.compute_streams = [torch.npu.Stream() for _ in range(world_size)]

    @classmethod
    def close(cls) -> None:
        """Release the legacy API owner without affecting other owners."""
        with cls._owner_lock:
            owner = cls._owner
            if owner is None:
                return
            cls._owner = None
            cls.comm_streams = []
            cls.compute_streams = []
            owner.close()

    @staticmethod
    def is_shmem_available() -> bool:
        """Return whether the symmetric-memory native adapter can be loaded."""
        try:
            _load_native()
        except ImportError:
            return False
        return True

    @staticmethod
    def empty(shape: Any, dtype: Any) -> Any:
        """Create a symmetric-memory tensor through the shared owner."""
        return TorchSymmetricMemoryHandler._get_owner().empty(shape, dtype)

    @staticmethod
    def barrier() -> None:
        """Synchronize all ranks via the shared owner."""
        TorchSymmetricMemoryHandler._get_owner().barrier()

    @staticmethod
    def rendezvous(tensor: Any, group: Any) -> None:
        """Allocate symmetric memory across ranks; not needed in CANN SHMEM v1.6.0."""
        raise NotImplementedError("In CANN SHMEM v1.6.0, rendezvous is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "so this function is not implemented. ")

    @staticmethod
    def set_signal_pad_size(size: int) -> None:
        """Set the signal pad size; not implemented for CANN SHMEM v1.6.0."""
        raise NotImplementedError("In CANN SHMEM v1.6.0, set_signal_pad_size is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "you can create symmetric signal memory by empty() "
                                  "so this function is not implemented. ")

    @staticmethod
    def get_signal_pad_size() -> int:
        """Return the signal pad size; not implemented for CANN SHMEM v1.6.0."""
        raise NotImplementedError("In CANN SHMEM v1.6.0, get_signal_pad_size is not needed, "
                                  "symmetric memory are allocated at init time by SYMMETRIC_MEMORY_HEAP_SIZE, "
                                  "you can create symmetric signal memory by empty() "
                                  "so this function is not implemented. ")

    @staticmethod
    def shmem_put(target: Any, target_offset: Any, src: Any, src_offset: Any,
                  size: Any, target_rank: int) -> None:
        """shmem_put operator: shmem_put(target, target_offset, src, src_offset, size, target_rank)"""
        import torch.distributed as dist  # pylint: disable=C0415

        TorchSymmetricMemoryHandler._get_owner()
        world_size = dist.get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        _ops.put_mem(target, target_offset, src, src_offset, size, target_rank)

    @staticmethod
    def shmem_get(target: Any, target_offset: Any, src: Any, src_offset: Any,
                  size: Any, target_rank: int) -> None:
        """shmem_get operator: shmem_get(target, target_offset, src, src_offset, size, target_rank)"""
        import torch.distributed as dist  # pylint: disable=C0415

        TorchSymmetricMemoryHandler._get_owner()
        world_size = dist.get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        _ops.get_mem(target, target_offset, src, src_offset, size, target_rank)

    @staticmethod
    def shmem_signal_op(signal: Any, signal_offset: Any, signal_value: Any,
                        signal_op: Any, target_rank: int) -> None:
        """shmem_signal_op operator: shmem_signal_op(signal, signal_offset, signal_value, signal_op, target_rank)"""
        import torch.distributed as dist  # pylint: disable=C0415

        TorchSymmetricMemoryHandler._get_owner()
        world_size = dist.get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        _ops.signal_op(signal, signal_offset, signal_value, signal_op, target_rank)

    @staticmethod
    def shmem_wait_for_signal(depend_tensor: Any, signal: Any, signal_offset: Any,
                              compare_value: Any, compare_op: Any) -> None:
        """
        shmem_wait_for_signal operator:
        shmem_wait_for_signal(depend_tensor, signal, signal_offset, compare_value, compare_op)
        """
        TorchSymmetricMemoryHandler._get_owner()
        _ops.signal_wait_until(depend_tensor, signal, signal_offset, compare_value, compare_op)

    @staticmethod
    def shmem_put_with_signal(target: Any, target_offset: Any, src: Any, src_offset: Any,
                              size: Any, signal: Any, signal_offset: Any, signal_value: Any,
                              signal_op: Any, target_rank: int) -> None:
        """
        shmem_put_with_signal operator:
        shmem_put_with_signal(target, target_offset, src, src_offset,
                            size, signal, signal_offset, signal_value, signal_op, target_rank)
        """
        import torch.distributed as dist  # pylint: disable=C0415

        TorchSymmetricMemoryHandler._get_owner()
        world_size = dist.get_world_size()
        if target_rank < 0 or target_rank >= world_size:
            raise ValueError(f"target_rank must be in range [0, {world_size - 1}], but get {target_rank}")
        _ops.put_mem_signal(target, target_offset, src, src_offset,
                            size, signal, signal_offset, signal_value, signal_op, target_rank)

    @classmethod
    def shmem_allgather(cls, output_tensor: Any, input_tensor: Any) -> None:
        """Gather equal-sized local inputs into a symmetric output tensor.

        Args:
            output_tensor: Symmetric output with ``world_size`` input segments.
            input_tensor: Local tensor contributed by this rank.
        """
        import torch  # pylint: disable=C0415
        import torch.distributed as dist  # pylint: disable=C0415

        def _to_tensor(value: Any, dtype: Any = torch.int64) -> Any:
            return torch.tensor([value], dtype=dtype, device='npu')
        owner = cls._get_owner()
        cls._init_streams()
        rank_id = dist.get_rank()
        world_size = dist.get_world_size()
        size = input_tensor.numel()
        if size * world_size != output_tensor.numel():
            raise ValueError(f"All tensor must have same size, but in rank {world_size}, the size "
                             f"of input_tensor is {size}, the size of output_tensor is {output_tensor.numel()}")
        signal = owner.empty(1, torch.int32)
        torch.zero_(signal)
        owner.barrier()
        remain = rank_id
        now_pe = 0
        while remain:
            for i in range(min(16, remain)):
                target_pe = now_pe + i
                with torch.npu.stream(cls.comm_streams[i]):
                    _ops.put_mem_signal(
                        output_tensor, _to_tensor(size * world_size), input_tensor, _to_tensor(0),
                        _to_tensor(size), signal, _to_tensor(0), _to_tensor(1, torch.int32), 1, target_pe
                    )
            now_pe += 16
            remain -= min(16, remain)
        _ops.signal_wait_until(output_tensor, signal, _to_tensor(0), _to_tensor(rank_id, torch.int32), 0)
        owner.free(signal)

    @classmethod
    def shmem_alltoall(cls, send_tensor_list: list[Any], receive_tensor: Any,
                       receive_list: Any) -> None:
        """Exchange variable-sized tensor segments through symmetric memory.

        Args:
            send_tensor_list: Per-rank tensors to send.
            receive_tensor: Symmetric destination containing received segments.
            receive_list: Per-rank receive sizes.
        """
        import torch  # pylint: disable=C0415
        import torch.distributed as dist  # pylint: disable=C0415

        def _to_tensor(value: Any, dtype: Any = torch.int64) -> Any:
            return torch.tensor([value], dtype=dtype, device='npu')
        owner = cls._get_owner()
        cls._init_streams()
        world_size = dist.get_world_size()
        receive_offsets = torch.zeros_like(receive_list)
        send_offsets = torch.zeros_like(receive_list)
        for i in range(1, world_size):
            receive_offsets[i] = receive_offsets[i - 1] + receive_list[i - 1]
        dist.all_to_all_single(send_offsets, receive_offsets)
        signal = owner.empty(1, torch.int32)
        torch.zero_(signal)
        owner.barrier()
        remain = world_size
        now_pe = 0
        while remain:
            for i in range(min(16, remain)):
                target_pe = now_pe + i
                with torch.npu.stream(cls.comm_streams[i]):
                    _ops.put_mem_signal(receive_tensor, send_offsets[target_pe], send_tensor_list[target_pe],
                                        _to_tensor(0), _to_tensor(send_tensor_list[target_pe].numel()),
                                        signal, _to_tensor(0), _to_tensor(1, torch.int32), 1, target_pe)
            now_pe += 16
            remain -= min(16, remain)
        _ops.signal_wait_until(receive_tensor, signal, _to_tensor(0), _to_tensor(world_size, torch.int32), 0)
        owner.free(signal)

    @classmethod
    def fused_all_gather_matmul(cls, a: Any, b: Any, c: Any, gather_out: Any,
                                signal: Any, block_size: int | None = None) -> tuple[Any, Any]:
        """Fuse symmetric all-gather of ``a`` with matrix multiplication.

        Args:
            a: Rank-local input matrix.
            b: Weight matrix.
            c: Output matrix populated in place.
            gather_out: Symmetric tensor receiving all rank-local inputs.
            signal: Symmetric signal tensor.
            block_size: Optional local rows per communication block.

        Returns:
            The populated ``gather_out`` and ``c`` tensors.
        """
        import torch  # pylint: disable=C0415
        import torch.distributed as dist  # pylint: disable=C0415

        def _to_tensor(value: Any, dtype: Any = torch.int64) -> Any:
            return torch.tensor(value, dtype=dtype, device='npu')

        cls._init_streams()
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
            start_idx_tensor = _to_tensor(start_row * k)
            block_local_size = block_sizes[block_idx] * k
            block_local_size_tensor = _to_tensor(block_local_size)

            remain = world_size
            now_pe = 0
            dst_offset_tensor = _to_tensor((rank_id * m + start_row) * k)
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
    def fused_matmul_reduce_scatter(cls, x1: Any, x2: Any, symm_tensor: Any,
                                    signal: Any, reduce_op: str = 'sum') -> Any:
        """Fuse matrix multiplication with symmetric reduce-scatter.

        Args:
            x1: Left matrix whose row count is divisible by world size.
            x2: Right matrix.
            symm_tensor: Symmetric buffer for partial outputs.
            signal: Symmetric signal tensor.
            reduce_op: Either ``"sum"`` or ``"avg"``.

        Returns:
            This rank's reduced output rows.
        """
        import torch  # pylint: disable=C0415
        import torch.distributed as dist  # pylint: disable=C0415

        def _to_tensor(value: Any, dtype: Any = torch.int64) -> Any:
            return torch.tensor([value], dtype=dtype, device='npu')

        cls._init_streams()
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

        size_tensor = _to_tensor(block_size * n)
        int32_1 = torch.ones(1, dtype=torch.int32, device='npu')
        offsets = torch.arange(0, world_size, dtype=torch.int64, device='npu')
        dst_offset_tensor = _to_tensor(block_size * n * rank_id)
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
