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
"""test data parallel"""
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.communication.management import init
import hyper_parallel.core.symmetric_memory as symm
from hyper_parallel.platform import get_platform

context.set_context(mode=context.PYNATIVE_MODE)
ms.set_seed(0)
init("hccl")
platform=get_platform()

def test_ms_symmetric_memory_put_mem():
    """
    Feature: symmetric memory put mem.
    Description: Test symmetric memory put mem api.
    Expectation: symmetric memory put mem api performs as expected.
    """
    rank = platform.get_rank()
    symm_tensor = symm.empty(16, dtype=ms.int64)

    target_offset = ms.Tensor([0], dtype=ms.int64)
    src_offset = ms.Tensor([0], dtype=ms.int64)
    size = ms.Tensor([4], dtype=ms.int64)
    if rank == 1:
        src = ms.ops.ones((16), dtype=ms.int64)
        symm.shmem_put(target=symm_tensor, target_offset=target_offset,
                       src=src, src_offset=src_offset, size=size, target_rank=0)
    ms.mint.distributed.barrier()
    if rank == 0:
        assert np.allclose(symm_tensor[:4].asnumpy(), np.ones((4), dtype=np.int64))

def test_ms_symmetric_memory_shmem_put_with_signal():
    """
    Feature: symmetric memory put mem with signal.
    Description: Test symmetric memory put mem with signal api.
    Expectation: symmetric memory put mem with signal api performs as expected.
    """
    rank = platform.get_rank()
    symm_tensor = symm.empty(4, dtype=ms.int64)
    symm_signal = symm.empty(2, dtype=ms.int32)
    symm_signal.fill_(0)
    src = ms.ops.ones((4), dtype=ms.int64)
    target_offset = ms.Tensor([0], dtype=ms.int64)
    src_offset = ms.Tensor([0], dtype=ms.int64)
    signal_offset = ms.Tensor([0], dtype=ms.int64)
    signal_value = ms.Tensor([1], dtype=ms.int32)
    size = ms.Tensor([4], dtype=ms.int64)
    ms.mint.distributed.barrier()
    if rank == 1:
        symm.shmem_put_with_signal(target=symm_tensor, target_offset=target_offset, src=src, src_offset=src_offset,
                                   size=size, signal=symm_signal, signal_offset=signal_offset,
                                   signal_value=signal_value, signal_op=0, target_rank=0)
    if rank == 0:
        symm.shmem_wait_for_signal(
            symm_tensor, symm_signal, signal_offset, signal_value, 0
        )
        assert np.allclose(symm_tensor.asnumpy(), np.ones((4), dtype=np.int64))
        assert np.allclose(symm_signal.asnumpy(), np.array([1, 0]))

def test_ms_symmetric_memory_shmem_allgather():
    """
    Feature: symmetric memory allgather.
    Description: Test symmetric memory allgather api.
    Expectation: symmetric memory allgather api performs as expected.
    """
    rank_id = platform.get_rank()
    world_size = platform.get_world_size()
    size = 4
    input_tensor = ms.ops.ones((size,), dtype=ms.int64) * (rank_id + 1)
    output_tensor = symm.empty(size * world_size, dtype=ms.int64)
    ms.mint.distributed.barrier()
    symm.shmem_allgather(output_tensor, input_tensor)
    assert output_tensor.numel() == size * world_size
    expected = np.concatenate([np.ones((size,), dtype=np.int64) * (i + 1) for i in range(world_size)])
    assert np.allclose(output_tensor.asnumpy(), expected)

def test_ms_symmetric_memory_shmem_wait_for_signal():
    """test signal op"""
    rank = platform.get_rank()
    signal = symm.empty(6, dtype=ms.int32)
    signal.fill_(0)
    ms.mint.distributed.barrier()
    if rank == 0:
        symm.shmem_signal_op(signal, ms.Tensor([0], dtype=ms.int64), ms.Tensor([1], dtype=ms.int32), 0, 1)
        symm.shmem_signal_op(signal, ms.Tensor([1], dtype=ms.int64), ms.Tensor([1], dtype=ms.int32), 1, 1)
        symm.shmem_signal_op(signal, ms.Tensor([2], dtype=ms.int64), ms.Tensor([-1], dtype=ms.int32), 0, 1)
        symm.shmem_signal_op(signal, ms.Tensor([3], dtype=ms.int64), ms.Tensor([-1], dtype=ms.int32), 1, 1)
        symm.shmem_signal_op(signal, ms.Tensor([4], dtype=ms.int64), ms.Tensor([2], dtype=ms.int32), 0, 1)
        symm.shmem_signal_op(signal, ms.Tensor([5], dtype=ms.int64), ms.Tensor([2], dtype=ms.int32), 1, 1)
    else:
        symm.shmem_wait_for_signal(signal, signal, ms.Tensor([0], dtype=ms.int64), ms.Tensor([1], dtype=ms.int32), 0)
        symm.shmem_wait_for_signal(signal, signal, ms.Tensor([1], dtype=ms.int64), ms.Tensor([1], dtype=ms.int32), 0)
        symm.shmem_wait_for_signal(signal, signal, ms.Tensor([2], dtype=ms.int64), ms.Tensor([0], dtype=ms.int32), 2)
        symm.shmem_wait_for_signal(signal, signal, ms.Tensor([3], dtype=ms.int64), ms.Tensor([0], dtype=ms.int32), 2)
        symm.shmem_wait_for_signal(signal, signal, ms.Tensor([4], dtype=ms.int64), ms.Tensor([1], dtype=ms.int32), 1)
        symm.shmem_wait_for_signal(signal, signal, ms.Tensor([5], dtype=ms.int64), ms.Tensor([1], dtype=ms.int32), 1)
    if rank == 1:
        assert np.allclose(signal.asnumpy(), np.array([1, 1, -1, -1, 2, 2]))
