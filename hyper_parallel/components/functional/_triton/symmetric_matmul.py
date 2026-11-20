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
"""Triangular Cube GEMM and rounded symmetric stores for Ascend NPUs."""

from functools import lru_cache
from typing import Optional

import torch
import torch_npu
import triton
import triton.language as tl


_MIRROR_BLOCK = 64


@triton.jit
def _symmetric_matmul_kernel(
    a, b, output,
    n: tl.constexpr, k: tl.constexpr, batch_count: tl.constexpr,
    stride_ab: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bb: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    cores: tl.constexpr,
):
    """Write the lower triangular GEMM tiles into the destination."""
    tiles: tl.constexpr = triton.cdiv(n, block_n)
    strips: tl.constexpr = block_n // block_m
    per_batch: tl.constexpr = tiles * (tiles + 1) // 2
    for task in range(tl.program_id(0), batch_count * per_batch * strips, cores):
        owner = task // strips
        batch = owner // per_batch
        local = owner % per_batch
        # Integer decoding keeps tile ownership off the Vector sqrt/workspace path.
        if tiles == 1:
            row = 0
        else:
            low = 0
            high = tiles
            while low + 1 < high:
                middle = (low + high) // 2
                if middle * (middle + 1) // 2 <= local:
                    low = middle
                else:
                    high = middle
            row = low
        column = local - row * (row + 1) // 2
        rows = row * block_n + task % strips * block_m + tl.arange(0, block_m)
        columns = column * block_n + tl.arange(0, block_n)
        reductions = tl.arange(0, block_k)
        batch_index = batch.to(tl.int64)
        left_base = a + batch_index * stride_ab
        right_base = b + batch_index * stride_bb
        accumulator = tl.zeros((block_m, block_n), tl.float32)
        for start in range(triton.cdiv(k, block_k)):
            ks = start * block_k + reductions
            left = tl.load(
                left_base + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=(rows[:, None] < n) & (ks[None, :] < k), other=0,
            )
            right = tl.load(
                right_base + ks[:, None] * stride_bk + columns[None, :] * stride_bn,
                mask=(ks[:, None] < k) & (columns[None, :] < n), other=0,
            )
            accumulator = tl.dot(left, right, accumulator)
        rounded = accumulator.to(output.dtype.element_ty)
        valid = (rows[:, None] < n) & (columns[None, :] < n)
        destination = output + batch_index * n * n
        tl.store(destination + rows[:, None] * n + columns[None, :], rounded, mask=valid)


@triton.jit
def _mirror_lower_kernel(output, n: tl.constexpr, batch_count: tl.constexpr, cores: tl.constexpr, block: tl.constexpr):
    """Fill the upper triangle from the rounded lower-triangular values."""
    tiles: tl.constexpr = triton.cdiv(n, block)
    per_batch: tl.constexpr = tiles * (tiles + 1) // 2
    for task in range(tl.program_id(0), batch_count * per_batch, cores):
        batch = task // per_batch
        local = task % per_batch
        if tiles == 1:
            row = 0
        else:
            low = 0
            high = tiles
            while low + 1 < high:
                middle = (low + high) // 2
                if middle * (middle + 1) // 2 <= local:
                    low = middle
                else:
                    high = middle
            row = low
        column = local - row * (row + 1) // 2
        rows = row * block + tl.arange(0, block)
        columns = column * block + tl.arange(0, block)
        destination = output + batch.to(tl.int64) * n * n
        valid = (rows[:, None] < n) & (columns[None, :] < n)
        values = tl.load(destination + rows[:, None] * n + columns[None, :], mask=valid, other=0)
        if row == column:
            # A select keeps the DMA mask rectangular and gives each diagonal tile one writer.
            symmetric = tl.where(rows[:, None] >= columns[None, :], values, tl.trans(values))
            tl.store(destination + rows[:, None] * n + columns[None, :], symmetric, mask=valid)
        else:
            tl.store(destination + columns[:, None] * n + rows[None, :], tl.trans(values),
                     mask=(columns[:, None] < n) & (rows[None, :] < n))


@lru_cache(maxsize=None)
def _core_counts(device: torch.device) -> tuple[int, int]:
    """Reuse static device properties without retaining tensors or streams."""
    properties = torch_npu.npu.get_device_properties(device)
    return properties.cube_core_num, properties.vector_core_num


def _static_block_shape(n: int, k: int, batch: int) -> tuple[int, int, int]:
    """Choose the Triton tiling for shapes without a compatible CANN plan."""
    if n <= 64:
        return 64, 64, 64
    if n <= 128:
        return 64, 128, 128
    if batch > 1 and n <= 1024 and k >= 2048:
        return 128, 128, 512
    return 128, 256, 256


def _cann_block_shape(
    device: torch.device,
    n: int,
    k: int,
    dtypes: tuple[torch.dtype, torch.dtype],
    transposed: tuple[bool, bool],
) -> Optional[tuple[int, int, int]]:
    """Query the official Ascend C matmul tiler through its Python binding."""
    try:
        # PyAsc is an optional host-side planner.
        from asc import adv  # pylint: disable=import-outside-toplevel
        from asc.lib import host  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as error:
        if error.name != "asc":
            raise
        return None

    platform = host.PlatformAscendCManager.get_instance(torch_npu.npu.get_device_name(device))
    if platform is None:
        return None
    tiler = host.MatmulApiTiling(platform)
    dtype_to_cann = {
        torch.float16: host.DataType.DT_FLOAT16,
        torch.bfloat16: host.DataType.DT_BF16,
        torch.float32: host.DataType.DT_FLOAT,
    }
    gm, nd = host.TPosition.GM, host.CubeFormat.ND
    input_dtype, output_dtype = (dtype_to_cann[dtype] for dtype in dtypes)
    status = (
        tiler.set_a_type(gm, nd, input_dtype, transposed[0]),
        tiler.set_b_type(gm, nd, input_dtype, transposed[1]),
        tiler.set_c_type(gm, nd, output_dtype),
        tiler.set_org_shape(n, n, k),
        tiler.set_shape(n, n, k),
        tiler.enable_bias(False),
        tiler.set_buffer_space(-1, -1, -1),
    )
    if any(status):
        return None
    plan = adv.TCubeTiling()
    if tiler.get_tiling(plan) == -1:
        return None
    return plan.base_m, plan.base_n, plan.base_k


def _cube_tasks(n: int, batch: int, block_m: int, block_n: int) -> int:
    """Count triangular work items with at least one valid output row."""
    full_rows, tail = divmod(n, block_n)
    full_tiles = full_rows * (full_rows + 1) // 2
    return batch * (full_tiles * (block_n // block_m) + (full_rows + 1) * triton.cdiv(tail, block_m))


@lru_cache(maxsize=128)
def _block_shape(
    device: torch.device,
    shape: tuple[int, int, int],
    dtypes: tuple[torch.dtype, torch.dtype],
    transposed: tuple[bool, bool],
) -> tuple[int, int, int]:
    """Use the CANN plan when it fits the triangular Triton kernel."""
    n, k, batch = shape
    static = _static_block_shape(n, k, batch)
    plan = _cann_block_shape(device, n, k, dtypes, transposed)
    if plan is None or any(size < 16 or size & (size - 1) for size in plan):
        return static
    block_m, block_n, base_k = plan
    # CANN baseK describes a Cube base block. A Triton dot can cover several of
    # those blocks per K iteration. Keep the tuned step for long reductions.
    block_k = max(base_k, min(static[2], triton.next_power_of_2(k)))
    # Keep the input and accumulator tiles within the tuned kernel's launch sizes.
    if (block_m > 128 or block_n > 256 or block_n % block_m
            or block_k > 512 or (block_m + block_n) * block_k > 128 * 1024):
        return static
    cube_cores, _ = _core_counts(device)
    if _cube_tasks(n, batch, block_m, block_n) < min(cube_cores, _cube_tasks(n, batch, static[0], static[1])):
        return static
    return block_m, block_n, block_k


@lru_cache(maxsize=128)
def _check_indexing_range(
    shape: tuple[int, int, int], blocks: tuple[int, int, int], core_counts: tuple[int, int],
) -> None:
    """Keep tile-local offsets and loop induction within signed 32-bit range."""
    n, k, batch = shape
    block_m, block_n, block_k = blocks
    cube_cores, vector_cores = core_counts
    cube_tiles = triton.cdiv(n, block_n)
    mirror_tiles = triton.cdiv(n, _MIRROR_BLOCK)
    last_n = cube_tiles * block_n - 1
    last_k = triton.cdiv(k, block_k) * block_k - 1
    last_output = max(last_n, mirror_tiles * _MIRROR_BLOCK - 1)
    cube_tasks = batch * (cube_tiles * (cube_tiles + 1) // 2) * (block_n // block_m)
    mirror_tasks = batch * (mirror_tiles * (mirror_tiles + 1) // 2)
    # Loads can form masked tail addresses, and the last loop increment can step past the task count.
    largest_index = max(
        last_n * k + last_k, last_k * n + last_n, last_output * (n + 1),
        cube_tasks + cube_cores - 1, mirror_tasks + vector_cores - 1,
    )
    if largest_index >= 2**31:
        raise ValueError("Shape exceeds the symmetric_matmul indexing range within a matrix or task grid")


def launch_symmetric_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Launch a validated product on its device's current stream.

    Args:
        a: Dense or transposed left NPU operand.
        b: Dense or transposed right NPU operand.
        out: Contiguous destination with the selected output dtype.
    """
    n, k = a.shape[-2:]
    batch = 1 if a.ndim == 2 else a.shape[0]
    shape = (n, k, batch)
    with torch_npu.npu.device(a.device):
        blocks = _block_shape(
            a.device, shape, (a.dtype, out.dtype), (not a.is_contiguous(), not b.is_contiguous()),
        )
        core_counts = _core_counts(a.device)
        _check_indexing_range(shape, blocks, core_counts)
        cores, vector_cores = core_counts
        stream = torch_npu.npu.current_stream(a.device)
        for tensor in (a, b, out):
            tensor.record_stream(stream)
        _symmetric_matmul_kernel[(cores,)](
            a, b, out, n, k, batch,
            0 if a.ndim == 2 else a.stride(0), a.stride(-2), a.stride(-1),
            0 if b.ndim == 2 else b.stride(0), b.stride(-2), b.stride(-1),
            *blocks, cores, num_stages=2,
        )
        _mirror_lower_kernel[(vector_cores,)](out, n, batch, vector_cores, _MIRROR_BLOCK)
