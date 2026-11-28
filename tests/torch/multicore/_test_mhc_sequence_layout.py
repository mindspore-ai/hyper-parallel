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
"""Compare native mHC on DTensor shards with full-sequence native execution."""

from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.components.modules.mhc import pipelined_mhc_post
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Check finite BF16 outputs and gradients using the mHC semantic tolerance."""
    actual = actual.detach().float().cpu().flatten()
    expected = expected.detach().float().cpu().flatten()
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError("Native mHC comparison contains non-finite values")
    difference = actual - expected
    relative = float(difference.norm() / expected.norm().clamp_min(1e-20))
    denominator = float(actual.norm() * expected.norm())
    cosine = float(actual.dot(expected)) / max(denominator, 1e-24) if denominator else 1.0
    result = {"max_abs": float(difference.abs().max()), "relative_l2": relative, "cosine": cosine}
    if relative > 2e-2 or cosine < 0.999:
        raise AssertionError(f"Native mHC shard/reference mismatch: {result}")
    return result


def _check_layout(axis: str, *, residual_only: bool) -> dict:
    """Verify one batch/sequence shard through two CANN calls and backward."""
    mesh = init_device_mesh("npu", (2,), mesh_dim_names=(axis,))
    rank = dist.get_rank()
    generator = torch.Generator().manual_seed(1623)
    values = [torch.randn(shape, generator=generator, dtype=dtype) for shape, dtype in (
        ((2, 256, 640), torch.bfloat16),
        ((2, 256, 4, 640), torch.bfloat16),
        ((2, 256, 4), torch.float32),
        ((2, 256, 4, 4), torch.float32),
    )]
    values[2] = values[2].sigmoid()
    values[3] = values[3].softmax(-1)
    dimension = 0 if axis == "dp" else 1
    interval = (slice(rank, rank + 1), slice(None)) if dimension == 0 else (
        slice(None), slice(rank * 128, (rank + 1) * 128)
    )
    local = [value[interval].clone().to("npu").requires_grad_() for value in values]
    placements = (Shard(dimension),)
    wrapped = [
        DTensor.from_local(value, mesh, placements) if not residual_only or index == 1 else value
        for index, value in enumerate(local)
    ]
    first = pipelined_mhc_post(*wrapped)
    second = pipelined_mhc_post(wrapped[0], first, wrapped[2], wrapped[3])
    reference = [value.to("npu").requires_grad_() for value in values]
    reference_first = pipelined_mhc_post(*reference)
    reference_second = pipelined_mhc_post(reference[0], reference_first, reference[2], reference[3])
    output_metrics = []
    for actual, expected in ((first, reference_first), (second, reference_second)):
        if (not isinstance(actual, DTensor) or actual.device_mesh is not mesh
                or actual.placements != placements or actual.shape != expected.shape):
            raise AssertionError(f"Native mHC lost {axis} layout; expected shape {expected.shape}")
        output_metrics.append(_metric(actual.to_local(), expected[interval]))
    second.to_local().square().sum().backward()
    reference_second[interval].square().sum().backward()
    names = ("output", "residual", "post", "combine")
    gradient_metrics = {}
    for name, actual, expected in zip(names, local, reference):
        if actual.grad is None or expected.grad is None:
            raise AssertionError(f"Native mHC gradient missing for {name}")
        gradient_metrics[name] = _metric(actual.grad, expected.grad[interval])
    return {"axis": axis, "residual_only": residual_only,
            "outputs": output_metrics, "gradients": gradient_metrics}


def test_mhc_sequence_layout_hccl() -> None:
    """Run actual CANN kernels with DP2, CP2 and TP2/SP layouts and mixed inputs."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        if dist.get_world_size() != 2:
            raise AssertionError(f"Expected two ranks, got {dist.get_world_size()}")
        cases = [_check_layout(axis, residual_only=mixed)
                 for axis in ("dp", "cp", "tp") for mixed in (False, True)]
        result = {"rank": dist.get_rank(), "passed": True, "cases": cases}
        output = os.environ.get("HP_MHC_SEQUENCE_REPORT")
        if output:
            Path(f"{output}.rank{dist.get_rank()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()
