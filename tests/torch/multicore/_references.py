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
"""Numerical fixtures and rank evidence for DeepSeek model integration workers."""

import hashlib
import json
import os
from pathlib import Path
import platform

import torch
import torch.distributed as dist
import torch_npu

import hyper_multicore
from hyper_multicore._loader import validate_native_compatibility

from tests.common.port_utils import allocate_port

_HIDDEN = 512
_INTERMEDIATE = 128


def _weights(experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate exactly representable weights shared by the device and CPU oracle."""
    generator = torch.Generator().manual_seed(9023)
    first = torch.randn(experts, _HIDDEN, 2 * _INTERMEDIATE, generator=generator) / _HIDDEN**0.5
    second = torch.randn(experts, _INTERMEDIATE, _HIDDEN, generator=generator) / _INTERMEDIATE**0.5
    return first.bfloat16(), second.bfloat16()


def _inputs(lengths: list[int], experts: int, top_k: int) -> tuple:
    """Make deterministic original-token inputs without synthetic routes."""
    states, indices, probabilities, upstream = [], [], [], []
    for rank, length in enumerate(lengths):
        generator = torch.Generator().manual_seed(11000 + rank + length)
        states.append((torch.randn(length, _HIDDEN, generator=generator) * 0.5).bfloat16())
        ids = torch.arange(length * top_k).reshape(length, top_k)
        indices.append(((ids + rank * top_k) % experts).int())
        probabilities.append(torch.softmax(torch.randn(length, top_k, generator=generator), dim=-1))
        upstream.append((torch.randn(length, _HIDDEN, generator=generator) * 0.1).bfloat16())
    return states, indices, probabilities, upstream


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Apply the unchanged common-MoE BF16 threshold to the rounded reference."""
    actual = actual.detach().float().cpu()
    expected = expected.float()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3, msg=lambda message: f"{name}: {message}")
    return (actual - expected).abs().max().item() if actual.numel() else 0.0


def start_shmem_lifetime() -> None:
    """Coordinate a fresh SHMEM endpoint for one MegaMoe scenario."""
    dist.barrier()
    endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(endpoint, src=0)
    os.environ["HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]


def environment_identity() -> dict:
    """Record the installed package and the native pair selected by this worker."""
    selected = validate_native_compatibility()
    variant = selected["variant"]
    paths = [selected["root"] / variant[name] for name in ("torch_so", "shmem_so")]
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "chip": torch.npu.get_device_name(),
        "hyper_multicore": str(Path(hyper_multicore.__file__).resolve()),
        "variant": variant["id"],
        "native_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }


def write_evidence(result: dict) -> None:
    """Persist every rank's acceptance data and the software identity."""
    rank_results = [None] * dist.get_world_size()
    dist.all_gather_object(rank_results, result)
    if dist.get_rank() == 0:
        path = Path(os.environ["HP_MEGA_MOE_LEVEL1_RESULT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        evidence = {"environment": environment_identity(), "ranks": rank_results}
        path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(f"MegaMoe Level1 evidence: {path}", flush=True)
